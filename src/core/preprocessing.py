import numpy as np

from tensorflow.keras.utils import Sequence
import imgaug
from imgaug import augmenters as iaa
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage

import copy

import cv2

from .utils import BoundBox, bbox_iou, normalize

class BatchGenerator(Sequence):
    def __init__(self, data, config, shuffle=True, jitter=True):

        self._data = data
        self._config = config

        self._shuffle = shuffle
        self._jitter = jitter

        self._anchors = [BoundBox(0, 0, config['ANCHORS'][2 * i], config['ANCHORS'][2 * i + 1])
                         for i in range(int(len(config['ANCHORS']) // 2))]

        # augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self._aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.0),  # vertically flip 20% of all images
                sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent
                    rotate=(-20, 20),  # rotate by -45 to +45 degrees
                    shear=(-5, 5),  # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=imgaug.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means (kernel sizes between 2 and 7)
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians (kernel sizes between 2 and 7)
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                               # search either for all edges or for directed edges
                               sometimes(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0, 0.7)),
                                   iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                               ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Invert(0.05, per_channel=True),  # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),  # change brightness of images
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

        if shuffle:
            np.random.shuffle(self._data)
    
    def __len__(self):
        return int(np.ceil(float(len(self._data)) / self._config['BATCH_SIZE']))

    def num_classes(self):
        return len(self._config['LABELS'])

    def size(self):
        return len(self._data)

    def load_data(self, i):
      return self._data[i]

    def load_annotation(self, i):
        annots = []

        for obj in self._data[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self._config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        if self._config['IMAGE_C'] == 1:
            image = cv2.imread(self._data[i]['filename'], cv2.IMREAD_GRAYSCALE)
            image = image[..., np.newaxis]
        elif self._config['IMAGE_C'] == 3:
            image = cv2.imread(self._data[i]['filename'])
        else:
            raise ValueError("Invalid number of image channels.")
        return image
    
    def __getitem__(self, idx):
        l_bound = idx * self._config['BATCH_SIZE']
        r_bound = (idx + 1) * self._config['BATCH_SIZE']

        if r_bound > len(self._data):
            r_bound = len(self._data)
            l_bound = r_bound - self._config['BATCH_SIZE']

        instance_count = 0
        x_batch = np.zeros((r_bound - l_bound, self._config['IMAGE_H'], self._config['IMAGE_W'],
                            self._config['IMAGE_C']))  # input images

        y_batch = np.zeros((r_bound - l_bound, self._config['GRID_H'], self._config['GRID_W'], self._config['BOX'],
                            4 + 1 + len(self._config['LABELS'])))  # desired network output

        anchors_populated_map = np.zeros((r_bound - l_bound, self._config['GRID_H'], self._config['GRID_W'],
                                          self._config['BOX']))


        for train_instance in self._data[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, jitter=self._jitter)

            for obj in all_objs:
                # check if it is a valid annotion
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self._config['LABELS']:
                    scale_w = float(self._config['IMAGE_W']) #/ self._config['GRID_W']
                    scale_h = float(self._config['IMAGE_H']) #/ self._config['GRID_H']
                    # get which grid cell it is from
                    obj_center_x = (obj['xmin'] + obj['xmax']) / 2
                    obj_center_x = obj_center_x / scale_w
                    obj_center_y = (obj['ymin'] + obj['ymax']) / 2
                    obj_center_y = obj_center_y / scale_h

                    obj_grid_x = int(np.floor(obj_center_x * self._config['GRID_W']))
                    obj_grid_y = int(np.floor(obj_center_y * self._config['GRID_H']))

                    if obj_grid_x < self._config['GRID_W'] and obj_grid_y < self._config['GRID_H']:
                        obj_indx = self._config['LABELS'].index(obj['name'])

                        obj_w = (obj['xmax'] - obj['xmin']) / scale_w
                        obj_h = (obj['ymax'] - obj['ymin']) / scale_h

                        box = [obj_center_x, obj_center_y, obj_w, obj_h]

                        # find the anchor that best predicts this box
                        # TODO: check if this part below is working correctly
                        best_anchor_idx = -1
                        max_iou = -1

                        shifted_box = BoundBox(0, 0, obj_w, obj_h)

                        for i in range(len(self._anchors)):
                            anchor = self._anchors[i]
                            iou = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor_idx = i
                                max_iou = iou

                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        self._change_obj_position(y_batch, anchors_populated_map,
                                                  [instance_count, obj_grid_y, obj_grid_x, best_anchor_idx, obj_indx],
                                                  box, max_iou)

            # assign input image to x_batch
            x_batch[instance_count] = normalize(img)
            # increase instance counter in current batch
            instance_count += 1

        return x_batch, y_batch
    
    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._data)
    
    def _change_obj_position(self, y_batch, anchors_map, idx, box, iou):
        bkp_box = y_batch[idx[0], idx[1], idx[2], idx[3], 0:4].copy()
        anchors_map[idx[0], idx[1], idx[2], idx[3]] = iou
        y_batch[idx[0], idx[1], idx[2], idx[3], 0:4] = box
        y_batch[idx[0], idx[1], idx[2], idx[3], 4] = 1.
        y_batch[idx[0], idx[1], idx[2], idx[3], 5:] = 0  # clear old values
        y_batch[idx[0], idx[1], idx[2], idx[3], 4 + 1 + idx[4]] = 1

        shifted_box = BoundBox(0, 0, bkp_box[2], bkp_box[3])

        for i in range(len(self._anchors)):
            anchor = self._anchors[i]
            iou = bbox_iou(shifted_box, anchor)
            if iou > anchors_map[idx[0], idx[1], idx[2], i]:
                self._change_obj_position(y_batch, anchors_map, [idx[0], idx[1], idx[2], i, idx[4]], bkp_box, iou)
                break
    
    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)

        if image is None:
            print('Cannot find ', image_name)

        h = image.shape[0]
        w = image.shape[1]
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            bbs = []
            for i, obj in enumerate(all_objs):
                xmin = obj['xmin']
                ymin = obj['ymin']
                xmax = obj['xmax']
                ymax = obj['ymax']
                # use label field to later match it with final boxes
                bbs.append(BoundingBox(x1=xmin, x2=xmax, y1=ymin, y2=ymax, label=i))
            bbs = BoundingBoxesOnImage(bbs, shape=image.shape)
            image, bbs = self._aug_pipe(image=image, bounding_boxes=bbs)
            bbs = bbs.remove_out_of_image().clip_out_of_image()

            if len(bbs.bounding_boxes) < len(all_objs):
                print("Some boxes were removed during augmentations.")

            filtered_objs = []
            for bb in bbs.bounding_boxes:
                obj = all_objs[bb.label]
                obj['xmin'] = bb.x1
                obj['xmax'] = bb.x2
                obj['ymin'] = bb.y1
                obj['ymax'] = bb.y2
                filtered_objs.append(obj)
            all_objs = filtered_objs

            # resize the image to standard size
        image = cv2.resize(image, (self._config['IMAGE_W'], self._config['IMAGE_H']))
        if self._config['IMAGE_C'] == 1:
            image = image[..., np.newaxis]
            image = image[..., ::-1]  # make it RGB (it is important for normalization of some backends)

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                obj[attr] = int(obj[attr] * float(self._config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self._config['IMAGE_W']), 0)

            for attr in ['ymin', 'ymax']:
                obj[attr] = int(obj[attr] * float(self._config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self._config['IMAGE_H']), 0)

        return image, all_objs