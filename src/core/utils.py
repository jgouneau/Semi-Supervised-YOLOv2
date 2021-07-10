import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import os

import xml.etree.ElementTree as et
from tqdm import tqdm

import cv2

#############################
# YOLO
#############################

def normalize(image):
    image = image / 255.
    image = image - 0.5
    image = image * 2.

    return image

def decode_netout(netout, nb_class, obj_threshold=0.5, nms_threshold=0.3):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]
                confidence = netout[row, col, b, 4]

                if confidence >= obj_threshold:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]

                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence, classes)
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_i].classes[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes


#############################
# Loss
#############################

def calculate_ious(a1, a2):

    def process_boxes(a):
        # ALign x-w, y-h
        a_xy = a[..., 0:2]
        a_wh = a[..., 2:4]

        a_wh_half = a_wh / 2.
        # Get x_min, y_min
        a_mins = a_xy - a_wh_half
        # Get x_max, y_max
        a_maxes = a_xy + a_wh_half

        return a_mins, a_maxes, a_wh

    # Process two sets
    a2_mins, a2_maxes, a2_wh = process_boxes(a2)
    a1_mins, a1_maxes, a1_wh = process_boxes(a1)

    # Intersection as min(Upper1, Upper2) - max(Lower1, Lower2)
    intersect_mins = K.maximum(a2_mins, a1_mins)
    intersect_maxes = K.minimum(a2_maxes, a1_maxes)

    # Getting the intersections in the xy (aka the width, height intersection)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)

    # Multiply to get intersecting area
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Values for the single sets
    true_areas = a1_wh[..., 0] * a1_wh[..., 1]
    pred_areas = a2_wh[..., 0] * a2_wh[..., 1]

    # Compute union for the IoU
    union_areas = pred_areas + true_areas - intersect_areas
    return intersect_areas / union_areas

def generate_yolo_grid(grid_size, nb_box):
    cell_x = tf.cast(tf.reshape(tf.tile(tf.range(grid_size[0]), [grid_size[1]]), (grid_size[1], grid_size[0],
                                                                                  1, 1)), tf.float32)
    cell_y = tf.cast(tf.reshape(tf.tile(tf.range(grid_size[1]), [grid_size[0]]), (grid_size[0], grid_size[1],
                                                                                  1, 1)), tf.float32)
    cell_y = tf.transpose(cell_y, (1, 0, 2, 3))

    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [1, 1, nb_box, 1])
    return cell_grid


#############################
# Dataset
#############################

def list_images(base_path, valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):
    # return the set of files that are valid
    return list_files(base_path, valid_exts, contains=contains)

def list_files(base_path, valid_exts="", contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(base_path):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(valid_exts):
                # construct the path to the image and yield it
                image_path = os.path.join(rootDir, filename)
                yield image_path


#############################
# Preprocessing
#############################

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]*self.c
        return self.score

    def __repr__(self):
        """
        Helper method for printing the object's values
        :return:
        """
        return "<BoundBox({}, {}, {}, {}, {}, {})>\n".format(
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.get_label(),
            self.get_score()
        )

def bbox_iou(box1, box2):

    def _interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union

def parse_annotation_xml(ann_dir, img_dir, labels=[]):
    # This parser is utilized on VOC dataset
    all_imgs = []
    seen_labels = {}

    ann_files = os.listdir(ann_dir)
    for ann in tqdm(sorted(ann_files)):
        img = {'object': []}

        tree = et.parse(os.path.join(ann_dir, ann))

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        #if len(img['object']) > 0:
        all_imgs += [img]

    return all_imgs, seen_labels


#############################
# mAP Evaluation
#############################

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


#############################
# GPU
#############################

def enable_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


#############################
# Prediction
#############################

def load_image(data, i):
    image = cv2.imread(data[i]['filename'])
    return image

def draw_boxes(image, boxes, labels):
    image_h, image_w, _ = image.shape

    color_levels = [0, 255, 128, 64, 32]
    colors = []
    for r in color_levels:
        for g in color_levels:
            for b in color_levels:
                if r == g and r == b:  # prevent grayscale colors
                    continue
                colors.append((b, g, r))

    for box in boxes:
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)

        line_width_factor = int(min(image_h, image_w) * 0.005)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[box.get_label()], line_width_factor * 2)
        cv2.putText(image, "{} {:.3f}".format(labels[box.get_label()], box.get_score()),
                    (xmin, ymin - line_width_factor * 3), cv2.FONT_HERSHEY_PLAIN, 2e-3 * min(image_h, image_w),
                    colors[box.get_label()], line_width_factor)

    return image