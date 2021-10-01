import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Reshape, Conv2D, Input, concatenate
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import os
import cv2

from .utils import normalize, decode_netout, generate_yolo_grid
from .preprocessing import BatchGenerator, SSLBatchGenerator
from .loss import Loss
from .scheduler import WarmupScheduler

class YOLO(object):

  def __init__(self, name, backend, input_size, labels, anchors, config):
        self.name = name
        self._input_size = input_size
        self.labels = list(labels)
        self._nb_class = len(self.labels)
        self._nb_box = len(anchors) // 2
        self._anchors = anchors

        ##########################
        # Make the model
        ##########################

        backend_path = config['backend_paths'][backend]

        # make the feature extractor layers
        self._input_size = (self._input_size[0], self._input_size[1], 3)
        input_image = Input(shape=self._input_size)

        mobilenet = MobileNet(input_shape=self._input_size, include_top=False)
        try:
            print("Loading pretrained weights: " + backend_path)
            mobilenet.load_weights(backend_path)
        except:
            print("Unable to load backend weights. Using a fresh model")
        x = mobilenet(input_image)
        self._feature_extractor = Model(input_image, x, name='MobileNet_backend')
        self._feature_extractor.trainable = False

        self._grid_h, self._grid_w = self._feature_extractor.output_shape[1:3]
        features = self._feature_extractor(input_image)

        # make the object detection layer
        out_xy = Conv2D(self._nb_box * 2,
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='Detection_layer_0',
                        kernel_initializer='lecun_normal')(features)
        out_xy = Reshape((self._grid_h, self._grid_w, self._nb_box, 2))(out_xy)
        out_xy = K.sigmoid(out_xy)
        c_grid = generate_yolo_grid([self._grid_h, self._grid_w], self._nb_box)
        out_xy = (out_xy + c_grid) / [self._grid_w, self._grid_h]

        out_wh = Conv2D(self._nb_box * 2,
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='Detection_layer_1',
                        kernel_initializer='lecun_normal')(features)
        out_wh = Reshape((self._grid_h, self._grid_w, self._nb_box, 2))(out_wh)
        out_wh = K.exp(out_wh)
        anch = tf.reshape(self._anchors, (self._nb_box, 2))
        out_wh = (out_wh * anch)

        out_c = Conv2D(self._nb_box * 1,
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='Detection_layer_2',
                        kernel_initializer='lecun_normal')(features)
        out_c = Reshape((self._grid_h, self._grid_w, self._nb_box, 1))(out_c)
        out_c = K.sigmoid(out_c)

        out_p = Conv2D(self._nb_box * self._nb_class,
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='Detection_layer_3',
                        kernel_initializer='lecun_normal')(features)
        out_p = Reshape((self._grid_h, self._grid_w, self._nb_box, self._nb_class))(out_p)
        out_p = K.softmax(out_p)

        output = concatenate([out_xy, out_wh, out_c, out_p], name="YOLO_output")
        self._model = Model(input_image, output, name=self.name)

        # initialize the weights of the detection layer
        for i in range(4):
          layer = self._model.get_layer("Detection_layer_" + str(i))
          weights = layer.get_weights()
          new_kernel = np.random.normal(size=weights[0].shape) / (self._grid_h * self._grid_w)
          new_bias = np.random.normal(size=weights[1].shape) / (self._grid_h * self._grid_w)
          layer.set_weights([new_kernel, new_bias])

        # create the associated folder
        self._agent_path = config['agents_paths'][self.name] + "/"
        self._weights_path = self._agent_path + "weights/"
        self._temp_path = self._agent_path + "temp/"
        os.makedirs(self._agent_path, exist_ok=True)
        os.makedirs(self._weights_path, exist_ok=True)
        os.makedirs(self._temp_path, exist_ok=True)

        # print a summary of the whole model
        print(self._model.summary())

        for i in range(self._nb_box):
          self._anchors[2*i] = self._anchors[2*i] * self._grid_w
          self._anchors[2*i+1] = self._anchors[2*i+1] * self._grid_h

  def load_weights(self, name):
        self._model.load_weights(self._weights_path + name)
    
  def train(self, train_data,
              valid_data,
              nb_epochs,
              batch_size,
              learning_rate,
              lamb_obj,
              lamb_noobj,
              lamb_coord,
              lamb_class,
              lamb_u,
              warmup_epochs,
              pseudo_lab_data=None,
              pseudo_lab_batch_size=0,
              workers=3,
              max_queue_size=8,
              early_stop=True,
              tb_logdir="./"):
              
        #######################################
        # Make train and validation generators
        #######################################
        generator_config = {
            'IMAGE_H': self._input_size[0],
            'IMAGE_W': self._input_size[1],
            'IMAGE_C': self._input_size[2],
            'GRID_H': self._grid_h,
            'GRID_W': self._grid_w,
            'BOX': self._nb_box,
            'LABELS': self.labels,
            'CLASS': len(self.labels),
            'ANCHORS': self._anchors,
            'BATCH_SIZE': batch_size,
            'LAB_BATCH_SIZE': batch_size,
            'PSEUDO_LAB_BATCH_SIZE': pseudo_lab_batch_size
        }
        if pseudo_lab_data == None:
            train_generator = BatchGenerator(train_data,
                                         generator_config,
                                         jitter=True)
        else:
            train_generator = SSLBatchGenerator(train_data,
                                         pseudo_lab_data,
                                         generator_config,
                                         jitter=True)
        valid_generator = BatchGenerator(valid_data,
                                         generator_config,
                                         jitter=False)

        ############################################
        # Compile the model
        ############################################
        opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        loss = Loss(lambda_coord=lamb_coord, lambda_noobj=lamb_noobj, lambda_obj=lamb_obj, lambda_class=lamb_class, lambda_u=lamb_u)
        self._model.compile(loss=loss, optimizer=opt)

        ############################################
        # Make a few callbacks
        ############################################
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience = 5, min_lr = 0.000001, verbose = 1)
        tensorboard_cb = TensorBoard(log_dir=tb_logdir,
                                     histogram_freq=0,
                                     write_graph=True,
                                     write_images=False)
        save_best_loss = ModelCheckpoint(self._weights_path + "bestLoss.h5",
                                        monitor='loss',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='min',
                                        save_freq='epoch')
        warmup = WarmupScheduler(warmup_epochs, learning_rate)
        # save_best_map = MapEvaluation(self, valid_generator,
        #                               save_best=False,
        #                               save_name=root + "_bestMap" + ext,
        #                               tensorboard=tensorboard_cb,
        #                               iou_threshold=iou_threshold,
        #                               score_threshold=score_threshold)
        callbacks = [reduce_lr, save_best_loss, warmup]

        #############################
        # Start the training process
        #############################
        self._model.fit(train_generator,
                        epochs=nb_epochs,
                        validation_data=valid_generator,
                        callbacks=callbacks,
                        workers=workers,
                        max_queue_size=max_queue_size)
        
  def predict(self, image, obj_threshold, nms_threshold):
        if len(image.shape) == 2 and not self._gray_mode:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = image[..., ::-1]  # make it RGB (it is important for normalization of some backends)

        image = cv2.resize(image, (self._input_size[1], self._input_size[0]))
        image = normalize(image)
        
        if len(image.shape) == 3:
            input_image = image[np.newaxis, :]
        else:
            input_image = image[np.newaxis, ..., np.newaxis]
        netout = self._model.predict(input_image)[0]
        boxes = decode_netout(netout, self._nb_class, obj_threshold, nms_threshold)

        return boxes