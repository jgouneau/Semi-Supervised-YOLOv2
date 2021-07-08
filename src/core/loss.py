import numpy as np
from tensorflow.keras import backend as K

class Loss(object):

    def __init__(self, lambda_coord=5, lambda_noobj=0.5, lambda_obj=1, lambda_class=1):
        self.__name__ = 'yolo_loss'

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class

    def coord_loss(self, y_true, y_pred):
        b_xy_pred = y_pred[..., :2]
        b_wh_pred = y_pred[..., 2:4]

        b_xy_true = y_true[..., 0:2]
        b_wh_true = y_true[..., 2:4]

        indicator_coord = K.expand_dims(y_true[..., 4], axis=-1)

        loss_xy = K.sum(K.square(b_xy_true - b_xy_pred) * indicator_coord) * self.lambda_coord
        loss_wh = K.sum(K.square(K.sqrt(b_wh_true) - K.sqrt(b_wh_pred)) * indicator_coord) * self.lambda_coord

        return loss_wh + loss_xy

    def obj_loss(self, y_true, y_pred):
        # TODO: should make a review in this part
        obj_conf_true = y_true[..., 4]
        obj_conf_pred = y_pred[..., 4]

        indicator_noobj = (1 - y_true[..., 4]) * self.lambda_noobj #* K.cast(best_ious < self.iou_filter, np.float32)
        indicator_obj = y_true[..., 4] * self.lambda_obj
        indicator_obj_noobj = indicator_obj + indicator_noobj

        loss_obj = K.sum(K.square(obj_conf_true - obj_conf_pred) * indicator_obj_noobj)
        return loss_obj

    def class_loss(self, y_true, y_pred):
        p_c_pred = K.softmax(y_pred[..., 5:])
        p_c_true = K.one_hot(K.argmax(y_true[..., 5:], axis=-1), y_pred.shape[4] - 5)
        loss_class_arg = K.sum(K.square(p_c_true - p_c_pred), axis=-1)

        indicator_class = y_true[..., 4] * self.lambda_class

        loss_class = K.sum(loss_class_arg * indicator_class)

        return loss_class

    def __call__(self, y_true, y_pred):
        total_coord_loss = self.coord_loss(y_true, y_pred)
        total_obj_loss = self.obj_loss(y_true, y_pred)
        total_class_loss = self.class_loss(y_true, y_pred)

        loss = total_coord_loss + total_obj_loss + total_class_loss

        return loss