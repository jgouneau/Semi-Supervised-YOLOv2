import numpy as np
from tensorflow.keras import backend as K

class Loss(object):

    # y_true(nb_batch, x_grid, y_grid, nb_box) = [x, y, w, h, c, pseudo_lab, pl_conf, classes]
    # y_pred(nb_batch, x_grid, y_grid, nb_box) = [x, y, w, h, c, classes]

    def __init__(self, lambda_coord=5, lambda_noobj=0.5, lambda_obj=1, lambda_class=1, lambda_u = 0.9):
        self.__name__ = 'yolo_loss'

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class
        self.lambda_u = lambda_u

    def coord_loss(self, y_true, y_pred):
        b_xy_pred = y_pred[..., 0:2]
        b_wh_pred = y_pred[..., 2:4]

        b_xy_true = y_true[..., 0:2]
        b_wh_true = y_true[..., 2:4]

        indicator_obj = K.expand_dims(y_true[..., 4], axis=-1)

        loss_xy = K.sum(K.square(b_xy_true - b_xy_pred) * indicator_obj) * self.lambda_coord
        loss_wh = K.sum(K.square(K.sqrt(b_wh_true) - K.sqrt(b_wh_pred)) * indicator_obj) * self.lambda_coord

        return loss_wh + loss_xy

    def obj_loss(self, y_true, y_pred):
        obj_conf_true = y_true[..., 4]
        obj_conf_pred = y_pred[..., 4]

        #indicator_noobj = (1 - y_true[..., 4]) * self.lambda_noobj
        #indicator_obj = y_true[..., 4] * self.lambda_obj
        indicator_obj_noobj = obj_conf_true*(self.lambda_obj - self.lambda_noobj) + self.lambda_noobj #indicator_obj + indicator_noobj

        loss_obj = K.sum(K.square(obj_conf_true - obj_conf_pred) * indicator_obj_noobj)
        return loss_obj

    def class_loss(self, y_true, y_pred):
        p_c_pred = y_pred[..., 5:]
        p_c_true = y_true[..., 7:]

        indicator_obj = y_true[..., 4] * self.lambda_class

        loss_class = K.sum(K.sum(K.square(p_c_true - p_c_pred), axis=-1) * indicator_obj)

        return loss_class

    def __call__(self, y_true, y_pred):
        #total_coord_loss = self.coord_loss(y_true, y_pred)
        #total_obj_loss = self.obj_loss(y_true, y_pred)
        #total_class_loss = self.class_loss(y_true, y_pred)

        #loss = total_coord_loss + total_obj_loss + total_class_loss

        xy_pred = y_pred[..., 0:2]
        xy_true = y_true[..., 0:2]
        sqrt_wh_pred = K.sqrt(y_pred[..., 2:4])
        sqrt_wh_true = K.sqrt(y_true[..., 2:4])

        conf_true = y_true[..., 5]
        conf_pred = y_pred[..., 5]

        p_c_pred = y_pred[..., 5:]
        p_c_true = y_true[..., 7:]

        ind_supervised = y_true[..., 4]
        ind_obj = y_true[..., 5]
        ind_maybe_noobj = 1-y_true[..., 6]

        ind_sup_unsup = ind_supervised*(1-self.lambda_u) + self.lambda_u
        ind_coord = self.lambda_coord*ind_obj
        ind_conf = ind_obj*(self.lambda_obj - self.lambda_noobj*ind_maybe_noobj) + self.lambda_noobj * ind_maybe_noobj
        ind_class = self.lambda_class*ind_obj

        l_coord = ind_coord*K.sum(K.square(xy_true - xy_pred) + K.square(sqrt_wh_true - sqrt_wh_pred), axis=-1)
        l_conf = ind_conf*K.square(conf_true - conf_pred)
        l_class = ind_class*K.sum(K.square(p_c_true - p_c_pred), axis=-1)

        l_tot = l_coord + l_conf + l_class
        loss = K.sum(ind_sup_unsup*l_tot)

        return loss