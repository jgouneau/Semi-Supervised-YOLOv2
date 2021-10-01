from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np

class WarmupScheduler(keras.callbacks.Callback):

    def __init__(self, warmup_ep=2, init_lr=1e-3):
        super(WarmupScheduler, self).__init__()
        self.warmup_epochs = warmup_ep
        self.init_learning_rate = init_lr
        self.global_step = 0

    def on_epoch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1

    def on_epoch_begin(self, batch, logs=None):
        if self.global_step == self.warmup_epochs:
            print("End of warmup")
            self.model.trainable = True
            #K.set_value(self.model.optimizer.learning_rate, self.init_learning_rate/10)
