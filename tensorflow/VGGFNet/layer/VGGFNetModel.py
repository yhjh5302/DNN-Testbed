import numpy as np
import tensorflow as tf
from tensorflow import keras

class VGGFNet_layer(keras.Model):
    def __init__(self, name=None, layer_list=None):
        super(VGGFNet_layer, self).__init__(name=name)
        self.layer_list = layer_list
        if 'features1' in self.layer_list:
            self.resize = keras.layers.Resizing(height=224, width=224, interpolation='nearest', name='resize')
            self.features1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(11,11), strides=4, activation='relu'),
                keras.layers.AveragePooling2D(pool_size=(1, 1), strides=1),
                keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
            ], name='features1')
            self.features1.load_weights('./VGGFNet_features1_weights')
        if 'features2' in self.layer_list:
            self.features2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=1, activation='relu'),
                keras.layers.AveragePooling2D(pool_size=(1, 1), strides=1),
                keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
            ], name='features2')
            self.features2.load_weights('./VGGFNet_features2_weights')
        if 'features3' in self.layer_list:
            self.features3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            ], name='features3')
            self.features3.load_weights('./VGGFNet_features3_weights')
        if 'features4' in self.layer_list:
            self.features4 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            ], name='features4')
            self.features4.load_weights('./VGGFNet_features4_weights')
        if 'features5' in self.layer_list:
            self.features5 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
            ], name='features5')
            self.features5.load_weights('./VGGFNet_features5_weights')
        if 'classifier1' in self.layer_list:
            self.flatten = keras.layers.Flatten()
            self.classifier1 = keras.models.Sequential([
                keras.layers.Dense(4096, activation='relu'),
                keras.layers.Dropout(0.5),
            ], name='classifier1')
            self.classifier1.load_weights('./VGGFNet_classifier1_weights')
        if 'classifier2' in self.layer_list:
            self.classifier2 = keras.models.Sequential([
                keras.layers.Dense(4096, activation='relu'),
                keras.layers.Dropout(0.5),
            ], name='classifier2')
            self.classifier2.load_weights('./VGGFNet_classifier2_weights')
        if 'classifier3' in self.layer_list:
            self.classifier3 = keras.models.Sequential([
                keras.layers.Dense(1000),
            ], name='classifier3')
            self.classifier3.load_weights('./VGGFNet_classifier3_weights')

    def get_random_input(self):
        if 'features1' in self.layer_list:
            return np.zeros((1,32,32,3))
        elif 'features2' in self.layer_list:
            return np.zeros((1,27,27,64))
        elif 'features3' in self.layer_list:
            return np.zeros((1,11,11,256))
        elif 'features4' in self.layer_list:
            return np.zeros((1,11,11,256))
        elif 'features5' in self.layer_list:
            return np.zeros((1,11,11,256))
        elif 'classifier1' in self.layer_list:
            return np.zeros((1,5,5,256))
        elif 'classifier2' in self.layer_list:
            return np.zeros(4096)
        elif 'classifier3' in self.layer_list:
            return np.zeros(4096)

    def call(self, x):
        if 'features1' in self.layer_list:
            x = self.resize(x)
            x = self.features1(x)
        if 'features2' in self.layer_list:
            x = self.features2(x)
        if 'features3' in self.layer_list:
            x = self.features3(x)
        if 'features4' in self.layer_list:
            x = self.features4(x)
        if 'features5' in self.layer_list:
            x = self.features5(x)
        if 'classifier1' in self.layer_list:
            x = self.flatten(x)
            x = self.classifier1(x)
        if 'classifier2' in self.layer_list:
            x = self.classifier2(x)
        if 'classifier3' in self.layer_list:
            x = self.classifier3(x)
        return x