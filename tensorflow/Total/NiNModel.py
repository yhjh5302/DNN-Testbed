import numpy as np
import tensorflow as tf
from tensorflow import keras


def nin_block(num_channels, kernel_size, strides, padding, name=None):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides,
                               padding=padding, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu')], name= name)


class NiN_layer(keras.Model):
    def __init__(self, name=None, layer_list=None):
        super(NiN_layer, self).__init__(name=name)
        self.layer_list = layer_list
        if 'features_1' in self.layer_list:
            self.resize = keras.layers.Resizing(height=224, width=224, interpolation='nearest', name='resize')
            self.features_1 = tf.keras.models.Sequential([
                nin_block(96, kernel_size=11, strides=4, padding='valid'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            ])
        if 'features_2' in self.layer_list:
            self.features_2 = keras.models.Sequential([
                nin_block(256, kernel_size=5, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            ], name='features_2')
        if 'features_3' in self.layer_list:
            self.features_3 = keras.models.Sequential([
                nin_block(384, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Dropout(0.5),
            ], name='features_3')
        if 'feature_4' in self.layer_list:
            self.features_4 = keras.models.Sequential([
                nin_block(10, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Reshape((1, 1, 10)),
            ], name='features_4')

    def get_random_input(self):
        if 'features_1' in self.layer_list:
            return np.zeros((1,224,224,3))
        elif 'features_2' in self.layer_list:
            return np.zeros((1, 28, 28, 96))
        elif 'features_3' in self.layer_list:
            return np.zeros((1, 14, 14, 256))
        elif 'features_4' in self.layer_list:
            return np.zeros((1, 7, 7, 384))

    def call(self, x):
        if 'features_1' in self.layer_list:
            x = self.resize(x)
            x = self.features_1(x)
        if 'features_2' in self.layer_list:
            x = self.features_2(x)
        if 'features_3' in self.layer_list:
            x = self.features_3(x)
        if 'features_4' in self.layer_list:
            x = self.feature_4(x)
        return x