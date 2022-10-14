import numpy as np
import tensorflow as tf
from tensorflow import keras

class AlexNet(keras.Model):
    def __init__(self, name=None, layer_list=None):
        super(AlexNet, self).__init__(name=name)
        self.layer_list = layer_list
        if 'conv_1' in self.layer_list:
            self.conv_1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(11,11), strides=4, activation='relu', padding='same', input_shape=(224,224,3)),
                keras.layers.BatchNormalization(),
            ], name='conv_1')
            self.conv_1.load_weights('./alexnet_conv_1_weights')
        if 'maxpool_1' in self.layer_list:
            self.maxpool_1 = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
        if 'conv_2' in self.layer_list:
            self.conv_2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=192, kernel_size=(5,5), strides=1, activation='relu', padding='same', input_shape=(27,27,64)),
                keras.layers.BatchNormalization(),
            ], name='conv_2')
            self.conv_2.load_weights('./alexnet_conv_2_weights')
        if 'maxpool_2' in self.layer_list:
            self.maxpool_2 = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
        if 'conv_3' in self.layer_list:
            self.conv_3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,192)),
                keras.layers.BatchNormalization(),
            ], name='conv_3')
            self.conv_3.load_weights('./alexnet_conv_3_weights')
        if 'conv_4' in self.layer_list:
            self.conv_4 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,384)),
                keras.layers.BatchNormalization(),
            ], name='conv_4')
            self.conv_4.load_weights('./alexnet_conv_4_weights')
        if 'conv_5' in self.layer_list:
            self.conv_5 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,256)),
                keras.layers.BatchNormalization(),
            ], name='conv_5')
            self.conv_5.load_weights('./alexnet_conv_5_weights')
        if 'maxpool_3' in self.layer_list:
            self.maxpool_3 = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
            self.flatten = keras.layers.Flatten()
        if 'classifier_1' in self.layer_list:
            self.classifier_1 = keras.models.Sequential([
                keras.layers.Dense(4096, activation='relu', input_shape=(256*6*6,)),
            ], name='classifier_1')
            self.classifier_1.load_weights('./alexnet_classifier_1_weights')
        if 'classifier_2' in self.layer_list:
            self.classifier_2 = keras.models.Sequential([
                keras.layers.Dense(4096, activation='relu', input_shape=(4096,)),
            ], name='classifier_2')
            self.classifier_2.load_weights('./alexnet_classifier_2_weights')
        if 'classifier_3' in self.layer_list:
            self.classifier_3 = keras.models.Sequential([
                keras.layers.Dense(1000, activation='softmax', input_shape=(4096,)),
            ], name='classifier_3')
            self.classifier_3.load_weights('./alexnet_classifier_3_weights')

    def get_random_input(self):
        if 'conv_1' in self.layer_list:
            return np.zeros((1,224,224,3))
        if 'maxpool_1' in self.layer_list:
            return np.zeros((1,55,55,256))
        elif 'conv_2' in self.layer_list:
            return np.zeros((1,27,27,64))
        if 'maxpool_2' in self.layer_list:
            return np.zeros((1,27,27,256))
        elif 'conv_3' in self.layer_list:
            return np.zeros((1,13,13,192))
        elif 'conv_4' in self.layer_list:
            return np.zeros((1,13,13,384))
        elif 'conv_5' in self.layer_list:
            return np.zeros((1,13,13,256))
        if 'maxpool_3' in self.layer_list:
            return np.zeros((1,13,13,256))
        elif 'classifier_1' in self.layer_list:
            return np.zeros(6*6*256)
        elif 'classifier_2' in self.layer_list:
            return np.zeros(4096)
        elif 'classifier_3' in self.layer_list:
            return np.zeros(4096)

    def call(self, x):
        if 'conv_1' in self.layer_list:
            x = tf.image.resize(x, size=(224, 224), method='nearest')
            x = self.conv_1(x)
        if 'maxpool_1' in self.layer_list:
            x = self.maxpool_1(x)
        if 'conv_2' in self.layer_list:
            x = self.conv_2(x)
        if 'maxpool_2' in self.layer_list:
            x = self.maxpool_2(x)
        if 'conv_3' in self.layer_list:
            x = self.conv_3(x)
        if 'conv_4' in self.layer_list:
            x = self.conv_4(x)
        if 'conv_5' in self.layer_list:
            x = self.conv_5(x)
        if 'maxpool_3' in self.layer_list:
            x = self.maxpool_3(x)
            x = self.flatten(x)
        if 'classifier_1' in self.layer_list:
            x = self.classifier_1(x)
        if 'classifier_2' in self.layer_list:
            x = self.classifier_2(x)
        if 'classifier_3' in self.layer_list:
            x = self.classifier_3(x)
        return x