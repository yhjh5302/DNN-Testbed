import numpy as np
import tensorflow as tf
from tensorflow import keras

class AlexNet_layer(keras.Model):
    def __init__(self, name=None, layer_list=None):
        super(AlexNet_layer, self).__init__(name=name)
        self.layer_list = layer_list
        if 'features_1' in self.layer_list:
            self.resize = keras.layers.Resizing(height=224, width=224, interpolation='nearest', name='resize')
            self.features_1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(11,11), strides=4, activation='relu', padding='same', input_shape=(224,224,3)),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(3,3), strides=2),
            ], name='features_1')
            self.features_1.load_weights('./alexnet_features_1_weights')
        if 'features_2' in self.layer_list:
            self.features_2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=192, kernel_size=(5,5), strides=1, activation='relu', padding='same', input_shape=(27,27,64)),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(3,3), strides=2),
            ], name='features_2')
            self.features_2.load_weights('./alexnet_features_2_weights')
        if 'features_3' in self.layer_list:
            self.features_3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,192)),
                keras.layers.BatchNormalization(),
            ], name='features_3')
            self.features_3.load_weights('./alexnet_features_3_weights')
        if 'features_4' in self.layer_list:
            self.features_4 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,384)),
                keras.layers.BatchNormalization(),
            ], name='features_4')
            self.features_4.load_weights('./alexnet_features_4_weights')
        if 'features_5' in self.layer_list:
            self.features_5 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,256)),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(3,3), strides=2),
            ], name='features_5')
            self.features_5.load_weights('./alexnet_features_5_weights')
        if 'classifier_1' in self.layer_list:
            self.classifier_1 = keras.models.Sequential([
                keras.layers.Flatten(),
                keras.layers.Dense(4096, activation='relu', input_shape=(256*6*6,)),
                keras.layers.Dropout(0.5),
            ], name='classifier_1')
            self.classifier_1.load_weights('./alexnet_classifier_1_weights')
        if 'classifier_2' in self.layer_list:
            self.classifier_2 = keras.models.Sequential([
                keras.layers.Dense(4096, activation='relu', input_shape=(4096,)),
                keras.layers.Dropout(0.5),
            ], name='classifier_2')
            self.classifier_2.load_weights('./alexnet_classifier_2_weights')
        if 'classifier_3' in self.layer_list:
            self.classifier_3 = keras.models.Sequential([
                keras.layers.Dense(1000, activation='softmax', input_shape=(4096,)),
            ], name='classifier_3')
            self.classifier_3.load_weights('./alexnet_classifier_3_weights')

    def get_random_input(self):
        if 'features_1' in self.layer_list:
            return np.zeros((1,32,32,3))
        elif 'features_2' in self.layer_list:
            return np.zeros((1,27,27,64))
        elif 'features_3' in self.layer_list:
            return np.zeros((1,13,13,192))
        elif 'features_4' in self.layer_list:
            return np.zeros((1,13,13,384))
        elif 'features_5' in self.layer_list:
            return np.zeros((1,13,13,256))
        elif 'classifier_1' in self.layer_list:
            return np.zeros((1,6,6,256))
        elif 'classifier_2' in self.layer_list:
            return np.zeros(4096)
        elif 'classifier_3' in self.layer_list:
            return np.zeros(4096)

    def call(self, x):
        if 'features_1' in self.layer_list:
            x = self.resize(x)
            x = self.features_1(x)
        if 'features_2' in self.layer_list:
            x = self.features_2(x)
        if 'features_3' in self.layer_list:
            x = self.features_3(x)
        if 'features_4' in self.layer_list:
            x = self.features_4(x)
        if 'features_5' in self.layer_list:
            x = self.features_5(x)
        if 'classifier_1' in self.layer_list:
            x = self.classifier_1(x)
        if 'classifier_2' in self.layer_list:
            x = self.classifier_2(x)
        if 'classifier_3' in self.layer_list:
            x = self.classifier_3(x)
        return x