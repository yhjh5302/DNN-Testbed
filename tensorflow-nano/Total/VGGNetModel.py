import numpy as np
import tensorflow as tf
from tensorflow import keras

class VGGNet_layer(keras.Model):
    def __init__(self, name=None, layer_list=None):
        super(VGGNet_layer, self).__init__(name=name)
        self.layer_list = layer_list
        if 'features1' in self.layer_list:
            self.features1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
                keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
                keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
            ], name='features1')
            # self.features1.load_weights('./VGGNet_features1_weights')
        if 'features2' in self.layer_list:
            self.features2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
                keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'),
            ], name='features2')
            # self.features2.load_weights('./VGGNet_features2_weights')
        if 'features3' in self.layer_list:
            self.features3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'),
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'),
                keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
            ], name='features3')
            # self.features3.load_weights('./VGGNet_features3_weights')
        if 'features4' in self.layer_list:
            self.features4 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'),
                keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
            ], name='features4')
            # self.features4.load_weights('./VGGNet_features4_weights')
        if 'features5' in self.layer_list:
            self.features5 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'),
                keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
            ], name='features5')
            # self.features5.load_weights('./VGGNet_features5_weights')
        if 'classifier1' in self.layer_list:
            self.flatten = keras.layers.Flatten()
            self.classifier1 = keras.models.Sequential([
                keras.layers.Dense(4096, activation='relu'),
                keras.layers.Dropout(0.5),
            ], name='classifier1')
            # self.classifier1.load_weights('./VGGNet_classifier1_weights')
        if 'classifier2' in self.layer_list:
            self.classifier2 = keras.models.Sequential([
                keras.layers.Dense(4096, activation='relu'),
                keras.layers.Dropout(0.5),
            ], name='classifier2')
            # self.classifier2.load_weights('./VGGNet_classifier2_weights')
        if 'classifier3' in self.layer_list:
            self.classifier3 = keras.models.Sequential([
                keras.layers.Dense(1000),
            ], name='classifier3')
            # self.classifier3.load_weights('./VGGNet_classifier3_weights')

    def get_random_input(self):
        if 'features1' in self.layer_list:
            return np.zeros((1,32,32,3))
        elif 'features2' in self.layer_list:
            return np.zeros((1,112,112,64))
        elif 'features3' in self.layer_list:
            return np.zeros((1,56,56,128))
        elif 'features4' in self.layer_list:
            return np.zeros((1,28,28,256))
        elif 'features5' in self.layer_list:
            return np.zeros((1,14,14,512))
        elif 'classifier1' in self.layer_list:
            return np.zeros((1,7,7,512))
        elif 'classifier2' in self.layer_list:
            return np.zeros(4096)
        elif 'classifier3' in self.layer_list:
            return np.zeros(4096)

    def call(self, x):
        if 'features1' in self.layer_list:
            x = tf.image.resize(x, size=(224,224), method='nearest')
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