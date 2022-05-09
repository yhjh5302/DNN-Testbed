import numpy as np
import tensorflow as tf
from tensorflow import keras

class AlexNet_layer(keras.Model):
    def __init__(self, name=None, layer_list=None):
        super(AlexNet_layer, self).__init__(name=name)
        self.layer_list = layer_list
        if 'features_1_1' in self.layer_list:            
            self.features_1_1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=48, kernel_size=(11,11), strides=4, activation='relu', padding='same', input_shape=(224,224,3)),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(3,3), strides=2),
            ], name='features_1_1')
            # self.features_1_1.load_weights('./alexnet_features_1_1_weights')
        
        if 'features_1_2' in self.layer_list:            
            self.features_1_2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=48, kernel_size=(11,11), strides=4, activation='relu', padding='same', input_shape=(224,224,3)),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(3,3), strides=2),
            ], name='features_1_2')
            # self.features_1_2.load_weights('./alexnet_features_1_2_weights')

        if 'features_2_1' in self.layer_list:
            self.features_2_1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=1, activation='relu', padding='same', input_shape=(27,27,48)),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(3,3), strides=2),
            ], name='features_2_1')
            # self.features_2_1.load_weights('./alexnet_features_2_1_weights')

        if 'features_2_2' in self.layer_list:
            self.features_2_2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=1, activation='relu', padding='same', input_shape=(27,27,48)),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(3,3), strides=2),
            ], name='features_2_2')
            # self.features_2_2.load_weights('./alexnet_features_2_2_weights')
        
        if 'features_3_1' in self.layer_list:
            self.features_3_1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,128)),
                keras.layers.BatchNormalization(),
            ], name='features_3_1')
            # self.features_3_1.load_weights('./alexnet_features_3_1_weights')

        if 'features_3_2' in self.layer_list:
            self.features_3_2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,128)),
                keras.layers.BatchNormalization(),
            ], name='features_3_2')
            # self.features_3_2.load_weights('./alexnet_features_3_2_weights')

        if 'features_4_1' in self.layer_list:
            self.features_4_1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,192)),
                keras.layers.BatchNormalization(),
            ], name='features_4_1')
            # self.features_4_1.load_weights('./alexnet_features_4_1_weights')
        
        if 'features_4_2' in self.layer_list:
            self.features_4_2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,192)),
                keras.layers.BatchNormalization(),
            ], name='features_4_2')
            # self.features_4_2.load_weights('./alexnet_features_4_2_weights')

        if 'features_5_1' in self.layer_list:
            self.features_5_1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,128)),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(3,3), strides=2),
            ], name='features_5_1')
            # self.features_5_1.load_weights('./alexnet_features_5_1_weights')
        
        if 'features_5_2' in self.layer_list:
            self.features_5_2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,128)),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(3,3), strides=2),
            ], name='features_5_2')
            # self.features_5_2.load_weights('./alexnet_features_5_2_weights')

        if 'classifier_1_1' in self.layer_list:
            self.classifier_1_1 = keras.models.Sequential([
                keras.layers.Flatten(),
                keras.layers.Dense(2048, activation='relu', input_shape=(128*6*6,)),
                keras.layers.Dropout(0.5),
            ], name='classifier_1_1')
            # self.classifier_1_1.load_weights('./alexnet_classifier_1_1_weights')
        
        if 'classifier_1_2' in self.layer_list:
            self.classifier_1_2 = keras.models.Sequential([
                keras.layers.Flatten(),
                keras.layers.Dense(2048, activation='relu', input_shape=(128*6*6,)),
                keras.layers.Dropout(0.5),
            ], name='classifier_1_2')
            # self.classifier_1_2.load_weights('./alexnet_classifier_1_2_weights')

        if 'classifier_2_1' in self.layer_list:
            self.classifier_2_1 = keras.models.Sequential([
                keras.layers.Dense(2048, activation='relu', input_shape=(2048,)),
                keras.layers.Dropout(0.5),
            ], name='classifier_2_1')
            # self.classifier_2_1.load_weights('./alexnet_classifier_2_1_weights')

        if 'classifier_2_2' in self.layer_list:
            self.classifier_2_2 = keras.models.Sequential([
                keras.layers.Dense(2048, activation='relu', input_shape=(2048,)),
                keras.layers.Dropout(0.5),
            ], name='classifier_2_2')
            # self.classifier_2_2.load_weights('./alexnet_classifier_2_2_weights')

        if 'classifier_3_1' in self.layer_list:
            self.classifier_3_1 = keras.models.Sequential([
                keras.layers.Dense(5, activation='softmax', input_shape=(2048,)),
            ], name='classifier_3_1')
            # self.classifier_3_1.load_weights('./alexnet_classifier_3_1_weights')
        
        if 'classifier_3_2' in self.layer_list:
            self.classifier_3_2 = keras.models.Sequential([
                keras.layers.Dense(5, activation='softmax', input_shape=(2048,)),
            ], name='classifier_3_2')
            # self.classifier_3_2.load_weights('./alexnet_classifier_3_2_weights')

    def get_random_input(self):
        if 'input' in self.layer_list:
            return np.zeros((1,224,224,3))
        if 'features_1_1' in self.layer_list:
            return np.zeros((1,224,224,3))
        elif 'features_1_2' in self.layer_list:
            return np.zeros((1,224,224,3))
        elif 'features_2_1' in self.layer_list:
            return np.zeros((1,27,27,48))
        elif 'features_2_2' in self.layer_list:
            return np.zeros((1,27,27,48))
        elif 'features_3_1' in self.layer_list:
            return np.zeros((1,13,13,128))
        elif 'features_3_2' in self.layer_list:
            return np.zeros((1,13,13,128))
        elif 'features_4_1' in self.layer_list:
            return np.zeros((1,13,13,192))
        elif 'features_4_2' in self.layer_list:
            return np.zeros((1,13,13,192))
        elif 'features_5_1' in self.layer_list:
            return np.zeros((1,13,13,128))
        elif 'features_5_2' in self.layer_list:
            return np.zeros((1,13,13,128))
        elif 'classifier_1_1' in self.layer_list:
            return np.zeros((1,6,6,128))
        elif 'classifier_1_2' in self.layer_list:
            return np.zeros((1,6,6,128))
        elif 'classifier_2_1' in self.layer_list:
            return np.zeros(2048)
        elif 'classifier_2_2' in self.layer_list:
            return np.zeros(2048)
        elif 'classifier_3_1' in self.layer_list:
            return np.zeros(2048)
        elif 'classifier_3_2' in self.layer_list:
            return np.zeros(2048)

    def call(self, x):
        x_1 = None; x_2 = None
        if type(x) in (tuple, list):
            x = np.concatenate(x, axis=-1)
        if 'input' in self.layer_list:
            x = tf.image.resize(x, size=(224,224), method='nearest')
        if 'features_1_1' in self.layer_list:            
            x_1 = self.features_1_1(x)
        if 'features_2_1' in self.layer_list:
            if x_1 is None:
                x_1 = x
            x_1 = self.features_2_1(x_1)
        if 'features_3_1' in self.layer_list:
            if x_1 is None:
                x_1 = x
            x_1 = self.features_3_1(x_1)
        if 'features_4_1' in self.layer_list:
            if x_1 is None:
                x_1 = x
            x_1 = self.features_4_1(x_1)
        if 'features_5_1' in self.layer_list:
            if x_1 is None:
                x_1 = x
            x_1 = self.features_5_1(x_1)
        if 'classifier_1_1' in self.layer_list:
            if x_1 is None:
                x_1 = x
            x_1 = self.classifier_1_1(x_1)
        if 'classifier_2_1' in self.layer_list:
            if x_1 is None:
                x_1 = x
            x_1 = self.classifier_2_1(x_1)
        if 'classifier_3_1' in self.layer_list:
            if x_1 is None:
                x_1 = x
            x_1 = self.classifier_3_1(x_1)

        if 'features_1_2' in self.layer_list:            
            x_2 = self.features_1_2(x)
        if 'features_2_2' in self.layer_list:
            if x_2 is None:
                x_2 = x
            x_2 = self.features_2_2(x_2)
        if 'features_3_2' in self.layer_list:
            if x_2 is None:
                x_2 = x
            x_2 = self.features_3_2(x_2)
        if 'features_4_2' in self.layer_list:
            if x_2 is None:
                x_2 = x
            x_2 = self.features_4_2(x_2)
        if 'features_5_2' in self.layer_list:
            if x_2 is None:
                x_2 = x
            x_2 = self.features_5_2(x_2)
        if 'classifier_1_2' in self.layer_list:
            if x_2 is None:
                x_2 = x
            x_2 = self.classifier_1_2(x_2)
        if 'classifier_2_2' in self.layer_list:
            if x_2 is None:
                x_2 = x
            x_2 = self.classifier_2_2(x_2)
        if 'classifier_3_2' in self.layer_list:
            if x_2 is None:
                x_2 = x
            x_2 = self.classifier_3_2(x_2)
        
        if x_1 is not None:
            if x_2 is not None:
                x = np.concatenate((x_1, x_2), -1)
            else:
                x = x_1
        elif x_2 is not None:
                x = x_2

        return x