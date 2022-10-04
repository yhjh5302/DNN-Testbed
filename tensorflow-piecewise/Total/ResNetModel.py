import numpy as np
import tensorflow as tf
from tensorflow import keras




class ResNet_layer(keras.Model):
    def __init__(self, name=None, layer_list=None):
        super(ResNet_layer, self).__init__(name=name)
        self.layer_list = layer_list
        if 'input' in self.layer_list:
            self.resize = keras.layers.Resizing(height=224, width=224, interpolation='nearest', name='resize')
            self.input_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(224,224,3)),
                keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same')
            ], name='input_layer')
        if 'cnn_1_2' in self.layer_list:
            self.cnn_1_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(56, 56, 64)),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(56, 56, 64)),
                keras.layers.BatchNormalization(),
            ], name='cnn_1_2_layer')
        if 'cnn_2_1' in self.layer_list:
            self.cnn_2_1_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(56, 56, 64)),
                keras.layers.BatchNormalization()], name='cnn_2_1_1_layer')
            self.cnn_2_1_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(56, 56, 64)),
                keras.layers.BatchNormalization()], name='cnn_2_1_2_layer')
        if 'cnn_3_2' in self.layer_list:
            self.cnn_3_2_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(56, 56, 64)),
                keras.layers.BatchNormalization()], name='cnn_3_2_1_layer')
            self.cnn_3_2_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(56, 56, 64)),
                keras.layers.BatchNormalization()], name='cnn_3_2_2_layer')
        if 'cnn_4_1' in self.layer_list:
            self.cnn_4_1_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same', input_shape=(56, 56, 64)),
                keras.layers.BatchNormalization()], name='cnn_4_1_1_layer')
            self.cnn_4_1_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(28, 28,128)),
                keras.layers.BatchNormalization()], name='cnn_4_1_2_layer')
        if 'cnn_5_2' in self.layer_list:
            self.cnn_5_2_down_scale = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=(2,2), activation='relu', padding='same', input_shape=(56, 56, 64)),
                keras.layers.BatchNormalization()], name='cnn_5_2_down_scale')
            self.cnn_5_2_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(28, 28,128)),
                keras.layers.BatchNormalization()], name='cnn_5_2_1_layer')
            self.cnn_5_2_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(28, 28,128)),
                keras.layers.BatchNormalization()], name='cnn_5_2_2_layer')
        if 'cnn_6_1' in self.layer_list:
            self.cnn_6_1_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(28, 28,128)),
                keras.layers.BatchNormalization()], name='cnn_6_1_1_layer')
            self.cnn_6_1_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(28, 28,128)),
                keras.layers.BatchNormalization()], name='cnn_6_1_2_layer')
        if 'cnn_7_2' in self.layer_list:
            self.cnn_7_2_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(28, 28, 128)),
                keras.layers.BatchNormalization()], name='cnn_7_2_1_layer')
            self.cnn_7_2_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(28, 28, 128)),
                keras.layers.BatchNormalization()], name='cnn_7_2_2_layer')
        if 'cnn_8_1' in self.layer_list:            
            self.cnn_8_1_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same', input_shape=(28, 28, 128)),
                keras.layers.BatchNormalization()], name='cnn_8_1_1_layer')
            self.cnn_8_1_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(14, 14, 256)),
                keras.layers.BatchNormalization()], name='cnn_8_1_2_layer')
        if 'cnn_9_2' in self.layer_list:
            self.cnn_9_2_down_scale = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(2,2), activation='relu', padding='same', input_shape=(28, 28, 128)),
                keras.layers.BatchNormalization()], name='cnn_9_2_down_scale')
            self.cnn_9_2_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(14, 14, 256)),
                keras.layers.BatchNormalization()], name='cnn_9_2_1_layer')
            self.cnn_9_2_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(14, 14, 256)),
                keras.layers.BatchNormalization()], name='cnn_9_2_2_layer')
        if 'cnn_10_1' in self.layer_list:
            self.cnn_10_1_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(14, 14, 256)),
                keras.layers.BatchNormalization()], name='cnn_10_1_1_layer')
            self.cnn_10_1_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(14, 14, 256)),
                keras.layers.BatchNormalization()], name='cnn_10_1_2_layer')
        if 'cnn_11_2' in self.layer_list:
            self.cnn_11_2_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(14, 14, 256)),
                keras.layers.BatchNormalization()], name='cnn_11_2_1_layer')
            self.cnn_11_2_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(14, 14, 256)),
                keras.layers.BatchNormalization()], name='cnn_11_2_2_layer')
        if 'cnn_12_1' in self.layer_list:
            self.cnn_12_1_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(14, 14, 256)),
                keras.layers.BatchNormalization()], name='cnn_12_1_1_layer')
            self.cnn_12_1_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(14, 14, 256)),
                keras.layers.BatchNormalization()], name='cnn_12_1_2_layer')
        if 'cnn_13_2' in self.layer_list:
            self.cnn_13_2_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(14, 14, 256)),
                keras.layers.BatchNormalization()], name='cnn_13_2_1_layer')
            self.cnn_13_2_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(14, 14, 256)),
                keras.layers.BatchNormalization()], name='cnn_13_2_2_layer')
        if 'cnn_14_1' in self.layer_list:
            self.cnn_14_1_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same', input_shape=(14, 14, 256)),
                keras.layers.BatchNormalization()], name='cnn_14_1_1_layer')
            self.cnn_14_1_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(7, 7, 512)),
                keras.layers.BatchNormalization()], name='cnn_14_1_2_layer')
        if 'cnn_15_2' in self.layer_list: # half accured
            self.cnn_15_2_down_scale = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=(2,2), activation='relu', padding='same', input_shape=(14, 14, 256)),
                keras.layers.BatchNormalization()], name='cnn_15_2_down_scale')
            self.cnn_15_2_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(7, 7, 512)),
                keras.layers.BatchNormalization()], name='cnn_15_2_1_layer')
            self.cnn_15_2_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(7, 7, 512)),
                keras.layers.BatchNormalization()], name='cnn_15_2_2_layer')
        if 'cnn_16_1' in self.layer_list:
            self.cnn_16_1_1_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(7, 7, 512)),
                keras.layers.BatchNormalization()], name='cnn_16_1_1_layer')
            self.cnn_16_1_2_layer = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(7, 7, 512)),
                keras.layers.BatchNormalization()], name='cnn_16_1_2_layer')
        if 'cnn_17' in self.layer_list:
            self.cnn_17_layer = keras.models.Sequential([
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(1000)
                ], name='cnn_17_layer')
        

    def get_random_input(self):
        if 'input' in self.layer_list:
            return np.zeros((1,224,224,3))
        elif 'cnn_1_2' in self.layer_list:
            return np.zeros((1, 56, 56, 64))
        elif 'cnn_2_1' in self.layer_list:
            return (np.zeros((1, 56, 56, 64)), np.zeros((1, 56, 56, 64)))
        elif 'cnn_3_2' in self.layer_list:
            return (np.zeros((1, 56, 56, 64)), np.zeros((1, 56, 56, 64)))
        elif 'cnn_4_1' in self.layer_list:
            return (np.zeros((1, 56, 56, 64)), np.zeros((1, 56, 56, 64)))
        elif 'cnn_5_2' in self.layer_list:
            return (np.zeros((1, 56, 56, 64)), np.zeros((1, 28, 28, 128)))
        elif 'cnn_6_1' in self.layer_list:
            return (np.zeros((1, 28, 28, 128)), np.zeros((1, 28, 28, 128)))
        elif 'cnn_7_2' in self.layer_list:
            return (np.zeros((1, 28, 28, 128)), np.zeros((1, 28, 28, 128)))
        elif 'cnn_8_1' in self.layer_list:
            return (np.zeros((1, 28, 28, 128)), np.zeros((1, 28, 28, 128)))
        elif 'cnn_9_2' in self.layer_list:
            return (np.zeros((1, 28, 28, 128)),  np.zeros((1, 14, 14, 256)))
        elif 'cnn_10_1' in self.layer_list:
            return (np.zeros((1, 14, 14, 256)), np.zeros((1, 14, 14, 256)))
        elif 'cnn_11_2' in self.layer_list:
            return (np.zeros((1, 14, 14, 256)), np.zeros((1, 14, 14, 256)))
        elif 'cnn_12_1' in self.layer_list:
            return (np.zeros((1, 14, 14, 256)), np.zeros((1, 14, 14, 256)))
        elif 'cnn_13_2' in self.layer_list:
            return (np.zeros((1, 14, 14, 256)), np.zeros((1, 14, 14, 256)))
        elif 'cnn_14_1' in self.layer_list:
            return (np.zeros((1, 14, 14, 256)), np.zeros((1, 14, 14, 256)))
        elif 'cnn_15_2' in self.layer_list:
            return (np.zeros((1, 14, 14, 256)), np.zeros((1, 7, 7, 512)))
        elif 'cnn_16_1' in self.layer_list:
            return (np.zeros((1, 7, 7, 512)), np.zeros((1, 7, 7, 512)))
        elif 'cnn_17' in self.layer_list:
            return (np.zeros((1, 7, 7, 512)), np.zeros((1, 7, 7, 512)))

    def call(self, x, shortcut=None, shortcut2=None):
        if 'input' in self.layer_list:
            x = self.resize(x)
            x = self.input_layer(x)
            shortcut = x
        if 'cnn_1_2' in self.layer_list:
            x = self.cnn_1_2_layer(x)
            shortcut2 = x
        if 'cnn_2_1' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut = x[0]
                x = x[1]

            x = keras.layers.Add()([shortcut, x])
            x = self.cnn_2_1_1_layer(x)            
            x = self.cnn_2_1_2_layer(x)
            shortcut = x
            
        if 'cnn_3_2' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut2 = x[0]
                x = x[1]
            
            x = keras.layers.Add()([shortcut2, x])
            x = self.cnn_3_2_1_layer(x)
            x = self.cnn_3_2_2_layer(x)
            shortcut2 = x
        if 'cnn_4_1' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut = x[0]
                x = x[1]
            
            x = keras.layers.Add()([shortcut, x])
            x = self.cnn_4_1_1_layer(x)            
            x = self.cnn_4_1_2_layer(x)
            shortcut = x
        if 'cnn_5_2' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut2 = x[0]
                x = x[1]
            
            shortcut2 = self.cnn_5_2_down_scale(shortcut2)
            x = keras.layers.Add()([shortcut2, x])
            x = self.cnn_5_2_1_layer(x)            
            x = self.cnn_5_2_2_layer(x)
            shortcut2 = x
        if 'cnn_6_1' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut = x[0]
                x = x[1]
            
            x = keras.layers.Add()([shortcut, x])
            x = self.cnn_6_1_1_layer(x)            
            x = self.cnn_6_1_2_layer(x)
            shortcut = x
        if 'cnn_7_2' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut2 = x[0]
                x = x[1]
            
            x = keras.layers.Add()([shortcut2, x])
            x = self.cnn_7_2_1_layer(x)            
            x = self.cnn_7_2_2_layer(x)
            shortcut2 = x
        if 'cnn_8_1' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut = x[0]
                x = x[1]
            
            x = keras.layers.Add()([shortcut, x])
            x = self.cnn_8_1_1_layer(x)            
            x = self.cnn_8_1_2_layer(x)
            shortcut = x
        if 'cnn_9_2' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut2 = x[0]
                x = x[1]
            
            shortcut2 = self.cnn_9_2_down_scale(shortcut2)
            x = keras.layers.Add()([shortcut2, x])
            x = self.cnn_9_2_1_layer(x)            
            x = self.cnn_9_2_2_layer(x)
            shortcut2 = x
        if 'cnn_10_1' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut = x[0]
                x = x[1]
            
            x = keras.layers.Add()([shortcut, x])
            x = self.cnn_10_1_1_layer(x)            
            x = self.cnn_10_1_2_layer(x)
            shortcut = x
        if 'cnn_11_2' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut2 = x[0]
                x = x[1]
            
            x = keras.layers.Add()([shortcut2, x])
            x = self.cnn_11_2_1_layer(x)            
            x = self.cnn_11_2_2_layer(x)
            shortcut2 = x
        if 'cnn_12_1' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut = x[0]
                x = x[1]
            
            x = keras.layers.Add()([shortcut, x])
            x = self.cnn_12_1_1_layer(x)            
            x = self.cnn_12_1_2_layer(x)
            shortcut = x
        if 'cnn_13_2' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut2 = x[0]
                x = x[1]
            
            x = keras.layers.Add()([shortcut2, x])
            x = self.cnn_13_2_1_layer(x)            
            x = self.cnn_13_2_2_layer(x)
            shortcut2 = x
        if 'cnn_14_1' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut = x[0]
                x = x[1]
            
            x = keras.layers.Add()([shortcut, x])
            x = self.cnn_14_1_1_layer(x)            
            x = self.cnn_14_1_2_layer(x)
            shortcut = x
        if 'cnn_15_2' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut2 = x[0]
                x = x[1]
            
            shortcut2 = self.cnn_15_2_down_scale(shortcut2)
            x = keras.layers.Add()([shortcut2, x])
            x = self.cnn_15_2_1_layer(x)            
            x = self.cnn_15_2_2_layer(x)
            shortcut2 = x
            
        if 'cnn_16_1' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut = x[0]
                x = x[1]
            
            x = keras.layers.Add()([shortcut, x])
            x = self.cnn_16_1_1_layer(x)            
            x = self.cnn_16_1_2_layer(x)
        if 'cnn_17' in self.layer_list:
            if type(x) in (tuple, list):
                shortcut2 = x[0]
                x = x[1]
            
            x = keras.layers.Add()([shortcut2, x])
            x = self.cnn_17_layer(x)
        return x, shortcut, shortcut2
