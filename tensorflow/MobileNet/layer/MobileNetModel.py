import numpy as np
import tensorflow as tf
from tensorflow import keras

class MobileNet_layer(keras.Model):
    def __init__(self, name=None, layer_list=None):
        super(MobileNet_layer, self).__init__(name=name)
        self.layer_list = layer_list
        if 'conv1' in self.layer_list:
            self.resize = keras.layers.Resizing(height=224, width=224, interpolation='nearest', name='resize')
            self.conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='conv1')
            self.conv1.load_weights('./MobileNet_conv_1_weights')
        if 'separable_conv2' in self.layer_list:
            self.separable_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='separable_conv2')
            self.separable_conv2.load_weights('./MobileNet_separable_conv2_weights')
        if 'separable_conv3' in self.layer_list:
            self.separable_conv3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='separable_conv3')
            self.separable_conv3.load_weights('./MobileNet_separable_conv3_weights')
        if 'separable_conv4' in self.layer_list:
            self.separable_conv4 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='separable_conv4')
            self.separable_conv4.load_weights('./MobileNet_separable_conv4_weights')
        if 'separable_conv5' in self.layer_list:
            self.separable_conv5 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=2, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='separable_conv5')
            self.separable_conv5.load_weights('./MobileNet_separable_conv5_weights')
        if 'separable_conv6' in self.layer_list:
            self.separable_conv6 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='separable_conv6')
            self.separable_conv6.load_weights('./MobileNet_separable_conv6_weights')
        if 'separable_conv7' in self.layer_list:
            self.separable_conv7 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=2, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='separable_conv7')
            self.separable_conv7.load_weights('./MobileNet_separable_conv7_weights')
        if 'separable_conv8' in self.layer_list:
            self.separable_conv8 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='separable_conv8')
            self.separable_conv8.load_weights('./MobileNet_separable_conv8_weights')
        if 'separable_conv9' in self.layer_list:
            self.separable_conv9 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='separable_conv9')
            self.separable_conv9.load_weights('./MobileNet_separable_conv9_weights')
        if 'separable_conv10' in self.layer_list:
            self.separable_conv10 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='separable_conv10')
            self.separable_conv10.load_weights('./MobileNet_separable_conv10_weights')
        if 'separable_conv11' in self.layer_list:
            self.separable_conv11 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='separable_conv11')
            self.separable_conv11.load_weights('./MobileNet_separable_conv11_weights')
        if 'separable_conv12' in self.layer_list:
            self.separable_conv12 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='separable_conv12')
            self.separable_conv12.load_weights('./MobileNet_separable_conv12_weights')
        if 'separable_conv13' in self.layer_list:
            self.separable_conv13 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=2, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='separable_conv13')
            self.separable_conv13.load_weights('./MobileNet_separable_conv13_weights')
        if 'separable_conv14' in self.layer_list:
            self.separable_conv14 = keras.models.Sequential([
                keras.layers.Conv2D(filters=1024, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='separable_conv14')
            self.separable_conv14.load_weights('./MobileNet_separable_conv14_weights')
        if 'fully_connected' in self.layer_list:
            self.fully_connected = keras.models.Sequential([
                keras.layers.AveragePooling2D(pool_size=(7, 7)),
                keras.layers.Flatten(),
                keras.layers.Dense(1000, activation='softmax'),
            ], name='fully_connected')
            self.fully_connected.load_weights('./MobileNet_fully_connected_weights')

    def get_random_input(self):
        if 'conv1' in self.layer_list:
            return np.zeros((1,32,32,3))
        elif 'separable_conv2' in self.layer_list:
            return np.zeros((1,112,112,32))
        elif 'separable_conv3' in self.layer_list:
            return np.zeros((1,112,112,64))
        elif 'separable_conv4' in self.layer_list:
            return np.zeros((1,56,56,128))
        elif 'separable_conv5' in self.layer_list:
            return np.zeros((1,56,56,128))
        elif 'separable_conv6' in self.layer_list:
            return np.zeros((1,28,28,256))
        elif 'separable_conv7' in self.layer_list:
            return np.zeros((1,28,28,256))
        elif 'separable_conv8' in self.layer_list:
            return np.zeros((1,14,14,512))
        elif 'separable_conv9' in self.layer_list:
            return np.zeros((1,14,14,512))
        elif 'separable_conv10' in self.layer_list:
            return np.zeros((1,14,14,512))
        elif 'separable_conv11' in self.layer_list:
            return np.zeros((1,14,14,512))
        elif 'separable_conv12' in self.layer_list:
            return np.zeros((1,14,14,512))
        elif 'separable_conv13' in self.layer_list:
            return np.zeros((1,14,14,512))
        elif 'separable_conv14' in self.layer_list:
            return np.zeros((1,7,7,1024))
        elif 'fully_connected' in self.layer_list:
            return np.zeros((1,7,7,1024))

    def call(self, x):
        if 'conv1' in self.layer_list:
            x = self.resize(x)
            x = self.conv1(x)
        if 'separable_conv2' in self.layer_list:
            x = self.separable_conv2(x)
        if 'separable_conv3' in self.layer_list:
            x = self.separable_conv3(x)
        if 'separable_conv4' in self.layer_list:
            x = self.separable_conv4(x)
        if 'separable_conv5' in self.layer_list:
            x = self.separable_conv5(x)
        if 'separable_conv6' in self.layer_list:
            x = self.separable_conv6(x)
        if 'separable_conv7' in self.layer_list:
            x = self.separable_conv7(x)
        if 'separable_conv8' in self.layer_list:
            x = self.separable_conv8(x)
        if 'separable_conv9' in self.layer_list:
            x = self.separable_conv9(x)
        if 'separable_conv10' in self.layer_list:
            x = self.separable_conv10(x)
        if 'separable_conv11' in self.layer_list:
            x = self.separable_conv11(x)
        if 'separable_conv12' in self.layer_list:
            x = self.separable_conv12(x)
        if 'separable_conv13' in self.layer_list:
            x = self.separable_conv13(x)
        if 'separable_conv14' in self.layer_list:
            x = self.separable_conv14(x)
        if 'fully_connected' in self.layer_list:
            x = self.fully_connected(x)
        return x