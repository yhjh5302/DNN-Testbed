import numpy as np
import tensorflow as tf
from tensorflow import keras

class VGGNet(keras.Model):
    def __init__(self, name=None, layer_list=None):
        super(VGGNet, self).__init__(name=name)
        self.layer_list = layer_list
        if 'conv_1' in self.layer_list:
            self.conv_1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(224,224,3)),
                keras.layers.BatchNormalization(),
            ], name='conv_1')
            self.conv_1.load_weights('./VGGNet_conv_1_weights')
        if 'conv_2' in self.layer_list:
            self.conv_2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(224,224,64)),
                keras.layers.BatchNormalization(),
            ], name='conv_2')
            self.conv_2.load_weights('./VGGNet_conv_2_weights')
        if 'maxpool_1' in self.layer_list:
            self.maxpool_1 = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        if 'conv_3' in self.layer_list:
            self.conv_3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(112,112,64)),
                keras.layers.BatchNormalization(),
            ], name='conv_3')
            self.conv_3.load_weights('./VGGNet_conv_3_weights')
        if 'conv_4' in self.layer_list:
            self.conv_4 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(112,112,128)),
                keras.layers.BatchNormalization(),
            ], name='conv_4')
            self.conv_4.load_weights('./VGGNet_conv_4_weights')
        if 'maxpool_2' in self.layer_list:
            self.maxpool_2 = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        if 'conv_5' in self.layer_list:
            self.conv_5 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(56,56,128)),
                keras.layers.BatchNormalization(),
            ], name='conv_5')
            self.conv_5.load_weights('./VGGNet_conv_5_weights')
        if 'conv_6' in self.layer_list:
            self.conv_6 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(56,56,256)),
                keras.layers.BatchNormalization(),
            ], name='conv_6')
            self.conv_6.load_weights('./VGGNet_conv_6_weights')
        if 'conv_7' in self.layer_list:
            self.conv_7 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(56,56,256)),
                keras.layers.BatchNormalization(),
            ], name='conv_7')
            self.conv_7.load_weights('./VGGNet_conv_7_weights')
        if 'maxpool_3' in self.layer_list:
            self.maxpool_3 = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        if 'conv_8' in self.layer_list:
            self.conv_8 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(28,28,256)),
                keras.layers.BatchNormalization(),
            ], name='conv_8')
            self.conv_8.load_weights('./VGGNet_conv_8_weights')
        if 'conv_9' in self.layer_list:
            self.conv_9 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(28,28,512)),
                keras.layers.BatchNormalization(),
            ], name='conv_9')
            self.conv_9.load_weights('./VGGNet_conv_9_weights')
        if 'conv_10' in self.layer_list:
            self.conv_10 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(28,28,512)),
                keras.layers.BatchNormalization(),
            ], name='conv_10')
            self.conv_10.load_weights('./VGGNet_conv_10_weights')
        if 'maxpool_4' in self.layer_list:
            self.maxpool_4 = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        if 'conv_11' in self.layer_list:
            self.conv_11 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(14,14,512)),
                keras.layers.BatchNormalization(),
            ], name='conv_11')
            self.conv_11.load_weights('./VGGNet_conv_11_weights')
        if 'conv_12' in self.layer_list:
            self.conv_12 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(14,14,512)),
                keras.layers.BatchNormalization(),
            ], name='conv_12')
            self.conv_12.load_weights('./VGGNet_conv_12_weights')
        if 'conv_13' in self.layer_list:
            self.conv_13 = keras.models.Sequential([
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(14,14,512)),
                keras.layers.BatchNormalization(),
            ], name='conv_13')
            self.conv_13.load_weights('./VGGNet_conv_13_weights')
        if 'maxpool_5' in self.layer_list:
            self.maxpool_5 = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
            self.flatten = keras.layers.Flatten()
        if 'classifier_1' in self.layer_list:
            self.classifier_1 = keras.models.Sequential([
                keras.layers.Dense(4096, activation='relu', input_shape=(7*7*512,)),
            ], name='classifier_1')
            self.classifier_1.load_weights('./VGGNet_classifier_1_weights')
        if 'classifier_2' in self.layer_list:
            self.classifier_2 = keras.models.Sequential([
                keras.layers.Dense(4096, activation='relu', input_shape=(4096,)),
            ], name='classifier_2')
            self.classifier_2.load_weights('./VGGNet_classifier_2_weights')
        if 'classifier_3' in self.layer_list:
            self.classifier_3 = keras.models.Sequential([
                keras.layers.Dense(1000, activation='softmax', input_shape=(4096,)),
            ], name='classifier_3')
            self.classifier_3.load_weights('./VGGNet_classifier_3_weights')

    def get_random_input(self):
        if 'conv_1' in self.layer_list:
            return np.zeros((1,224,224,3))
        elif 'conv_2' in self.layer_list:
            return np.zeros((1,224,224,64))
        if 'maxpool_1' in self.layer_list:
            return np.zeros((1,224,224,64))
        elif 'conv_3' in self.layer_list:
            return np.zeros((1,112,112,64))
        elif 'conv_4' in self.layer_list:
            return np.zeros((1,112,112,128))
        if 'maxpool_2' in self.layer_list:
            return np.zeros((1,112,112,128))
        elif 'conv_5' in self.layer_list:
            return np.zeros((1,56,56,128))
        elif 'conv_6' in self.layer_list:
            return np.zeros((1,56,56,256))
        elif 'conv_7' in self.layer_list:
            return np.zeros((1,56,56,256))
        if 'maxpool_3' in self.layer_list:
            return np.zeros((1,56,56,256))
        elif 'conv_8' in self.layer_list:
            return np.zeros((1,28,28,256))
        elif 'conv_9' in self.layer_list:
            return np.zeros((1,28,28,512))
        elif 'conv_10' in self.layer_list:
            return np.zeros((1,28,28,512))
        if 'maxpool_4' in self.layer_list:
            return np.zeros((1,28,28,512))
        elif 'conv_11' in self.layer_list:
            return np.zeros((1,14,14,512))
        elif 'conv_12' in self.layer_list:
            return np.zeros((1,14,14,512))
        elif 'conv_13' in self.layer_list:
            return np.zeros((1,14,14,512))
        if 'maxpool_5' in self.layer_list:
            return np.zeros((1,14,14,512))
        elif 'classifier_1' in self.layer_list:
            return np.zeros(7*7*512)
        elif 'classifier_2' in self.layer_list:
            return np.zeros(4096)
        elif 'classifier_3' in self.layer_list:
            return np.zeros(4096)

    def call(self, x):
        if 'conv_1' in self.layer_list:
            x = tf.image.resize(x, size=(224, 224), method='nearest')
            x = self.conv_1(x)
        elif 'conv_2' in self.layer_list:
            x = self.conv_2(x)
        if 'maxpool_1' in self.layer_list:
            x = self.maxpool_1(x)
        elif 'conv_3' in self.layer_list:
            x = self.conv_3(x)
        elif 'conv_4' in self.layer_list:
            x = self.conv_4(x)
        if 'maxpool_2' in self.layer_list:
            x = self.maxpool_2(x)
        elif 'conv_5' in self.layer_list:
            x = self.conv_5(x)
        elif 'conv_6' in self.layer_list:
            x = self.conv_6(x)
        elif 'conv_7' in self.layer_list:
            x = self.conv_7(x)
        if 'maxpool_3' in self.layer_list:
            x = self.maxpool_3(x)
        elif 'conv_8' in self.layer_list:
            x = self.conv_8(x)
        elif 'conv_9' in self.layer_list:
            x = self.conv_9(x)
        elif 'conv_10' in self.layer_list:
            x = self.conv_10(x)
        if 'maxpool_4' in self.layer_list:
            x = self.maxpool_4(x)
        elif 'conv_11' in self.layer_list:
            x = self.conv_11(x)
        elif 'conv_12' in self.layer_list:
            x = self.conv_12(x)
        elif 'conv_13' in self.layer_list:
            x = self.conv_13(x)
        if 'maxpool_5' in self.layer_list:
            x = self.maxpool_5(x)
        elif 'classifier_1' in self.layer_list:
            x = self.classifier_1(x)
        elif 'classifier_2' in self.layer_list:
            x = self.classifier_2(x)
        elif 'classifier_3' in self.layer_list:
            x = self.classifier_3(x)
        return x