import numpy as np
import tensorflow as tf
from tensorflow import keras

class GoogLeNet(keras.Model):
    def __init__(self, name=None, layer_list=None):
        super(GoogLeNet, self).__init__(name=name)
        self.layer_list = layer_list
        if 'conv1' in self.layer_list:
            self.conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='conv1')
            self.conv1.load_weights('./GoogLeNet_conv1_weights')
        if 'conv1_maxpool' in self.layer_list:
            self.conv1_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
        if 'conv2' in self.layer_list:
            self.conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='conv2')
            self.conv2.load_weights('./GoogLeNet_conv2_weights')
        if 'conv3' in self.layer_list:
            self.conv3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='conv3')
            self.conv3.load_weights('./GoogLeNet_conv3_weights')
        if 'conv3_maxpool' in self.layer_list:
            self.conv3_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)

        if 'inception3a_branch1' in self.layer_list:
            self.inception3a_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3a_branch1')
            self.inception3a_branch1.load_weights('./GoogLeNet_inception3a_branch1_weights')
        if 'inception3a_branch2_conv1' in self.layer_list:
            self.inception3a_branch2_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=96, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception3a_branch2_conv1')
            self.inception3a_branch2_conv1.load_weights('./GoogLeNet_inception3a_branch2_conv1_weights')
        if 'inception3a_branch2_conv2' in self.layer_list:
            self.inception3a_branch2_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3a_branch2_conv2')
            self.inception3a_branch2_conv2.load_weights('./GoogLeNet_inception3a_branch2_conv2_weights')
        if 'inception3a_branch3_conv1' in self.layer_list:
            self.inception3a_branch3_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=16, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception3a_branch3_conv1')
            self.inception3a_branch3_conv1.load_weights('./GoogLeNet_inception3a_branch3_conv1_weights')
        if 'inception3a_branch3_conv2' in self.layer_list:
            self.inception3a_branch3_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3a_branch3_conv2')
            self.inception3a_branch3_conv2.load_weights('./GoogLeNet_inception3a_branch3_conv2_weights')
        if 'inception3a_branch4_maxpool' in self.layer_list:
            self.inception3a_branch4_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same')
        if 'inception3a_branch4_conv' in self.layer_list:
            self.inception3a_branch4_conv = keras.models.Sequential([
                keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3a_branch4_conv')
            self.inception3a_branch4_conv.load_weights('./GoogLeNet_inception3a_branch4_conv_weights')

        if 'inception3b_branch1' in self.layer_list:
            self.inception3b_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3b_branch1')
            self.inception3b_branch1.load_weights('./GoogLeNet_inception3b_branch1_weights')
        if 'inception3b_branch2_conv1' in self.layer_list:
            self.inception3b_branch2_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception3b_branch2_conv1')
            self.inception3b_branch2_conv1.load_weights('./GoogLeNet_inception3b_branch2_conv1_weights')
        if 'inception3b_branch2_conv2' in self.layer_list:
            self.inception3b_branch2_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3b_branch2_conv2')
            self.inception3b_branch2_conv2.load_weights('./GoogLeNet_inception3b_branch2_conv2_weights')
        if 'inception3b_branch3_conv1' in self.layer_list:
            self.inception3b_branch3_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception3b_branch3_conv1')
            self.inception3b_branch3_conv1.load_weights('./GoogLeNet_inception3b_branch3_conv1_weights')
        if 'inception3b_branch3_conv2' in self.layer_list:
            self.inception3b_branch3_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=96, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3b_branch3_conv2')
            self.inception3b_branch3_conv2.load_weights('./GoogLeNet_inception3b_branch3_conv2_weights')
        if 'inception3b_branch4_maxpool' in self.layer_list:
            self.inception3b_branch4_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same')
        if 'inception3b_branch4_conv' in self.layer_list:
            self.inception3b_branch4_conv = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3b_branch4_conv')
            self.inception3b_branch4_conv.load_weights('./GoogLeNet_inception3b_branch4_conv_weights')
        if 'inception3b_maxpool' in self.layer_list:
            self.inception3b_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)

        if 'inception4a_branch1' in self.layer_list:
            self.inception4a_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=192, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4a_branch1')
            self.inception4a_branch1.load_weights('./GoogLeNet_inception4a_branch1_weights')
        if 'inception4a_branch2_conv1' in self.layer_list:
            self.inception4a_branch2_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=96, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception4a_branch2_conv1')
            self.inception4a_branch2_conv1.load_weights('./GoogLeNet_inception4a_branch2_conv1_weights')
        if 'inception4a_branch2_conv2' in self.layer_list:
            self.inception4a_branch2_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=208, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4a_branch2_conv2')
            self.inception4a_branch2_conv2.load_weights('./GoogLeNet_inception4a_branch2_conv2_weights')
        if 'inception4a_branch3_conv1' in self.layer_list:
            self.inception4a_branch3_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=16, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception4a_branch3_conv1')
            self.inception4a_branch3_conv1.load_weights('./GoogLeNet_inception4a_branch3_conv1_weights')
        if 'inception4a_branch3_conv2' in self.layer_list:
            self.inception4a_branch3_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=48, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4a_branch3_conv2')
            self.inception4a_branch3_conv2.load_weights('./GoogLeNet_inception4a_branch3_conv2_weights')
        if 'inception4a_branch4_maxpool' in self.layer_list:
            self.inception4a_branch4_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same')
        if 'inception4a_branch4_conv' in self.layer_list:
            self.inception4a_branch4_conv = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4a_branch4_conv')
            self.inception4a_branch4_conv.load_weights('./GoogLeNet_inception4a_branch4_conv_weights')

        if 'inception4b_branch1' in self.layer_list:
            self.inception4b_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=160, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4b_branch1')
            self.inception4b_branch1.load_weights('./GoogLeNet_inception4b_branch1_weights')
        if 'inception4b_branch2_conv1' in self.layer_list:
            self.inception4b_branch2_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=112, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception4b_branch2_conv1')
            self.inception4b_branch2_conv1.load_weights('./GoogLeNet_inception4b_branch2_conv1_weights')
        if 'inception4b_branch2_conv2' in self.layer_list:
            self.inception4b_branch2_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=224, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4b_branch2_conv2')
            self.inception4b_branch2_conv2.load_weights('./GoogLeNet_inception4b_branch2_conv2_weights')
        if 'inception4b_branch3_conv1' in self.layer_list:
            self.inception4b_branch3_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=24, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception4b_branch3_conv1')
            self.inception4b_branch3_conv1.load_weights('./GoogLeNet_inception4b_branch3_conv1_weights')
        if 'inception4b_branch3_conv2' in self.layer_list:
            self.inception4b_branch3_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4b_branch3_conv2')
            self.inception4b_branch3_conv2.load_weights('./GoogLeNet_inception4b_branch3_conv2_weights')
        if 'inception4b_branch4_maxpool' in self.layer_list:
            self.inception4b_branch4_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same')
        if 'inception4b_branch4_conv' in self.layer_list:
            self.inception4b_branch4_conv = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4b_branch4_conv')
            self.inception4b_branch4_conv.load_weights('./GoogLeNet_inception4b_branch4_conv_weights')

        if 'inception4c_branch1' in self.layer_list:
            self.inception4c_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4c_branch1')
            self.inception4c_branch1.load_weights('./GoogLeNet_inception4c_branch1_weights')
        if 'inception4c_branch2_conv1' in self.layer_list:
            self.inception4c_branch2_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception4c_branch2_conv1')
            self.inception4c_branch2_conv1.load_weights('./GoogLeNet_inception4c_branch2_conv1_weights')
        if 'inception4c_branch2_conv2' in self.layer_list:
            self.inception4c_branch2_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4c_branch2_conv2')
            self.inception4c_branch2_conv2.load_weights('./GoogLeNet_inception4c_branch2_conv2_weights')
        if 'inception4c_branch3_conv1' in self.layer_list:
            self.inception4c_branch3_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=24, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception4c_branch3_conv1')
            self.inception4c_branch3_conv1.load_weights('./GoogLeNet_inception4c_branch3_conv1_weights')
        if 'inception4c_branch3_conv2' in self.layer_list:
            self.inception4c_branch3_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4c_branch3_conv2')
            self.inception4c_branch3_conv2.load_weights('./GoogLeNet_inception4c_branch3_conv2_weights')
        if 'inception4c_branch4_maxpool' in self.layer_list:
            self.inception4c_branch4_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same')
        if 'inception4c_branch4_conv' in self.layer_list:
            self.inception4c_branch4_conv = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4c_branch4_conv')
            self.inception4c_branch4_conv.load_weights('./GoogLeNet_inception4c_branch4_conv_weights')

        if 'inception4d_branch1' in self.layer_list:
            self.inception4d_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=112, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4d_branch1')
            self.inception4d_branch1.load_weights('./GoogLeNet_inception4d_branch1_weights')
        if 'inception4d_branch2_conv1' in self.layer_list:
            self.inception4d_branch2_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=144, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception4d_branch2_conv1')
            self.inception4d_branch2_conv1.load_weights('./GoogLeNet_inception4d_branch2_conv1_weights')
        if 'inception4d_branch2_conv2' in self.layer_list:
            self.inception4d_branch2_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=288, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4d_branch2_conv2')
            self.inception4d_branch2_conv2.load_weights('./GoogLeNet_inception4d_branch2_conv2_weights')
        if 'inception4d_branch3_conv1' in self.layer_list:
            self.inception4d_branch3_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception4d_branch3_conv1')
            self.inception4d_branch3_conv1.load_weights('./GoogLeNet_inception4d_branch3_conv1_weights')
        if 'inception4d_branch3_conv2' in self.layer_list:
            self.inception4d_branch3_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4d_branch3_conv2')
            self.inception4d_branch3_conv2.load_weights('./GoogLeNet_inception4d_branch3_conv2_weights')
        if 'inception4d_branch4_maxpool' in self.layer_list:
            self.inception4d_branch4_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same')
        if 'inception4d_branch4_conv' in self.layer_list:
            self.inception4d_branch4_conv = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4d_branch4_conv')
            self.inception4d_branch4_conv.load_weights('./GoogLeNet_inception4d_branch4_conv_weights')

        if 'inception4e_branch1' in self.layer_list:
            self.inception4e_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4e_branch1')
            self.inception4e_branch1.load_weights('./GoogLeNet_inception4e_branch1_weights')
        if 'inception4e_branch2_conv1' in self.layer_list:
            self.inception4e_branch2_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=160, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception4e_branch2_conv1')
            self.inception4e_branch2_conv1.load_weights('./GoogLeNet_inception4e_branch2_conv1_weights')
        if 'inception4e_branch2_conv2' in self.layer_list:
            self.inception4e_branch2_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=320, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4e_branch2_conv2')
            self.inception4e_branch2_conv2.load_weights('./GoogLeNet_inception4e_branch2_conv2_weights')
        if 'inception4e_branch3_conv1' in self.layer_list:
            self.inception4e_branch3_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception4e_branch3_conv1')
            self.inception4e_branch3_conv1.load_weights('./GoogLeNet_inception4e_branch3_conv1_weights')
        if 'inception4e_branch3_conv2' in self.layer_list:
            self.inception4e_branch3_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4e_branch3_conv2')
            self.inception4e_branch3_conv2.load_weights('./GoogLeNet_inception4e_branch3_conv2_weights')
        if 'inception4e_branch4_maxpool' in self.layer_list:
            self.inception4e_branch4_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same')
        if 'inception4e_branch4_conv' in self.layer_list:
            self.inception4e_branch4_conv = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4e_branch4_conv')
            self.inception4e_branch4_conv.load_weights('./GoogLeNet_inception4e_branch4_conv_weights')
        if 'inception4e_maxpool' in self.layer_list:
            self.inception4e_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)

        if 'inception5a_branch1' in self.layer_list:
            self.inception5a_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5a_branch1')
            self.inception5a_branch1.load_weights('./GoogLeNet_inception5a_branch1_weights')
        if 'inception5a_branch2_conv1' in self.layer_list:
            self.inception5a_branch2_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=160, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception5a_branch2_conv1')
            self.inception5a_branch2_conv1.load_weights('./GoogLeNet_inception5a_branch2_conv1_weights')
        if 'inception5a_branch2_conv2' in self.layer_list:
            self.inception5a_branch2_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=320, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5a_branch2_conv2')
            self.inception5a_branch2_conv2.load_weights('./GoogLeNet_inception5a_branch2_conv2_weights')
        if 'inception5a_branch3_conv1' in self.layer_list:
            self.inception5a_branch3_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception5a_branch3_conv1')
            self.inception5a_branch3_conv1.load_weights('./GoogLeNet_inception5a_branch3_conv1_weights')
        if 'inception5a_branch3_conv2' in self.layer_list:
            self.inception5a_branch3_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5a_branch3_conv2')
            self.inception5a_branch3_conv2.load_weights('./GoogLeNet_inception5a_branch3_conv2_weights')
        if 'inception5a_branch4_maxpool' in self.layer_list:
            self.inception5a_branch4_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same')
        if 'inception5a_branch4_conv' in self.layer_list:
            self.inception5a_branch4_conv = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5a_branch4_conv')
            self.inception5a_branch4_conv.load_weights('./GoogLeNet_inception5a_branch4_conv_weights')

        if 'inception5b_branch1' in self.layer_list:
            self.inception5b_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5b_branch1')
            self.inception5b_branch1.load_weights('./GoogLeNet_inception5b_branch1_weights')
        if 'inception5b_branch2_conv1' in self.layer_list:
            self.inception5b_branch2_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=192, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception5b_branch2_conv1')
            self.inception5b_branch2_conv1.load_weights('./GoogLeNet_inception5b_branch2_conv1_weights')
        if 'inception5b_branch2_conv2' in self.layer_list:
            self.inception5b_branch2_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5b_branch2_conv2')
            self.inception5b_branch2_conv2.load_weights('./GoogLeNet_inception5b_branch2_conv2_weights')
        if 'inception5b_branch3_conv1' in self.layer_list:
            self.inception5b_branch3_conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=48, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
            ], name='inception5b_branch3_conv1')
            self.inception5b_branch3_conv1.load_weights('./GoogLeNet_inception5b_branch3_conv1_weights')
        if 'inception5b_branch3_conv2' in self.layer_list:
            self.inception5b_branch3_conv2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5b_branch3_conv2')
            self.inception5b_branch3_conv2.load_weights('./GoogLeNet_inception5b_branch3_conv2_weights')
        if 'inception5b_branch4_maxpool' in self.layer_list:
            self.inception5b_branch4_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same')
        if 'inception5b_branch4_conv' in self.layer_list:
            self.inception5b_branch4_conv = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5b_branch4_conv')
            self.inception5b_branch4_conv.load_weights('./GoogLeNet_inception5b_branch4_conv_weights')
            self.flatten = keras.layers.Flatten()

        if 'fully_connected' in self.layer_list:
            self.fully_connected = keras.models.Sequential([
                keras.layers.Dense(1000, activation='softmax'),
            ], name='fully_connected')
            self.fully_connected.load_weights('./GoogLeNet_fully_connected_weights')

    def get_random_input(self):
        return np.zeros((1,224,224,3))

    def call(self, inputs):
        if 'conv1' in self.layer_list:
            x = tf.image.resize(inputs, size=(224,224), method='nearest')
            x = self.conv1(x)
        if 'conv1_maxpool' in self.layer_list:
            x = self.conv1_maxpool(x)
        if 'conv2' in self.layer_list:
            x = self.conv2(x)
        if 'conv3' in self.layer_list:
            x = self.conv3(x)
        if 'conv3_maxpool' in self.layer_list:
            x = self.conv3_maxpool(x)
            # inception3a()
        if 'inception3a_branch1' in self.layer_list:
            branch1 = self.inception3a_branch1(x)
        if 'inception3a_branch2_conv1' in self.layer_list:
            branch2 = self.inception3a_branch2_conv1(x)
        if 'inception3a_branch2_conv2' in self.layer_list:
            branch2 = self.inception3a_branch2_conv2(branch2)
        if 'inception3a_branch3_conv1' in self.layer_list:
            branch3 = self.inception3a_branch3_conv1(x)
        if 'inception3a_branch3_conv2' in self.layer_list:
            branch3 = self.inception3a_branch3_conv2(branch3)
        if 'inception3a_branch4_maxpool' in self.layer_list:
            branch4 = self.inception3a_branch4_maxpool(x)
        if 'inception3a_branch4_conv' in self.layer_list:
            branch4 = self.inception3a_branch4_conv(branch4)
        if 'inception3a_concat' in self.layer_list:
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
            # inception3b and max pool()
        if 'inception3b_branch1' in self.layer_list:
            branch1 = self.inception3b_branch1(x)
        if 'inception3b_branch2_conv1' in self.layer_list:
            branch2 = self.inception3b_branch2_conv1(x)
        if 'inception3b_branch2_conv2' in self.layer_list:
            branch2 = self.inception3b_branch2_conv2(branch2)
        if 'inception3b_branch3_conv1' in self.layer_list:
            branch3 = self.inception3b_branch3_conv1(x)
        if 'inception3b_branch3_conv2' in self.layer_list:
            branch3 = self.inception3b_branch3_conv2(branch3)
        if 'inception3b_branch4_maxpool' in self.layer_list:
            branch4 = self.inception3b_branch4_maxpool(x)
        if 'inception3b_branch4_conv' in self.layer_list:
            branch4 = self.inception3b_branch4_conv(branch4)
        if 'inception3b_concat' in self.layer_list:
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
        if 'inception3b_maxpool' in self.layer_list:
            x = self.inception3b_maxpool(x)
            # inception4a()
        if 'inception4a_branch1' in self.layer_list:
            branch1 = self.inception4a_branch1(x)
        if 'inception4a_branch2_conv1' in self.layer_list:
            branch2 = self.inception4a_branch2_conv1(x)
        if 'inception4a_branch2_conv2' in self.layer_list:
            branch2 = self.inception4a_branch2_conv2(branch2)
        if 'inception4a_branch3_conv1' in self.layer_list:
            branch3 = self.inception4a_branch3_conv1(x)
        if 'inception4a_branch3_conv2' in self.layer_list:
            branch3 = self.inception4a_branch3_conv2(branch3)
        if 'inception4a_branch4_maxpool' in self.layer_list:
            branch4 = self.inception4a_branch4_maxpool(x)
        if 'inception4a_branch4_conv' in self.layer_list:
            branch4 = self.inception4a_branch4_conv(branch4)
        if 'inception4a_concat' in self.layer_list:
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
            # inception4b()
        if 'inception4b_branch1' in self.layer_list:
            branch1 = self.inception4b_branch1(x)
        if 'inception4b_branch2_conv1' in self.layer_list:
            branch2 = self.inception4b_branch2_conv1(x)
        if 'inception4b_branch2_conv2' in self.layer_list:
            branch2 = self.inception4b_branch2_conv2(branch2)
        if 'inception4b_branch3_conv1' in self.layer_list:
            branch3 = self.inception4b_branch3_conv1(x)
        if 'inception4b_branch3_conv2' in self.layer_list:
            branch3 = self.inception4b_branch3_conv2(branch3)
        if 'inception4b_branch4_maxpool' in self.layer_list:
            branch4 = self.inception4b_branch4_maxpool(x)
        if 'inception4b_branch4_conv' in self.layer_list:
            branch4 = self.inception4b_branch4_conv(branch4)
        if 'inception4b_concat' in self.layer_list:
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
            # inception4c()
        if 'inception4c_branch1' in self.layer_list:
            branch1 = self.inception4c_branch1(x)
        if 'inception4c_branch2_conv1' in self.layer_list:
            branch2 = self.inception4c_branch2_conv1(x)
        if 'inception4c_branch2_conv2' in self.layer_list:
            branch2 = self.inception4c_branch2_conv2(branch2)
        if 'inception4c_branch3_conv1' in self.layer_list:
            branch3 = self.inception4c_branch3_conv1(x)
        if 'inception4c_branch3_conv2' in self.layer_list:
            branch3 = self.inception4c_branch3_conv2(branch3)
        if 'inception4c_branch4_maxpool' in self.layer_list:
            branch4 = self.inception4c_branch4_maxpool(x)
        if 'inception4c_branch4_conv' in self.layer_list:
            branch4 = self.inception4c_branch4_conv(branch4)
        if 'inception4c_concat' in self.layer_list:
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
            # inception4d()
        if 'inception4d_branch1' in self.layer_list:
            branch1 = self.inception4d_branch1(x)
        if 'inception4d_branch2_conv1' in self.layer_list:
            branch2 = self.inception4d_branch2_conv1(x)
        if 'inception4d_branch2_conv2' in self.layer_list:
            branch2 = self.inception4d_branch2_conv2(branch2)
        if 'inception4d_branch3_conv1' in self.layer_list:
            branch3 = self.inception4d_branch3_conv1(x)
        if 'inception4d_branch3_conv2' in self.layer_list:
            branch3 = self.inception4d_branch3_conv2(branch3)
        if 'inception4d_branch4_maxpool' in self.layer_list:
            branch4 = self.inception4d_branch4_maxpool(x)
        if 'inception4d_branch4_conv' in self.layer_list:
            branch4 = self.inception4d_branch4_conv(branch4)
        if 'inception4d_concat' in self.layer_list:
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
            # inception4e()
        if 'inception4e_branch1' in self.layer_list:
            branch1 = self.inception4e_branch1(x)
        if 'inception4e_branch2_conv1' in self.layer_list:
            branch2 = self.inception4e_branch2_conv1(x)
        if 'inception4e_branch2_conv2' in self.layer_list:
            branch2 = self.inception4e_branch2_conv2(branch2)
        if 'inception4e_branch3_conv1' in self.layer_list:
            branch3 = self.inception4e_branch3_conv1(x)
        if 'inception4e_branch3_conv2' in self.layer_list:
            branch3 = self.inception4e_branch3_conv2(branch3)
        if 'inception4e_branch4_maxpool' in self.layer_list:
            branch4 = self.inception4e_branch4_maxpool(x)
        if 'inception4e_branch4_conv' in self.layer_list:
            branch4 = self.inception4e_branch4_conv(branch4)
        if 'inception4e_concat' in self.layer_list:
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
        if 'inception4e_maxpool' in self.layer_list:
            x = self.inception4e_maxpool(x)
            # inception5a()
        if 'inception5a_branch1' in self.layer_list:
            branch1 = self.inception5a_branch1(x)
        if 'inception5a_branch2_conv1' in self.layer_list:
            branch2 = self.inception5a_branch2_conv1(x)
        if 'inception5a_branch2_conv2' in self.layer_list:
            branch2 = self.inception5a_branch2_conv2(branch2)
        if 'inception5a_branch3_conv1' in self.layer_list:
            branch3 = self.inception5a_branch3_conv1(x)
        if 'inception5a_branch3_conv2' in self.layer_list:
            branch3 = self.inception5a_branch3_conv2(branch3)
        if 'inception5a_branch4_maxpool' in self.layer_list:
            branch4 = self.inception5a_branch4_maxpool(x)
        if 'inception5a_branch4_conv' in self.layer_list:
            branch4 = self.inception5a_branch4_conv(branch4)
        if 'inception5a_concat' in self.layer_list:
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
            # inception5b()
        if 'inception5b_branch1' in self.layer_list:
            branch1 = self.inception5b_branch1(x)
        if 'inception5b_branch2_conv1' in self.layer_list:
            branch2 = self.inception5b_branch2_conv1(x)
        if 'inception5b_branch2_conv2' in self.layer_list:
            branch2 = self.inception5b_branch2_conv2(branch2)
        if 'inception5b_branch3_conv1' in self.layer_list:
            branch3 = self.inception5b_branch3_conv1(x)
        if 'inception5b_branch3_conv2' in self.layer_list:
            branch3 = self.inception5b_branch3_conv2(branch3)
        if 'inception5b_branch4_maxpool' in self.layer_list:
            branch4 = self.inception5b_branch4_maxpool(x)
        if 'inception5b_branch4_conv' in self.layer_list:
            branch4 = self.inception5b_branch4_conv(branch4)
        if 'inception5b_concat' in self.layer_list:
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
            x = self.flatten(x)
            # avg pool, flatten and fully_connected
        if 'fully_connected' in self.layer_list:
            x = self.fully_connected(x)
        return x