import numpy as np
import tensorflow as tf
from tensorflow import keras

class GoogLeNet_layer(keras.Model):
    def __init__(self, name=None, layer_list=None):
        super(GoogLeNet_layer, self).__init__(name=name)
        self.layer_list = layer_list
        if 'conv1' in self.layer_list:
            self.resize = keras.layers.Resizing(height=224, width=224, interpolation='nearest', name='resize')
            self.conv1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='conv1')
            self.conv1_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
            self.conv1.load_weights('./GoogLeNet_conv1_weights')
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
            self.conv3_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
            self.conv3.load_weights('./GoogLeNet_conv3_weights')

        if 'inception3a' in self.layer_list:
            self.inception3a_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3a_branch1')
            self.inception3a_branch2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=96, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3a_branch2')
            self.inception3a_branch3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=16, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3a_branch3')
            self.inception3a_branch4 = keras.models.Sequential([
                keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
                keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3a_branch4')
            self.inception3a_branch1.load_weights('./GoogLeNet_inception3a_branch1_weights')
            self.inception3a_branch2.load_weights('./GoogLeNet_inception3a_branch2_weights')
            self.inception3a_branch3.load_weights('./GoogLeNet_inception3a_branch3_weights')
            self.inception3a_branch4.load_weights('./GoogLeNet_inception3a_branch4_weights')

        if 'inception3b' in self.layer_list:
            self.inception3b_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3b_branch1')
            self.inception3b_branch2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3b_branch2')
            self.inception3b_branch3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=96, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3b_branch3')
            self.inception3b_branch4 = keras.models.Sequential([
                keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception3b_branch4')
            self.inception3b_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
            self.inception3b_branch1.load_weights('./GoogLeNet_inception3b_branch1_weights')
            self.inception3b_branch2.load_weights('./GoogLeNet_inception3b_branch2_weights')
            self.inception3b_branch3.load_weights('./GoogLeNet_inception3b_branch3_weights')
            self.inception3b_branch4.load_weights('./GoogLeNet_inception3b_branch4_weights')

        if 'inception4a' in self.layer_list:
            self.inception4a_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=192, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4a_branch1')
            self.inception4a_branch2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=96, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=208, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4a_branch2')
            self.inception4a_branch3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=16, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=48, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4a_branch3')
            self.inception4a_branch4 = keras.models.Sequential([
                keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4a_branch4')
            self.inception4a_branch1.load_weights('./GoogLeNet_inception4a_branch1_weights')
            self.inception4a_branch2.load_weights('./GoogLeNet_inception4a_branch2_weights')
            self.inception4a_branch3.load_weights('./GoogLeNet_inception4a_branch3_weights')
            self.inception4a_branch4.load_weights('./GoogLeNet_inception4a_branch4_weights')

        if 'inception4b' in self.layer_list:
            self.inception4b_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=160, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4b_branch1')
            self.inception4b_branch2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=112, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=224, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4b_branch2')
            self.inception4b_branch3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=24, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4b_branch3')
            self.inception4b_branch4 = keras.models.Sequential([
                keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4b_branch4')
            self.inception4b_branch1.load_weights('./GoogLeNet_inception4b_branch1_weights')
            self.inception4b_branch2.load_weights('./GoogLeNet_inception4b_branch2_weights')
            self.inception4b_branch3.load_weights('./GoogLeNet_inception4b_branch3_weights')
            self.inception4b_branch4.load_weights('./GoogLeNet_inception4b_branch4_weights')

        if 'inception4c' in self.layer_list:
            self.inception4c_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4c_branch1')
            self.inception4c_branch2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4c_branch2')
            self.inception4c_branch3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=24, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4c_branch3')
            self.inception4c_branch4 = keras.models.Sequential([
                keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4c_branch4')
            self.inception4c_branch1.load_weights('./GoogLeNet_inception4c_branch1_weights')
            self.inception4c_branch2.load_weights('./GoogLeNet_inception4c_branch2_weights')
            self.inception4c_branch3.load_weights('./GoogLeNet_inception4c_branch3_weights')
            self.inception4c_branch4.load_weights('./GoogLeNet_inception4c_branch4_weights')

        if 'inception4d' in self.layer_list:
            self.inception4d_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=112, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4d_branch1')
            self.inception4d_branch2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=144, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=288, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4d_branch2')
            self.inception4d_branch3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4d_branch3')
            self.inception4d_branch4 = keras.models.Sequential([
                keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
                keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4d_branch4')
            self.inception4d_branch1.load_weights('./GoogLeNet_inception4d_branch1_weights')
            self.inception4d_branch2.load_weights('./GoogLeNet_inception4d_branch2_weights')
            self.inception4d_branch3.load_weights('./GoogLeNet_inception4d_branch3_weights')
            self.inception4d_branch4.load_weights('./GoogLeNet_inception4d_branch4_weights')

        if 'inception4e' in self.layer_list:
            self.inception4e_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4e_branch1')
            self.inception4e_branch2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=160, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=320, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4e_branch2')
            self.inception4e_branch3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4e_branch3')
            self.inception4e_branch4 = keras.models.Sequential([
                keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception4e_branch4')
            self.inception4e_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
            self.inception4e_branch1.load_weights('./GoogLeNet_inception4e_branch1_weights')
            self.inception4e_branch2.load_weights('./GoogLeNet_inception4e_branch2_weights')
            self.inception4e_branch3.load_weights('./GoogLeNet_inception4e_branch3_weights')
            self.inception4e_branch4.load_weights('./GoogLeNet_inception4e_branch4_weights')

        if 'inception5a' in self.layer_list:
            self.inception5a_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5a_branch1')
            self.inception5a_branch2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=160, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=320, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5a_branch2')
            self.inception5a_branch3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5a_branch3')
            self.inception5a_branch4 = keras.models.Sequential([
                keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5a_branch4')
            self.inception5a_branch1.load_weights('./GoogLeNet_inception5a_branch1_weights')
            self.inception5a_branch2.load_weights('./GoogLeNet_inception5a_branch2_weights')
            self.inception5a_branch3.load_weights('./GoogLeNet_inception5a_branch3_weights')
            self.inception5a_branch4.load_weights('./GoogLeNet_inception5a_branch4_weights')

        if 'inception5b' in self.layer_list:
            self.inception5b_branch1 = keras.models.Sequential([
                keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5b_branch1')
            self.inception5b_branch2 = keras.models.Sequential([
                keras.layers.Conv2D(filters=192, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5b_branch2')
            self.inception5b_branch3 = keras.models.Sequential([
                keras.layers.Conv2D(filters=48, kernel_size=(1,1), strides=1, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5b_branch3')
            self.inception5b_branch4 = keras.models.Sequential([
                keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
                keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
                keras.layers.BatchNormalization(),
            ], name='inception5b_branch4')
            self.inception5b_branch1.load_weights('./GoogLeNet_inception5b_branch1_weights')
            self.inception5b_branch2.load_weights('./GoogLeNet_inception5b_branch2_weights')
            self.inception5b_branch3.load_weights('./GoogLeNet_inception5b_branch3_weights')
            self.inception5b_branch4.load_weights('./GoogLeNet_inception5b_branch4_weights')

        if 'fully_connected' in self.layer_list:
            self.fully_connected = keras.models.Sequential([
                keras.layers.Flatten(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1000, activation='softmax'),
            ], name='fully_connected')
            self.fully_connected.load_weights('./GoogLeNet_fully_connected_weights')

    def get_random_input(self):
        if 'conv1' in self.layer_list:
            return np.zeros((1,32,32,3))
        elif 'conv2' in self.layer_list:
            return np.zeros((1,55,55,64))
        elif 'conv3' in self.layer_list:
            return np.zeros((1,55,55,192))
        elif 'inception3a' in self.layer_list:
            return np.zeros((1,27,27,192))
        elif 'inception3b' in self.layer_list:
            return np.zeros((1,27,27,256))
        elif 'inception4a' in self.layer_list:
            return np.zeros((1,13,13,480))
        elif 'inception4b' in self.layer_list:
            return np.zeros((1,13,13,512))
        elif 'inception4c' in self.layer_list:
            return np.zeros((1,13,13,512))
        elif 'inception4d' in self.layer_list:
            return np.zeros((1,13,13,512))
        elif 'inception4e' in self.layer_list:
            return np.zeros((1,13,13,528))
        elif 'inception5a' in self.layer_list:
            return np.zeros((1,6,6,832))
        elif 'inception5b' in self.layer_list:
            return np.zeros((1,6,6,832))
        elif 'fully_connected' in self.layer_list:
            return np.zeros(6*6*1024)

    def call(self, x):
        if 'conv1' in self.layer_list:
            x = self.resize(x)
            # conv1 and max pool()
            x = self.conv1(x)
            x = self.conv1_maxpool(x)
        if 'conv2' in self.layer_list:
            # conv2()
            x = self.conv2(x)
        if 'conv3' in self.layer_list:
            # conv3 and max pool()
            x = self.conv3(x)
            x = self.conv3_maxpool(x)
        if 'inception3a' in self.layer_list:
            # inception3a()
            branch1 = self.inception3a_branch1(x)
            branch2 = self.inception3a_branch2(x)
            branch3 = self.inception3a_branch3(x)
            branch4 = self.inception3a_branch4(x)
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
        if 'inception3b' in self.layer_list:
            # inception3b and max pool()
            branch1 = self.inception3b_branch1(x)
            branch2 = self.inception3b_branch2(x)
            branch3 = self.inception3b_branch3(x)
            branch4 = self.inception3b_branch4(x)
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
            x = self.inception3b_maxpool(x)
        if 'inception4a' in self.layer_list:
            # inception4a()
            branch1 = self.inception4a_branch1(x)
            branch2 = self.inception4a_branch2(x)
            branch3 = self.inception4a_branch3(x)
            branch4 = self.inception4a_branch4(x)
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
        if 'inception4b' in self.layer_list:
            # inception4b()
            branch1 = self.inception4b_branch1(x)
            branch2 = self.inception4b_branch2(x)
            branch3 = self.inception4b_branch3(x)
            branch4 = self.inception4b_branch4(x)
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
        if 'inception4c' in self.layer_list:
            # inception4c()
            branch1 = self.inception4c_branch1(x)
            branch2 = self.inception4c_branch2(x)
            branch3 = self.inception4c_branch3(x)
            branch4 = self.inception4c_branch4(x)
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
        if 'inception4d' in self.layer_list:
            # inception4d()
            branch1 = self.inception4d_branch1(x)
            branch2 = self.inception4d_branch2(x)
            branch3 = self.inception4d_branch3(x)
            branch4 = self.inception4d_branch4(x)
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
        if 'inception4e' in self.layer_list:
            # inception4e()
            branch1 = self.inception4e_branch1(x)
            branch2 = self.inception4e_branch2(x)
            branch3 = self.inception4e_branch3(x)
            branch4 = self.inception4e_branch4(x)
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
            x = self.inception4e_maxpool(x)
        if 'inception5a' in self.layer_list:
            # inception5a()
            branch1 = self.inception5a_branch1(x)
            branch2 = self.inception5a_branch2(x)
            branch3 = self.inception5a_branch3(x)
            branch4 = self.inception5a_branch4(x)
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
        if 'inception5b' in self.layer_list:
            # inception5b()
            branch1 = self.inception5b_branch1(x)
            branch2 = self.inception5b_branch2(x)
            branch3 = self.inception5b_branch3(x)
            branch4 = self.inception5b_branch4(x)
            x = tf.concat([branch1, branch2, branch3, branch4], -1)
        if 'fully_connected' in self.layer_list:
            # avg pool, flatten and fully_connected
            x = self.fully_connected(x)
        return x