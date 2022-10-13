import tensorflow as tf
from tensorflow import keras

class AlexNet(keras.Model):
    def __init__(self, name=None):
        super(AlexNet, self).__init__(name=name)
        self.conv_1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(11,11), strides=4, activation='relu', padding='same', input_shape=(224,224,3)),
            keras.layers.BatchNormalization(),
        ], name='conv_1')
        self.maxpool_1 = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
        self.conv_2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=192, kernel_size=(5,5), strides=1, activation='relu', padding='same', input_shape=(27,27,64)),
            keras.layers.BatchNormalization(),
        ], name='conv_2')
        self.maxpool_2 = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
        self.conv_3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,192)),
            keras.layers.BatchNormalization(),
        ], name='conv_3')
        self.conv_4 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,384)),
            keras.layers.BatchNormalization(),
        ], name='conv_4')
        self.conv_5 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,256)),
            keras.layers.BatchNormalization(),
        ], name='conv_5')
        self.maxpool_3 = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
        self.flatten = keras.layers.Flatten()
        self.classifier_1 = keras.models.Sequential([
            keras.layers.Dense(4096, activation='relu', input_shape=(256*6*6,)),
            # keras.layers.Dropout(0.5),
        ], name='classifier_1')
        self.classifier_2 = keras.models.Sequential([
            keras.layers.Dense(4096, activation='relu', input_shape=(4096,)),
            # keras.layers.Dropout(0.5),
        ], name='classifier_2')
        self.classifier_3 = keras.models.Sequential([
            keras.layers.Dense(1000, activation='softmax', input_shape=(4096,)),
        ], name='classifier_3')

    def call(self, inputs):
        x = tf.image.resize(inputs, size=(224, 224), method='nearest')
        x = self.conv_1(x)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.maxpool_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.maxpool_3(x)
        x = self.flatten(x)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        return x