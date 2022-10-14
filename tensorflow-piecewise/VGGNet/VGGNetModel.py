import tensorflow as tf
from tensorflow import keras

class VGGNet(keras.Model):
    def __init__(self, name=None):
        super(VGGNet, self).__init__(name=name)
        self.conv_1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(224,224,3)),
            keras.layers.BatchNormalization(),
        ], name='conv_1')
        self.conv_2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(224,224,64)),
            keras.layers.BatchNormalization(),
        ], name='conv_2')
        self.maxpool_1 = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        self.conv_3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(112,112,64)),
            keras.layers.BatchNormalization(),
        ], name='conv_3')
        self.conv_4 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(112,112,128)),
            keras.layers.BatchNormalization(),
        ], name='conv_4')
        self.maxpool_2 = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        self.conv_5 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(56,56,128)),
            keras.layers.BatchNormalization(),
        ], name='conv_5')
        self.conv_6 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(56,56,256)),
            keras.layers.BatchNormalization(),
        ], name='conv_6')
        self.conv_7 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(56,56,256)),
            keras.layers.BatchNormalization(),
        ], name='conv_7')
        self.maxpool_3 = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        self.conv_8 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(28,28,256)),
            keras.layers.BatchNormalization(),
        ], name='conv_8')
        self.conv_9 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(28,28,512)),
            keras.layers.BatchNormalization(),
        ], name='conv_9')
        self.conv_10 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(28,28,512)),
            keras.layers.BatchNormalization(),
        ], name='conv_10')
        self.maxpool_4 = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        self.conv_11 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(14,14,512)),
            keras.layers.BatchNormalization(),
        ], name='conv_11')
        self.conv_12 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(14,14,512)),
            keras.layers.BatchNormalization(),
        ], name='conv_12')
        self.conv_13 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(14,14,512)),
            keras.layers.BatchNormalization(),
        ], name='conv_13')
        self.maxpool_5 = keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        self.flatten = keras.layers.Flatten()
        self.classifier_1 = keras.models.Sequential([
            keras.layers.Dense(4096, activation='relu', input_shape=(7*7*512,)),
        ], name='classifier_1')
        self.classifier_2 = keras.models.Sequential([
            keras.layers.Dense(4096, activation='relu', input_shape=(4096,)),
        ], name='classifier_2')
        self.classifier_3 = keras.models.Sequential([
            keras.layers.Dense(1000, activation='softmax', input_shape=(4096,)),
        ], name='classifier_3')

    def call(self, inputs):
        x = tf.image.resize(inputs, size=(224, 224), method='nearest')
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.maxpool_1(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.maxpool_2(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        x = self.maxpool_3(x)
        x = self.conv_8(x)
        x = self.conv_9(x)
        x = self.conv_10(x)
        x = self.maxpool_4(x)
        x = self.conv_11(x)
        x = self.conv_12(x)
        x = self.conv_13(x)
        x = self.maxpool_5(x)
        x = self.flatten(x)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        return x