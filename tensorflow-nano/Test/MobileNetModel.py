import tensorflow as tf
from tensorflow import keras

class MobileNet_layer_1(keras.Model):
    def __init__(self, name=None):
        super(MobileNet_layer_1, self).__init__(name=name)
        self.conv1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='conv1')
        self.separable_conv2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
        ], name='separable_conv2')
        self.separable_conv3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
        ], name='separable_conv3')
        self.separable_conv4 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
        ], name='separable_conv4')
        self.separable_conv5 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=2, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
        ], name='separable_conv5')

    def call(self, inputs):
        x = tf.image.resize(inputs, size=(224,224), method='nearest')
        x = self.conv1(x)
        x = self.separable_conv2(x)
        x = self.separable_conv3(x)
        x = self.separable_conv4(x)
        x = self.separable_conv5(x)
        return x

class MobileNet_layer_2(keras.Model):
    def __init__(self, name=None):
        super(MobileNet_layer_2, self).__init__(name=name)
        self.separable_conv6 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
        ], name='separable_conv6')
        self.separable_conv7 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=2, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
        ], name='separable_conv7')
        self.separable_conv8 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
        ], name='separable_conv8')
        self.separable_conv9 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
        ], name='separable_conv9')
        self.separable_conv10 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
        ], name='separable_conv10')

    def call(self, inputs):
        x = self.separable_conv6(inputs)
        x = self.separable_conv7(x)
        x = self.separable_conv8(x)
        x = self.separable_conv9(x)
        x = self.separable_conv10(x)
        return x

class MobileNet_layer_3(keras.Model):
    def __init__(self, name=None):
        super(MobileNet_layer_3, self).__init__(name=name)
        self.separable_conv11 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
        ], name='separable_conv11')
        self.separable_conv12 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
        ], name='separable_conv12')
        self.separable_conv13 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=2, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
        ], name='separable_conv13')
        self.separable_conv14 = keras.models.Sequential([
            keras.layers.Conv2D(filters=1024, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
        ], name='separable_conv14')
        self.fully_connected = keras.models.Sequential([
            keras.layers.AveragePooling2D(pool_size=(7, 7)),
            keras.layers.Flatten(),
            keras.layers.Dense(1000, activation='softmax'),
        ], name='fully_connected')

    def call(self, inputs):
        x = self.separable_conv11(inputs)
        x = self.separable_conv12(x)
        x = self.separable_conv13(x)
        x = self.separable_conv14(x)
        x = self.fully_connected(x)
        return x