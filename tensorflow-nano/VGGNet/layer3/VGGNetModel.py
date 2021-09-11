import tensorflow as tf
from tensorflow import keras

class VGGNet_layer_1(keras.Model):
    def __init__(self, name=None):
        super(VGGNet_layer_1, self).__init__(name=name)
        self.resize = keras.layers.Resizing(height=224, width=224, interpolation='nearest', name='resize')
        self.features1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
            keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        ], name='features1')
        self.features2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
            keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        ], name='features2')
        self.features3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'),
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'),
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        ], name='features3')

    def call(self, inputs):
        x = self.resize(inputs)
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        return x

class VGGNet_layer_2(keras.Model):
    def __init__(self, name=None):
        super(VGGNet_layer_2, self).__init__(name=name)
        self.features4 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'),
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'),
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        ], name='features4')
        self.features5 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'),
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'),
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        ], name='features5')
        self.flatten = keras.layers.Flatten()

    def call(self, inputs):
        x = self.features4(inputs)
        x = self.features5(x)
        x = self.flatten(x)
        return x

class VGGNet_layer_3(keras.Model):
    def __init__(self, name=None):
        super(VGGNet_layer_3, self).__init__(name=name)
        self.classifier1 = keras.models.Sequential([
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
        ], name='classifier1')
        self.classifier2 = keras.models.Sequential([
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
        ], name='classifier2')
        self.classifier3 = keras.models.Sequential([
            keras.layers.Dense(1000),
        ], name='classifier3')

    def call(self, inputs):
        x = self.classifier1(inputs)
        x = self.classifier2(x)
        x = self.classifier3(x)
        return x