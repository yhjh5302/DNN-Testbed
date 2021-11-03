import tensorflow as tf
from tensorflow import keras
import math
import time

class VGGFNet_layer_1(keras.Model):
    def __init__(self, name=None):
        super(VGGFNet_layer_1, self).__init__(name=name)
        self.features1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(11,11), strides=4, activation='relu'),
            keras.layers.AveragePooling2D(pool_size=(1, 1), strides=1),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        ], name='features1')
        self.features2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=1, activation='relu'),
            keras.layers.AveragePooling2D(pool_size=(1, 1), strides=1),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        ], name='features2')
        self.features3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
        ], name='features3')
        self.features4 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
        ], name='features4')
        self.features5 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        ], name='features5')

    def call(self, inputs):
        x = tf.image.resize(inputs, size=(224,224), method='nearest')
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        return x

class VGGFNet_layer_2(keras.Model):
    def __init__(self, name=None):
        super(VGGFNet_layer_2, self).__init__(name=name)
        self.flatten = keras.layers.Flatten()
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
        x = self.flatten(inputs)
        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)
        return x

if __name__ == '__main__':
    set_gpu = False
    vram_limit = 1024
    if set_gpu:
        gpu_devices = tf.config.list_physical_devices(device_type='GPU')
        if not gpu_devices:
            raise ValueError('Cannot detect physical GPU device in TF')
        tf.config.set_logical_device_configuration(gpu_devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=vram_limit)])
        tf.config.list_logical_devices()
    else:
        tf.config.set_visible_devices([], 'GPU')

    # load dataset
    _, (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.reshape(10000, 32, 32, 3).astype('float32') / 255

    # loading weights
    layer1 = VGGFNet_layer_1(name='layer1')
    layer2 = VGGFNet_layer_2(name='layer2')
    layer1.features1.load_weights('./VGGFNet_features1_weights')
    layer1.features2.load_weights('./VGGFNet_features2_weights')
    layer1.features3.load_weights('./VGGFNet_features3_weights')
    layer1.features4.load_weights('./VGGFNet_features4_weights')
    layer1.features5.load_weights('./VGGFNet_features5_weights')
    layer2.classifier1.load_weights('./VGGFNet_classifier1_weights')
    layer2.classifier2.load_weights('./VGGFNet_classifier2_weights')
    layer2.classifier3.load_weights('./VGGFNet_classifier3_weights')

    # for cudnn load
    layer1(x_test[0:1])

    batch_size = 1
    max = math.ceil(1000/batch_size)
    correct, l1, l2 = 0, 0, 0
    for i in range(max):
        start = i * batch_size
        if i == max-1:
            end = 1000
        else:
            end = (i+1) * batch_size

        inputs = x_test[start:end]
        test = y_test[start:end]

        t = time.time()
        x = layer1(inputs)
        l1 += time.time() - t

        t = time.time()
        x = layer2(x)
        l2 += time.time() - t

        predict = tf.argmax(x, 1)
        answer = test.reshape(-1)
        correct += tf.reduce_sum(tf.cast(predict == answer, tf.float32))

    print("accuracy: {:.2f}%".format(correct/10))
    print("layer1 took {:.3f} ms".format(l1))
    print("layer2 took {:.3f} ms".format(l2))