import tensorflow as tf
from tensorflow import keras
import math
import time

class AlexNet_layer_1(keras.Model):
    def __init__(self, name=None):
        super(AlexNet_layer_1, self).__init__(name=name)
        self.features_1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(11,11), strides=4, activation='relu', padding='same', input_shape=(224,224,3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=2),
        ], name='features_1')
        self.features_2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=192, kernel_size=(5,5), strides=1, activation='relu', padding='same', input_shape=(27,27,64)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=2),
        ], name='features_2')

    def call(self, inputs):
        x = tf.image.resize(inputs, size=(224,224), method='nearest')
        x = self.features_1(x)
        x = self.features_2(x)
        return x

class AlexNet_layer_2(keras.Model):
    def __init__(self, name=None):
        super(AlexNet_layer_2, self).__init__(name=name)
        self.features_3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,192)),
            keras.layers.BatchNormalization(),
        ], name='features_3')
        self.features_4 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,384)),
            keras.layers.BatchNormalization(),
        ], name='features_4')
        self.features_5 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same', input_shape=(13,13,256)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=2),
        ], name='features_5')

    def call(self, inputs):
        x = self.features_3(inputs)
        x = self.features_4(x)
        x = self.features_5(x)
        return x

class AlexNet_layer_3(keras.Model):
    def __init__(self, name=None):
        super(AlexNet_layer_3, self).__init__(name=name)
        self.classifier_1 = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu', input_shape=(256*6*6,)),
            keras.layers.Dropout(0.5),
        ], name='classifier_1')
        self.classifier_2 = keras.models.Sequential([
            keras.layers.Dense(4096, activation='relu', input_shape=(4096,)),
            keras.layers.Dropout(0.5),
        ], name='classifier_2')
        self.classifier_3 = keras.models.Sequential([
            keras.layers.Dense(1000, activation='softmax', input_shape=(4096,)),
        ], name='classifier_3')

    def call(self, inputs):
        x = self.classifier_1(inputs)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        return x

if __name__ == '__main__':
    set_gpu = True
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
    layer1 = AlexNet_layer_1(name='layer1')
    layer2 = AlexNet_layer_2(name='layer2')
    layer3 = AlexNet_layer_3(name='layer3')
    layer1.features_1.load_weights('./alexnet_features_1_weights')
    layer1.features_2.load_weights('./alexnet_features_2_weights')
    layer2.features_3.load_weights('./alexnet_features_3_weights')
    layer2.features_4.load_weights('./alexnet_features_4_weights')
    layer2.features_5.load_weights('./alexnet_features_5_weights')
    layer3.classifier_1.load_weights('./alexnet_classifier_1_weights')
    layer3.classifier_2.load_weights('./alexnet_classifier_2_weights')
    layer3.classifier_3.load_weights('./alexnet_classifier_3_weights')

    # for cudnn load
    layer1(x_test[0:1])

    batch_size = 1
    max = math.ceil(1000/batch_size)
    correct, l1, l2, l3 = 0, 0, 0, 0
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

        t = time.time()
        x = layer3(x)
        l3 += time.time() - t

        predict = tf.argmax(x, 1)
        answer = test.reshape(-1)
        correct += tf.reduce_sum(tf.cast(predict == answer, tf.float32))

    print("accuracy: {:.2f}%".format(correct/10))
    print("layer1 took {:.3f} ms".format(l1))
    print("layer2 took {:.3f} ms".format(l2))
    print("layer3 took {:.3f} ms".format(l3))