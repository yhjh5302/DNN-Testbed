import tensorflow as tf
from tensorflow import keras
import math
import time

class AlexNet_layer_1(keras.Model):
    def __init__(self, name=None):
        super(AlexNet_layer_1, self).__init__(name=name)
        self.resize = keras.layers.Resizing(height=224, width=224, interpolation='nearest', name='resize')
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
        x = self.resize(inputs)
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
    alex_layer1 = AlexNet_layer_1(name='alex_layer1')
    alex_layer2 = AlexNet_layer_2(name='alex_layer2')
    alex_layer3 = AlexNet_layer_3(name='alex_layer3')
    alex_layer1.features_1.load_weights('./alexnet_features_1_weights')
    alex_layer1.features_2.load_weights('./alexnet_features_2_weights')
    alex_layer2.features_3.load_weights('./alexnet_features_3_weights')
    alex_layer2.features_4.load_weights('./alexnet_features_4_weights')
    alex_layer2.features_5.load_weights('./alexnet_features_5_weights')
    alex_layer3.classifier_1.load_weights('./alexnet_classifier_1_weights')
    alex_layer3.classifier_2.load_weights('./alexnet_classifier_2_weights')
    alex_layer3.classifier_3.load_weights('./alexnet_classifier_3_weights')

    # loading weights
    vgg_layer1 = VGGNet_layer_1(name='vgg_layer1')
    vgg_layer2 = VGGNet_layer_2(name='vgg_layer2')
    vgg_layer3 = VGGNet_layer_3(name='vgg_layer3')
    vgg_layer1.features1.load_weights('./VGGNet_features1_weights')
    vgg_layer1.features2.load_weights('./VGGNet_features2_weights')
    vgg_layer1.features3.load_weights('./VGGNet_features3_weights')
    vgg_layer2.features4.load_weights('./VGGNet_features4_weights')
    vgg_layer2.features5.load_weights('./VGGNet_features5_weights')
    vgg_layer3.classifier1.load_weights('./VGGNet_classifier1_weights')
    vgg_layer3.classifier2.load_weights('./VGGNet_classifier2_weights')
    vgg_layer3.classifier3.load_weights('./VGGNet_classifier3_weights')

    # for cudnn load
    alex_layer1(x_test[0:1])
    vgg_layer1(x_test[0:1])

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
        x = alex_layer1(inputs)
        l1 += time.time() - t

        t = time.time()
        x = alex_layer2(x)
        l2 += time.time() - t

        t = time.time()
        x = alex_layer3(x)
        l3 += time.time() - t

        predict = tf.argmax(x, 1)
        answer = test.reshape(-1)
        correct += tf.reduce_sum(tf.cast(predict == answer, tf.float32))

    print("accuracy: {:.2f}%".format(correct/10))
    print("alex_layer1 took {:.3f} ms".format(l1))
    print("alex_layer2 took {:.3f} ms".format(l2))
    print("alex_layer3 took {:.3f} ms".format(l3))

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
        x = vgg_layer1(inputs)
        l1 += time.time() - t

        t = time.time()
        x = vgg_layer2(x)
        l2 += time.time() - t

        t = time.time()
        x = vgg_layer3(x)
        l3 += time.time() - t

        predict = tf.argmax(x, 1)
        answer = test.reshape(-1)
        correct += tf.reduce_sum(tf.cast(predict == answer, tf.float32))

    print("vgg accuracy: {:.2f}%".format(correct/10))
    print("vgg_layer1 took {:.3f} ms".format(l1))
    print("vgg_layer2 took {:.3f} ms".format(l2))
    print("vgg_layer3 took {:.3f} ms".format(l3))