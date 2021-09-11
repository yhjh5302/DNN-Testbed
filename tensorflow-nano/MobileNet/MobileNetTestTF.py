import tensorflow as tf
from tensorflow import keras
import math
import time

class MobileNet_layer_1(keras.Model):
    def __init__(self, name=None):
        super(MobileNet_layer_1, self).__init__(name=name)
        self.resize = keras.layers.Resizing(height=224, width=224, interpolation='nearest', name='resize')
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
        x = self.resize(inputs)
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

if __name__ == '__main__':
    # load dataset
    _, (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.reshape(10000, 32, 32, 3).astype('float32') / 255

    # loading weights
    layer1 = MobileNet_layer_1(name='layer1')
    layer2 = MobileNet_layer_2(name='layer2')
    layer3 = MobileNet_layer_3(name='layer3')
    layer1.conv1.load_weights('./MobileNet_conv_1_weights')
    layer1.separable_conv2.load_weights('./MobileNet_separable_conv2_weights')
    layer1.separable_conv3.load_weights('./MobileNet_separable_conv3_weights')
    layer1.separable_conv4.load_weights('./MobileNet_separable_conv4_weights')
    layer1.separable_conv5.load_weights('./MobileNet_separable_conv5_weights')
    layer2.separable_conv6.load_weights('./MobileNet_separable_conv6_weights')
    layer2.separable_conv7.load_weights('./MobileNet_separable_conv7_weights')
    layer2.separable_conv8.load_weights('./MobileNet_separable_conv8_weights')
    layer2.separable_conv9.load_weights('./MobileNet_separable_conv9_weights')
    layer2.separable_conv10.load_weights('./MobileNet_separable_conv10_weights')
    layer3.separable_conv11.load_weights('./MobileNet_separable_conv11_weights')
    layer3.separable_conv12.load_weights('./MobileNet_separable_conv12_weights')
    layer3.separable_conv13.load_weights('./MobileNet_separable_conv13_weights')
    layer3.separable_conv14.load_weights('./MobileNet_separable_conv14_weights')
    layer3.fully_connected.load_weights('./MobileNet_fully_connected_weights')

    # for cudnn load
    layer1(x_test[0:1])

    batch_size = 1
    max = math.ceil(10000/batch_size)
    correct, l1, l2, l3 = 0, 0, 0, 0
    for i in range(max):
        start = i * batch_size
        if i == max-1:
            end = 10000
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

    print("accuracy: {:.2f}%".format(correct/100))
    print("layer1 took {:.3f}ms".format(l1/10))
    print("layer2 took {:.3f}ms".format(l2/10))
    print("layer3 took {:.3f}ms".format(l3/10))