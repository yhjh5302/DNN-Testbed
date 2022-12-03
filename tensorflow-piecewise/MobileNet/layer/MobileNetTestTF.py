import tensorflow as tf
from tensorflow import keras
import math
import time

class MobileNet(keras.Model):
    def __init__(self, name=None):
        super(MobileNet, self).__init__(name=name)
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
        x = tf.image.resize(inputs, size=(224,224), method='nearest')
        x = self.conv1(x)
        x = self.separable_conv2(x)
        x = self.separable_conv3(x)
        x = self.separable_conv4(x)
        x = self.separable_conv5(x)
        x = self.separable_conv6(x)
        x = self.separable_conv7(x)
        x = self.separable_conv8(x)
        x = self.separable_conv9(x)
        x = self.separable_conv10(x)
        x = self.separable_conv11(x)
        x = self.separable_conv12(x)
        x = self.separable_conv13(x)
        x = self.separable_conv14(x)
        x = self.fully_connected(x)
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
    model = MobileNet(name='MobileNet')
    model.conv1.load_weights('./MobileNet_conv_1_weights')
    model.separable_conv2.load_weights('./MobileNet_separable_conv2_weights')
    model.separable_conv3.load_weights('./MobileNet_separable_conv3_weights')
    model.separable_conv4.load_weights('./MobileNet_separable_conv4_weights')
    model.separable_conv5.load_weights('./MobileNet_separable_conv5_weights')
    model.separable_conv6.load_weights('./MobileNet_separable_conv6_weights')
    model.separable_conv7.load_weights('./MobileNet_separable_conv7_weights')
    model.separable_conv8.load_weights('./MobileNet_separable_conv8_weights')
    model.separable_conv9.load_weights('./MobileNet_separable_conv9_weights')
    model.separable_conv10.load_weights('./MobileNet_separable_conv10_weights')
    model.separable_conv11.load_weights('./MobileNet_separable_conv11_weights')
    model.separable_conv12.load_weights('./MobileNet_separable_conv12_weights')
    model.separable_conv13.load_weights('./MobileNet_separable_conv13_weights')
    model.separable_conv14.load_weights('./MobileNet_separable_conv14_weights')
    model.fully_connected.load_weights('./MobileNet_fully_connected_weights')

    # for cudnn load
    batch_size = 1
    iter = 100
    model(x_test[0:batch_size])
    correct, total_time = 0, 0
    for i in range(iter-1):
        start = i * batch_size
        if i == iter-1:
            end = iter
        else:
            end = (i+1) * batch_size

        inputs = x_test[start:end]
        test = y_test[start:end]

        t = time.time()
        x = model(inputs)

        predict = tf.argmax(x, 1)
        answer = test.reshape(-1)
        correct += tf.reduce_sum(tf.cast(predict == answer, tf.float32))
        t = time.time() - t
        total_time += t
        print(t)

    print("accuracy: {:.2f}%".format(correct))
    print("took {:.3f} ms".format(total_time/iter*1000))