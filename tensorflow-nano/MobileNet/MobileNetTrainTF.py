import tensorflow as tf
from tensorflow import keras

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
    # load dataset
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.reshape(50000, 32, 32, 3).astype('float32') / 255

    # model training
    model = MobileNet(name='MobileNet')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001, momentum=0.9), metrics=['accuracy'])
    model.build((None,32,32,3))
    model.summary()
    history = model.fit(x_train, y_train, batch_size=64, epochs=10)

    # saving weights
    model.conv1.save_weights('./MobileNet_conv_1_weights', save_format='tf')
    model.separable_conv2.save_weights('./MobileNet_separable_conv2_weights', save_format='tf')
    model.separable_conv3.save_weights('./MobileNet_separable_conv3_weights', save_format='tf')
    model.separable_conv4.save_weights('./MobileNet_separable_conv4_weights', save_format='tf')
    model.separable_conv5.save_weights('./MobileNet_separable_conv5_weights', save_format='tf')
    model.separable_conv6.save_weights('./MobileNet_separable_conv6_weights', save_format='tf')
    model.separable_conv7.save_weights('./MobileNet_separable_conv7_weights', save_format='tf')
    model.separable_conv8.save_weights('./MobileNet_separable_conv8_weights', save_format='tf')
    model.separable_conv9.save_weights('./MobileNet_separable_conv9_weights', save_format='tf')
    model.separable_conv10.save_weights('./MobileNet_separable_conv10_weights', save_format='tf')
    model.separable_conv11.save_weights('./MobileNet_separable_conv11_weights', save_format='tf')
    model.separable_conv12.save_weights('./MobileNet_separable_conv12_weights', save_format='tf')
    model.separable_conv13.save_weights('./MobileNet_separable_conv13_weights', save_format='tf')
    model.separable_conv14.save_weights('./MobileNet_separable_conv14_weights', save_format='tf')
    model.fully_connected.save_weights('./MobileNet_fully_connected_weights', save_format='tf')