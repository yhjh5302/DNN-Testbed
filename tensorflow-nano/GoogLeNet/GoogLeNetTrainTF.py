import tensorflow as tf
from tensorflow import keras

class GoogLeNet(keras.Model):
    def __init__(self, name=None):
        super(GoogLeNet, self).__init__(name=name)
        self.conv1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='conv1')
        self.conv1_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
        self.conv2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='conv2')
        self.conv3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='conv3')
        self.conv3_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)

        self.inception3a_branch1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception3a_branch1')
        self.inception3a_branch2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=96, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception3a_branch2')
        self.inception3a_branch3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=16, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception3a_branch3')
        self.inception3a_branch4 = keras.models.Sequential([
            keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
            keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception3a_branch4')

        self.inception3b_branch1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception3b_branch1')
        self.inception3b_branch2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception3b_branch2')
        self.inception3b_branch3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=96, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception3b_branch3')
        self.inception3b_branch4 = keras.models.Sequential([
            keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
            keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception3b_branch4')
        self.inception3b_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)

        self.inception4a_branch1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=192, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4a_branch1')
        self.inception4a_branch2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=96, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=208, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4a_branch2')
        self.inception4a_branch3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=16, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=48, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4a_branch3')
        self.inception4a_branch4 = keras.models.Sequential([
            keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
            keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4a_branch4')

        self.inception4b_branch1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=160, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4b_branch1')
        self.inception4b_branch2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=112, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=224, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4b_branch2')
        self.inception4b_branch3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=24, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4b_branch3')
        self.inception4b_branch4 = keras.models.Sequential([
            keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
            keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4b_branch4')

        self.inception4c_branch1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4c_branch1')
        self.inception4c_branch2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4c_branch2')
        self.inception4c_branch3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=24, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4c_branch3')
        self.inception4c_branch4 = keras.models.Sequential([
            keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
            keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4c_branch4')

        self.inception4d_branch1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=112, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4d_branch1')
        self.inception4d_branch2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=144, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=288, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4d_branch2')
        self.inception4d_branch3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4d_branch3')
        self.inception4d_branch4 = keras.models.Sequential([
            keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
            keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4d_branch4')

        self.inception4e_branch1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4e_branch1')
        self.inception4e_branch2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=160, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=320, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4e_branch2')
        self.inception4e_branch3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4e_branch3')
        self.inception4e_branch4 = keras.models.Sequential([
            keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
            keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception4e_branch4')
        self.inception4e_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=2)

        self.inception5a_branch1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception5a_branch1')
        self.inception5a_branch2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=160, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=320, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception5a_branch2')
        self.inception5a_branch3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception5a_branch3')
        self.inception5a_branch4 = keras.models.Sequential([
            keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
            keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception5a_branch4')

        self.inception5b_branch1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception5b_branch1')
        self.inception5b_branch2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=192, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception5b_branch2')
        self.inception5b_branch3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=48, kernel_size=(1,1), strides=1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception5b_branch3')
        self.inception5b_branch4 = keras.models.Sequential([
            keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
            keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
        ], name='inception5b_branch4')

        self.fully_connected = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1000, activation='softmax'),
        ], name='fully_connected')

    def call(self, inputs):
        x = tf.image.resize(inputs, size=(224,224), method='nearest')
        # conv1 and max pool()
        x = self.conv1(x)
        x = self.conv1_maxpool(x)
        # conv2()
        x = self.conv2(x)
        # conv3 and max pool()
        x = self.conv3(x)
        x = self.conv3_maxpool(x)
        # inception3a()
        branch1 = self.inception3a_branch1(x)
        branch2 = self.inception3a_branch2(x)
        branch3 = self.inception3a_branch3(x)
        branch4 = self.inception3a_branch4(x)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # inception3b and max pool()
        branch1 = self.inception3b_branch1(x)
        branch2 = self.inception3b_branch2(x)
        branch3 = self.inception3b_branch3(x)
        branch4 = self.inception3b_branch4(x)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        x = self.inception3b_maxpool(x)
        # inception4a()
        branch1 = self.inception4a_branch1(x)
        branch2 = self.inception4a_branch2(x)
        branch3 = self.inception4a_branch3(x)
        branch4 = self.inception4a_branch4(x)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # inception4b()
        branch1 = self.inception4b_branch1(x)
        branch2 = self.inception4b_branch2(x)
        branch3 = self.inception4b_branch3(x)
        branch4 = self.inception4b_branch4(x)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # inception4c()
        branch1 = self.inception4c_branch1(x)
        branch2 = self.inception4c_branch2(x)
        branch3 = self.inception4c_branch3(x)
        branch4 = self.inception4c_branch4(x)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # inception4d()
        branch1 = self.inception4d_branch1(x)
        branch2 = self.inception4d_branch2(x)
        branch3 = self.inception4d_branch3(x)
        branch4 = self.inception4d_branch4(x)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # inception4e()
        branch1 = self.inception4e_branch1(x)
        branch2 = self.inception4e_branch2(x)
        branch3 = self.inception4e_branch3(x)
        branch4 = self.inception4e_branch4(x)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        x = self.inception4e_maxpool(x)
        # inception5a()
        branch1 = self.inception5a_branch1(x)
        branch2 = self.inception5a_branch2(x)
        branch3 = self.inception5a_branch3(x)
        branch4 = self.inception5a_branch4(x)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # inception5b()
        branch1 = self.inception5b_branch1(x)
        branch2 = self.inception5b_branch2(x)
        branch3 = self.inception5b_branch3(x)
        branch4 = self.inception5b_branch4(x)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # avg pool, flatten and fully_connected
        x = self.fully_connected(x)
        return x

if __name__ == '__main__':
    # load dataset
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.reshape(50000, 32, 32, 3).astype('float32') / 255

    # model training
    model = GoogLeNet(name='GoogLeNet')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001, momentum=0.9), metrics=['accuracy'])
    model.build((None,32,32,3))
    model.summary()
    history = model.fit(x_train, y_train, batch_size=64, epochs=10)

    # saving weights
    model.conv1.save_weights('./GoogLeNet_conv1_weights', save_format='tf')
    model.conv2.save_weights('./GoogLeNet_conv2_weights', save_format='tf')
    model.conv3.save_weights('./GoogLeNet_conv3_weights', save_format='tf')
    model.inception3a_branch1.save_weights('./GoogLeNet_inception3a_branch1_weights', save_format='tf')
    model.inception3a_branch2.save_weights('./GoogLeNet_inception3a_branch2_weights', save_format='tf')
    model.inception3a_branch3.save_weights('./GoogLeNet_inception3a_branch3_weights', save_format='tf')
    model.inception3a_branch4.save_weights('./GoogLeNet_inception3a_branch4_weights', save_format='tf')
    model.inception3b_branch1.save_weights('./GoogLeNet_inception3b_branch1_weights', save_format='tf')
    model.inception3b_branch2.save_weights('./GoogLeNet_inception3b_branch2_weights', save_format='tf')
    model.inception3b_branch3.save_weights('./GoogLeNet_inception3b_branch3_weights', save_format='tf')
    model.inception3b_branch4.save_weights('./GoogLeNet_inception3b_branch4_weights', save_format='tf')
    model.inception4a_branch1.save_weights('./GoogLeNet_inception4a_branch1_weights', save_format='tf')
    model.inception4a_branch2.save_weights('./GoogLeNet_inception4a_branch2_weights', save_format='tf')
    model.inception4a_branch3.save_weights('./GoogLeNet_inception4a_branch3_weights', save_format='tf')
    model.inception4a_branch4.save_weights('./GoogLeNet_inception4a_branch4_weights', save_format='tf')
    model.inception4b_branch1.save_weights('./GoogLeNet_inception4b_branch1_weights', save_format='tf')
    model.inception4b_branch2.save_weights('./GoogLeNet_inception4b_branch2_weights', save_format='tf')
    model.inception4b_branch3.save_weights('./GoogLeNet_inception4b_branch3_weights', save_format='tf')
    model.inception4b_branch4.save_weights('./GoogLeNet_inception4b_branch4_weights', save_format='tf')
    model.inception4c_branch1.save_weights('./GoogLeNet_inception4c_branch1_weights', save_format='tf')
    model.inception4c_branch2.save_weights('./GoogLeNet_inception4c_branch2_weights', save_format='tf')
    model.inception4c_branch3.save_weights('./GoogLeNet_inception4c_branch3_weights', save_format='tf')
    model.inception4c_branch4.save_weights('./GoogLeNet_inception4c_branch4_weights', save_format='tf')
    model.inception4d_branch1.save_weights('./GoogLeNet_inception4d_branch1_weights', save_format='tf')
    model.inception4d_branch2.save_weights('./GoogLeNet_inception4d_branch2_weights', save_format='tf')
    model.inception4d_branch3.save_weights('./GoogLeNet_inception4d_branch3_weights', save_format='tf')
    model.inception4d_branch4.save_weights('./GoogLeNet_inception4d_branch4_weights', save_format='tf')
    model.inception4e_branch1.save_weights('./GoogLeNet_inception4e_branch1_weights', save_format='tf')
    model.inception4e_branch2.save_weights('./GoogLeNet_inception4e_branch2_weights', save_format='tf')
    model.inception4e_branch3.save_weights('./GoogLeNet_inception4e_branch3_weights', save_format='tf')
    model.inception4e_branch4.save_weights('./GoogLeNet_inception4e_branch4_weights', save_format='tf')
    model.inception5a_branch1.save_weights('./GoogLeNet_inception5a_branch1_weights', save_format='tf')
    model.inception5a_branch2.save_weights('./GoogLeNet_inception5a_branch2_weights', save_format='tf')
    model.inception5a_branch3.save_weights('./GoogLeNet_inception5a_branch3_weights', save_format='tf')
    model.inception5a_branch4.save_weights('./GoogLeNet_inception5a_branch4_weights', save_format='tf')
    model.inception5b_branch1.save_weights('./GoogLeNet_inception5b_branch1_weights', save_format='tf')
    model.inception5b_branch2.save_weights('./GoogLeNet_inception5b_branch2_weights', save_format='tf')
    model.inception5b_branch3.save_weights('./GoogLeNet_inception5b_branch3_weights', save_format='tf')
    model.inception5b_branch4.save_weights('./GoogLeNet_inception5b_branch4_weights', save_format='tf')
    model.fully_connected.save_weights('./GoogLeNet_fully_connected_weights', save_format='tf')