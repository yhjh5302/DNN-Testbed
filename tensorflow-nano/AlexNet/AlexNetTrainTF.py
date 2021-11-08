import tensorflow as tf
from tensorflow import keras

class AlexNet(keras.Model):
    def __init__(self, name=None):
        super(AlexNet, self).__init__(name=name)
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
        x = tf.image.resize(inputs, size=(224,224), method='nearest')
        x = self.features_1(x)
        x = self.features_2(x)
        x = self.features_3(x)
        x = self.features_4(x)
        x = self.features_5(x)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        return x

if __name__ == '__main__':
    # load dataset
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.reshape(50000, 32, 32, 3).astype('float32') / 255

    # model training
    model = AlexNet(name='AlexNet')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001, momentum=0.9), metrics=['accuracy'])
    model.build((None,32,32,3))
    model.summary()
    history = model.fit(x_train, y_train, batch_size=64, epochs=10)

    # saving weights
    model.features_1.save_weights('./alexnet_features_1_weights', save_format='tf')
    model.features_2.save_weights('./alexnet_features_2_weights', save_format='tf')
    model.features_3.save_weights('./alexnet_features_3_weights', save_format='tf')
    model.features_4.save_weights('./alexnet_features_4_weights', save_format='tf')
    model.features_5.save_weights('./alexnet_features_5_weights', save_format='tf')
    model.classifier_1.save_weights('./alexnet_classifier_1_weights', save_format='tf')
    model.classifier_2.save_weights('./alexnet_classifier_2_weights', save_format='tf')
    model.classifier_3.save_weights('./alexnet_classifier_3_weights', save_format='tf')