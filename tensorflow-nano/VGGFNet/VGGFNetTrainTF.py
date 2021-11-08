import tensorflow as tf
from tensorflow import keras

class VGGFNet(keras.Model):
    def __init__(self, name=None):
        super(VGGFNet, self).__init__(name=name)
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
        x = tf.image.resize(inputs, size=(224,224), method='nearest')
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = self.flatten(x)
        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)
        return x

if __name__ == '__main__':
    # load dataset
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.reshape(50000, 32, 32, 3).astype('float32') / 255

    # model training
    model = VGGFNet(name='VGGFNet')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001, momentum=0.9), metrics=['accuracy'])
    model.build((None,32,32,3))
    model.summary()
    history = model.fit(x_train, y_train, batch_size=64, epochs=10)

    # saving weights
    model.features1.save_weights('./VGGFNet_features1_weights', save_format='tf')
    model.features2.save_weights('./VGGFNet_features2_weights', save_format='tf')
    model.features3.save_weights('./VGGFNet_features3_weights', save_format='tf')
    model.features4.save_weights('./VGGFNet_features4_weights', save_format='tf')
    model.features5.save_weights('./VGGFNet_features5_weights', save_format='tf')
    model.classifier1.save_weights('./VGGFNet_classifier1_weights', save_format='tf')
    model.classifier2.save_weights('./VGGFNet_classifier2_weights', save_format='tf')
    model.classifier3.save_weights('./VGGFNet_classifier3_weights', save_format='tf')