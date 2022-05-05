import tensorflow as tf
from tensorflow import keras

class AlexNet(keras.Model):
    def __init__(self, name=None):
        super(AlexNet, self).__init__(name=name)
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
        x = self.resize(inputs)
        x = self.features_1(x)
        x = self.features_2(x)
        x = self.features_3(x)
        x = self.features_4(x)
        x = self.features_5(x)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        return x

class VGGNet(keras.Model):
    def __init__(self, name=None):
        super(VGGNet, self).__init__(name=name)
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
        x = self.resize(inputs)
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
    model = VGGNet(name='VGGNet')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001, momentum=0.9), metrics=['accuracy'])
    model.build((None,32,32,3))
    model.summary()
    history = model.fit(x_train, y_train, batch_size=64, epochs=10)

    # saving weights
    model.features1.save_weights('./VGGNet_features1_weights', save_format='tf')
    model.features2.save_weights('./VGGNet_features2_weights', save_format='tf')
    model.features3.save_weights('./VGGNet_features3_weights', save_format='tf')
    model.features4.save_weights('./VGGNet_features4_weights', save_format='tf')
    model.features5.save_weights('./VGGNet_features5_weights', save_format='tf')
    model.classifier1.save_weights('./VGGNet_classifier1_weights', save_format='tf')
    model.classifier2.save_weights('./VGGNet_classifier2_weights', save_format='tf')
    model.classifier3.save_weights('./VGGNet_classifier3_weights', save_format='tf')

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