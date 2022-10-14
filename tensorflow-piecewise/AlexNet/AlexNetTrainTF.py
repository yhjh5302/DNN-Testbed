from AlexNetModel import *

if __name__ == '__main__':
    # load dataset
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.reshape(50000, 32, 32, 3).astype('float32') / 255

    # model training
    model = AlexNet(name='AlexNet')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001, momentum=0.9, clipnorm=1.), metrics=['accuracy'])
    model.build((None,32,32,3))
    model.summary()
    history = model.fit(x_train, y_train, batch_size=64, epochs=10)

    # saving weights
    model.conv_1.save_weights('./alexnet_conv_1_weights', save_format='tf')
    model.conv_2.save_weights('./alexnet_conv_2_weights', save_format='tf')
    model.conv_3.save_weights('./alexnet_conv_3_weights', save_format='tf')
    model.conv_4.save_weights('./alexnet_conv_4_weights', save_format='tf')
    model.conv_5.save_weights('./alexnet_conv_5_weights', save_format='tf')
    model.classifier_1.save_weights('./alexnet_classifier_1_weights', save_format='tf')
    model.classifier_2.save_weights('./alexnet_classifier_2_weights', save_format='tf')
    model.classifier_3.save_weights('./alexnet_classifier_3_weights', save_format='tf')