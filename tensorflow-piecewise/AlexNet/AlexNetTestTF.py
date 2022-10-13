from AlexNetModel import *
import math
import time

if __name__ == '__main__':
    set_gpu = False # True
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
    model = AlexNet(name='AlexNet')
    model.conv_1.load_weights('./alexnet_conv_1_weights')
    model.conv_2.load_weights('./alexnet_conv_2_weights')
    model.conv_3.load_weights('./alexnet_conv_3_weights')
    model.conv_4.load_weights('./alexnet_conv_4_weights')
    model.conv_5.load_weights('./alexnet_conv_5_weights')
    model.classifier_1.load_weights('./alexnet_classifier_1_weights')
    model.classifier_2.load_weights('./alexnet_classifier_2_weights')
    model.classifier_3.load_weights('./alexnet_classifier_3_weights')

    # for cudnn load
    model(x_test[0:1])

    batch_size = 1
    max = math.ceil(1000/batch_size)
    correct, took = 0, 0
    for i in range(max):
        start = i * batch_size
        if i == max-1:
            end = 1000
        else:
            end = (i+1) * batch_size

        inputs = x_test[start:end]
        test = y_test[start:end]

        t = time.time()
        x = model(inputs)
        took += time.time() - t

        predict = tf.argmax(x, 1)
        answer = test.reshape(-1)
        correct += tf.reduce_sum(tf.cast(predict == answer, tf.float32))

    print("accuracy: {:.2f}%".format(correct/10))
    print("took {:.3f} ms".format(took))