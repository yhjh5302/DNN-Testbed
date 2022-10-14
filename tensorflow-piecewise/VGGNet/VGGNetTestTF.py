from VGGNetModel import *
import math
import time

if __name__ == '__main__':
    set_gpu = True
    vram_limit = 3072
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
    model = VGGNet(name='VGGNet')
    model.conv_1.load_weights('./VGGNet_conv_1_weights')
    model.conv_2.load_weights('./VGGNet_conv_2_weights')
    model.conv_3.load_weights('./VGGNet_conv_3_weights')
    model.conv_4.load_weights('./VGGNet_conv_4_weights')
    model.conv_5.load_weights('./VGGNet_conv_5_weights')
    model.conv_6.load_weights('./VGGNet_conv_6_weights')
    model.conv_7.load_weights('./VGGNet_conv_7_weights')
    model.conv_8.load_weights('./VGGNet_conv_8_weights')
    model.conv_9.load_weights('./VGGNet_conv_9_weights')
    model.conv_10.load_weights('./VGGNet_conv_10_weights')
    model.conv_11.load_weights('./VGGNet_conv_11_weights')
    model.conv_12.load_weights('./VGGNet_conv_12_weights')
    model.conv_13.load_weights('./VGGNet_conv_13_weights')
    model.classifier_1.load_weights('./VGGNet_classifier_1_weights')
    model.classifier_2.load_weights('./VGGNet_classifier_2_weights')
    model.classifier_3.load_weights('./VGGNet_classifier_3_weights')

    # for cudnn load
    model(x_test[0:1])

    batch_size = 1
    max = math.ceil(1000/batch_size)
    correct, total_took = 0, 0
    for i in range(max):
        start = i * batch_size
        if i == max-1:
            end = 1000
        else:
            end = (i+1) * batch_size

        inputs = x_test[start:end]
        test = y_test[start:end]

        t = time.time()
        x = tf.image.resize(inputs, size=(224, 224), method='nearest')
        x = model.conv_1(x)
        x = model.conv_2(x)
        x = model.maxpool_1(x)
        x = model.conv_3(x)
        x = model.conv_4(x)
        x = model.maxpool_2(x)
        x = model.conv_5(x)
        x = model.conv_6(x)
        x = model.conv_7(x)
        x = model.maxpool_3(x)
        x = model.conv_8(x)
        x = model.conv_9(x)
        x = model.conv_10(x)
        x = model.maxpool_4(x)
        x = model.conv_11(x)
        x = model.conv_12(x)
        x = model.conv_13(x)
        x = model.maxpool_5(x)
        x = model.flatten(x)
        x = model.classifier_1(x)
        x = model.classifier_2(x)
        x = model.classifier_3(x)

        temp_took = time.time() - t
        predict = tf.argmax(x, 1)
        answer = test.reshape(-1)
        print("#{} took: {:.3f}ms answer: {}".format(i+1, temp_took*1000, (predict == answer)[0]))
        total_took += temp_took
        correct += tf.reduce_sum(tf.cast(predict == answer, tf.float32))

    print("accuracy: {:.2f}%".format(correct/10))
    print("avg took {:.3f} ms".format(total_took))