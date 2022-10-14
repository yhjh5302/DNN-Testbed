from GoogLeNetModel import *
import math
import time

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
    model = GoogLeNet(name='GoogLeNet')
    model.conv1.load_weights('./GoogLeNet_conv1_weights')
    model.conv2.load_weights('./GoogLeNet_conv2_weights')
    model.conv3.load_weights('./GoogLeNet_conv3_weights')
    model.inception3a_branch1.load_weights('./GoogLeNet_inception3a_branch1_weights')
    model.inception3a_branch2_conv1.load_weights('./GoogLeNet_inception3a_branch2_conv1_weights')
    model.inception3a_branch2_conv2.load_weights('./GoogLeNet_inception3a_branch2_conv2_weights')
    model.inception3a_branch3_conv1.load_weights('./GoogLeNet_inception3a_branch3_conv1_weights')
    model.inception3a_branch3_conv2.load_weights('./GoogLeNet_inception3a_branch3_conv2_weights')
    model.inception3a_branch4_conv.load_weights('./GoogLeNet_inception3a_branch4_conv_weights')
    model.inception3b_branch1.load_weights('./GoogLeNet_inception3b_branch1_weights')
    model.inception3b_branch2_conv1.load_weights('./GoogLeNet_inception3b_branch2_conv1_weights')
    model.inception3b_branch2_conv2.load_weights('./GoogLeNet_inception3b_branch2_conv2_weights')
    model.inception3b_branch3_conv1.load_weights('./GoogLeNet_inception3b_branch3_conv1_weights')
    model.inception3b_branch3_conv2.load_weights('./GoogLeNet_inception3b_branch3_conv2_weights')
    model.inception3b_branch4_conv.load_weights('./GoogLeNet_inception3b_branch4_conv_weights')
    model.inception4a_branch1.load_weights('./GoogLeNet_inception4a_branch1_weights')
    model.inception4a_branch2_conv1.load_weights('./GoogLeNet_inception4a_branch2_conv1_weights')
    model.inception4a_branch2_conv2.load_weights('./GoogLeNet_inception4a_branch2_conv2_weights')
    model.inception4a_branch3_conv1.load_weights('./GoogLeNet_inception4a_branch3_conv1_weights')
    model.inception4a_branch3_conv2.load_weights('./GoogLeNet_inception4a_branch3_conv2_weights')
    model.inception4a_branch4_conv.load_weights('./GoogLeNet_inception4a_branch4_conv_weights')
    model.inception4b_branch1.load_weights('./GoogLeNet_inception4b_branch1_weights')
    model.inception4b_branch2_conv1.load_weights('./GoogLeNet_inception4b_branch2_conv1_weights')
    model.inception4b_branch2_conv2.load_weights('./GoogLeNet_inception4b_branch2_conv2_weights')
    model.inception4b_branch3_conv1.load_weights('./GoogLeNet_inception4b_branch3_conv1_weights')
    model.inception4b_branch3_conv2.load_weights('./GoogLeNet_inception4b_branch3_conv2_weights')
    model.inception4b_branch4_conv.load_weights('./GoogLeNet_inception4b_branch4_conv_weights')
    model.inception4c_branch1.load_weights('./GoogLeNet_inception4c_branch1_weights')
    model.inception4c_branch2_conv1.load_weights('./GoogLeNet_inception4c_branch2_conv1_weights')
    model.inception4c_branch2_conv2.load_weights('./GoogLeNet_inception4c_branch2_conv2_weights')
    model.inception4c_branch3_conv1.load_weights('./GoogLeNet_inception4c_branch3_conv1_weights')
    model.inception4c_branch3_conv2.load_weights('./GoogLeNet_inception4c_branch3_conv2_weights')
    model.inception4c_branch4_conv.load_weights('./GoogLeNet_inception4c_branch4_conv_weights')
    model.inception4d_branch1.load_weights('./GoogLeNet_inception4d_branch1_weights')
    model.inception4d_branch2_conv1.load_weights('./GoogLeNet_inception4d_branch2_conv1_weights')
    model.inception4d_branch2_conv2.load_weights('./GoogLeNet_inception4d_branch2_conv2_weights')
    model.inception4d_branch3_conv1.load_weights('./GoogLeNet_inception4d_branch3_conv1_weights')
    model.inception4d_branch3_conv2.load_weights('./GoogLeNet_inception4d_branch3_conv2_weights')
    model.inception4d_branch4_conv.load_weights('./GoogLeNet_inception4d_branch4_conv_weights')
    model.inception4e_branch1.load_weights('./GoogLeNet_inception4e_branch1_weights')
    model.inception4e_branch2_conv1.load_weights('./GoogLeNet_inception4e_branch2_conv1_weights')
    model.inception4e_branch2_conv2.load_weights('./GoogLeNet_inception4e_branch2_conv2_weights')
    model.inception4e_branch3_conv1.load_weights('./GoogLeNet_inception4e_branch3_conv1_weights')
    model.inception4e_branch3_conv2.load_weights('./GoogLeNet_inception4e_branch3_conv2_weights')
    model.inception4e_branch4_conv.load_weights('./GoogLeNet_inception4e_branch4_conv_weights')
    model.inception5a_branch1.load_weights('./GoogLeNet_inception5a_branch1_weights')
    model.inception5a_branch2_conv1.load_weights('./GoogLeNet_inception5a_branch2_conv1_weights')
    model.inception5a_branch2_conv2.load_weights('./GoogLeNet_inception5a_branch2_conv2_weights')
    model.inception5a_branch3_conv1.load_weights('./GoogLeNet_inception5a_branch3_conv1_weights')
    model.inception5a_branch3_conv2.load_weights('./GoogLeNet_inception5a_branch3_conv2_weights')
    model.inception5a_branch4_conv.load_weights('./GoogLeNet_inception5a_branch4_conv_weights')
    model.inception5b_branch1.load_weights('./GoogLeNet_inception5b_branch1_weights')
    model.inception5b_branch2_conv1.load_weights('./GoogLeNet_inception5b_branch2_conv1_weights')
    model.inception5b_branch2_conv2.load_weights('./GoogLeNet_inception5b_branch2_conv2_weights')
    model.inception5b_branch3_conv1.load_weights('./GoogLeNet_inception5b_branch3_conv1_weights')
    model.inception5b_branch3_conv2.load_weights('./GoogLeNet_inception5b_branch3_conv2_weights')
    model.inception5b_branch4_conv.load_weights('./GoogLeNet_inception5b_branch4_conv_weights')
    model.fully_connected.load_weights('./GoogLeNet_fully_connected_weights')

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
        x = tf.image.resize(inputs, size=(224,224), method='nearest')
        # conv1 and max pool()
        x = model.conv1(x)
        x = model.conv1_maxpool(x)
        # conv2()
        x = model.conv2(x)
        # conv3 and max pool()
        x = model.conv3(x)
        x = model.conv3_maxpool(x)
        # inception3a()
        branch1 = model.inception3a_branch1(x)
        branch2 = model.inception3a_branch2_conv1(x)
        branch2 = model.inception3a_branch2_conv2(branch2)
        branch3 = model.inception3a_branch3_conv1(x)
        branch3 = model.inception3a_branch3_conv2(branch3)
        branch4 = model.inception3a_branch4_maxpool(x)
        branch4 = model.inception3a_branch4_conv(branch4)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # inception3b and max pool()
        branch1 = model.inception3b_branch1(x)
        branch2 = model.inception3b_branch2_conv1(x)
        branch2 = model.inception3b_branch2_conv2(branch2)
        branch3 = model.inception3b_branch3_conv1(x)
        branch3 = model.inception3b_branch3_conv2(branch3)
        branch4 = model.inception3b_branch4_maxpool(x)
        branch4 = model.inception3b_branch4_conv(branch4)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        x = model.inception3b_maxpool(x)
        # inception4a()
        branch1 = model.inception4a_branch1(x)
        branch2 = model.inception4a_branch2_conv1(x)
        branch2 = model.inception4a_branch2_conv2(branch2)
        branch3 = model.inception4a_branch3_conv1(x)
        branch3 = model.inception4a_branch3_conv2(branch3)
        branch4 = model.inception4a_branch4_maxpool(x)
        branch4 = model.inception4a_branch4_conv(branch4)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # inception4b()
        branch1 = model.inception4b_branch1(x)
        branch2 = model.inception4b_branch2_conv1(x)
        branch2 = model.inception4b_branch2_conv2(branch2)
        branch3 = model.inception4b_branch3_conv1(x)
        branch3 = model.inception4b_branch3_conv2(branch3)
        branch4 = model.inception4b_branch4_maxpool(x)
        branch4 = model.inception4b_branch4_conv(branch4)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # inception4c()
        branch1 = model.inception4c_branch1(x)
        branch2 = model.inception4c_branch2_conv1(x)
        branch2 = model.inception4c_branch2_conv2(branch2)
        branch3 = model.inception4c_branch3_conv1(x)
        branch3 = model.inception4c_branch3_conv2(branch3)
        branch4 = model.inception4c_branch4_maxpool(x)
        branch4 = model.inception4c_branch4_conv(branch4)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # inception4d()
        branch1 = model.inception4d_branch1(x)
        branch2 = model.inception4d_branch2_conv1(x)
        branch2 = model.inception4d_branch2_conv2(branch2)
        branch3 = model.inception4d_branch3_conv1(x)
        branch3 = model.inception4d_branch3_conv2(branch3)
        branch4 = model.inception4d_branch4_maxpool(x)
        branch4 = model.inception4d_branch4_conv(branch4)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # inception4e()
        branch1 = model.inception4e_branch1(x)
        branch2 = model.inception4e_branch2_conv1(x)
        branch2 = model.inception4e_branch2_conv2(branch2)
        branch3 = model.inception4e_branch3_conv1(x)
        branch3 = model.inception4e_branch3_conv2(branch3)
        branch4 = model.inception4e_branch4_maxpool(x)
        branch4 = model.inception4e_branch4_conv(branch4)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        x = model.inception4e_maxpool(x)
        # inception5a()
        branch1 = model.inception5a_branch1(x)
        branch2 = model.inception5a_branch2_conv1(x)
        branch2 = model.inception5a_branch2_conv2(branch2)
        branch3 = model.inception5a_branch3_conv1(x)
        branch3 = model.inception5a_branch3_conv2(branch3)
        branch4 = model.inception5a_branch4_maxpool(x)
        branch4 = model.inception5a_branch4_conv(branch4)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # inception5b()
        branch1 = model.inception5b_branch1(x)
        branch2 = model.inception5b_branch2_conv1(x)
        branch2 = model.inception5b_branch2_conv2(branch2)
        branch3 = model.inception5b_branch3_conv1(x)
        branch3 = model.inception5b_branch3_conv2(branch3)
        branch4 = model.inception5b_branch4_maxpool(x)
        branch4 = model.inception5b_branch4_conv(branch4)
        x = tf.concat([branch1, branch2, branch3, branch4], -1)
        # avg pool, flatten and fully_connected
        x = model.flatten(x)
        x = model.fully_connected(x)

        temp_took = time.time() - t
        predict = tf.argmax(x, 1)
        answer = test.reshape(-1)
        print("#{} took: {:.3f}ms answer: {}".format(i+1, temp_took*1000, (predict == answer)[0]))
        total_took += temp_took
        correct += tf.reduce_sum(tf.cast(predict == answer, tf.float32))

    print("accuracy: {:.2f}%".format(correct/10))
    print("avg took {:.3f} ms".format(total_took))