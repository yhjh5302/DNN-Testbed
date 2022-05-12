import tensorflow as tf
from tensorflow import keras
import math
import time
from dag_config import *
from AlexNetModel import *
from VGGNetModel import *
from NiNModel import *
from ResNetModel import *
# from tensorflow.Total.VGGNetModel import VGGNet_layer


if __name__ == '__main__':
    set_gpu = True
    vram_limit = 1024
    if set_gpu:
        gpu_devices = tf.config.list_physical_devices(device_type='GPU')
        if not gpu_devices:
            raise ValueError('Cannot detect physical GPU device in TF')
        # tf.config.set_logical_device_configuration(gpu_devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=vram_limit)])
        tf.config.list_logical_devices()
    else:
        tf.config.set_visible_devices([], 'GPU')

    # load dataset
    _, (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.reshape(10000, 32, 32, 3).astype('float32') / 255

    # loading weights
    alexnet_in = AlexNet_layer(name='AlexNet-in', layer_list=PARTITION_INFOS["AlexNet-in"])
    alexnet_1 = AlexNet_layer(name='AlexNet-1', layer_list=PARTITION_INFOS["AlexNet-1"])
    alexnet_2 = AlexNet_layer(name='AlexNet-2', layer_list=PARTITION_INFOS["AlexNet-2"])
    alexnet_out = AlexNet_layer(name='AlexNet-out', layer_list=PARTITION_INFOS["AlexNet-out"])
    
    # loading weights
    vgg = VGGNet_layer(name='VGG', layer_list=PARTITION_INFOS['VGG'])
    nin = NiN_layer(name='NIN',  layer_list=PARTITION_INFOS['NiN'])

    resnet_in = ResNet_layer(name='ResNet-in',  layer_list=PARTITION_INFOS['ResNet-in'])
    resnet_1 = ResNet_layer(name='ResNet-CNN_1_2',  layer_list=PARTITION_INFOS['ResNet-CNN_1_2'])
    resnet_2 = ResNet_layer(name='ResNet-CNN_2_1',  layer_list=PARTITION_INFOS['ResNet-CNN_2_1'])
    resnet_3 = ResNet_layer(name='ResNet-CNN_3_2',  layer_list=PARTITION_INFOS['ResNet-CNN_3_2'])
    resnet_4 = ResNet_layer(name='ResNet-CNN_4_1',  layer_list=PARTITION_INFOS['ResNet-CNN_4_1'])
    resnet_5 = ResNet_layer(name='ResNet-CNN_5_2',  layer_list=PARTITION_INFOS['ResNet-CNN_5_2'])
    resnet_6 = ResNet_layer(name='ResNet-CNN_6_1',  layer_list=PARTITION_INFOS['ResNet-CNN_6_1'])
    resnet_7 = ResNet_layer(name='ResNet-CNN_7_2',  layer_list=PARTITION_INFOS['ResNet-CNN_7_2'])
    resnet_8 = ResNet_layer(name='ResNet-CNN_8_1',  layer_list=PARTITION_INFOS['ResNet-CNN_8_1'])
    resnet_9 = ResNet_layer(name='ResNet-CNN_9_2',  layer_list=PARTITION_INFOS['ResNet-CNN_9_2'])
    resnet_10 = ResNet_layer(name='ResNet-CNN_10_1',  layer_list=PARTITION_INFOS['ResNet-CNN_10_1'])
    resnet_11 = ResNet_layer(name='ResNet-CNN_11_2',  layer_list=PARTITION_INFOS['ResNet-CNN_11_2'])
    resnet_12 = ResNet_layer(name='ResNet-CNN_12_1',  layer_list=PARTITION_INFOS['ResNet-CNN_12_1'])
    resnet_13 = ResNet_layer(name='ResNet-CNN_13_2',  layer_list=PARTITION_INFOS['ResNet-CNN_13_2'])
    resnet_14 = ResNet_layer(name='ResNet-CNN_14_1',  layer_list=PARTITION_INFOS['ResNet-CNN_14_1'])
    resnet_15 = ResNet_layer(name='ResNet-CNN_15_2',  layer_list=PARTITION_INFOS['ResNet-CNN_15_2'])
    resnet_16 = ResNet_layer(name='ResNet-CNN_16_1',  layer_list=PARTITION_INFOS['ResNet-CNN_16_1'])
    resnet_17 = ResNet_layer(name='ResNet-CNN_17',  layer_list=PARTITION_INFOS['ResNet-CNN_17'])

    # for cudnn load
    alexnet_in(x_test[0:1])
    vgg(x_test[0:1])
    nin(x_test[0:1])
    resnet_in(x_test[0:1])


    batch_size = 1
    max = math.ceil(1000/batch_size)
    correct, l1, l2, l3, l4 = 0, 0, 0, 0, 0

    for i in range(max):
        start = i * batch_size
        if i == max-1:
            end = 1000
        else:
            end = (i+1) * batch_size

        inputs = x_test[start:end]
        test = y_test[start:end]

        t = time.time()
        x = alexnet_in(inputs)
        x = x.numpy()
        l1 += time.time() - t

        t = time.time()
        x_1 = alexnet_1(x)
        x_1 = x_1.numpy()
        l2 += time.time() - t

        t = time.time()
        x_2 = alexnet_2(x)
        x_2 = x_2.numpy()
        l3 += time.time() - t

        t = time.time()
        x = np.concatenate((x_1, x_2), axis=-1)
        x = alexnet_out(x)
        x = x.numpy()
        l4 += time.time() - t

        predict = tf.argmax(x, 1)
        answer = test.reshape(-1)
        correct += tf.reduce_sum(tf.cast(predict == answer, tf.float32))

    
    print("accuracy: {}%".format(correct/10))
    print("alexnet-in took {} ms".format(l1 / max * 1000))
    print("alexnet-1 took {} ms".format(l2 / max * 1000))
    print("alexnet-2 took {} ms".format(l3 / max * 1000))
    print("alexnet-out took {} ms".format(l4 / max * 1000))

    correct, l1 = 0, 0
    for i in range(max):
        start = i * batch_size
        if i == max-1:
            end = 1000
        else:
            end = (i+1) * batch_size

        inputs = x_test[start:end]
        test = y_test[start:end]

        t = time.time()
        x = vgg(inputs)
        x = x.numpy()
        l1 += time.time() - t

        predict = tf.argmax(x, 1)
        answer = test.reshape(-1)
        correct += tf.reduce_sum(tf.cast(predict == answer, tf.float32))

    print("vgg accuracy: {}%".format(correct/10))
    print("vgg took {} ms".format(l1*1000/max))

    correct, l1 = 0, 0
    for i in range(max):
        start = i * batch_size
        if i == max-1:
            end = 1000
        else:
            end = (i+1) * batch_size

        inputs = x_test[start:end]
        test = y_test[start:end]

        t = time.time()
        x = nin(inputs)
        x = x.numpy()
        l1 += time.time() - t

        predict = tf.argmax(x, 1)
        answer = test.reshape(-1)
        correct += tf.reduce_sum(tf.cast(predict == answer, tf.float32))

    print("nin accuracy: {}%".format(correct/10))
    print("nin took {} ms".format(l1*1000/max))

    correct, l_in, l1, l2, l3, l4 = 0, 0, 0, 0, 0, 0
    l5, l6, l7, l8, l9, l10 = 0, 0, 0, 0, 0, 0
    l11, l12, l13, l14, l15, l16, l17, l8 = 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(max):
        start = i * batch_size
        if i == max-1:
            end = 1000
        else:
            end = (i+1) * batch_size

        inputs = x_test[start:end]
        test = y_test[start:end]

        ex_t = time.time()
        x, shortcut, shortcut2 = resnet_in(inputs)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()        
        cur_t = time.time()
        l_in += cur_t - ex_t
        ex_t = cur_t

        x, shortcut, shortcut2 = resnet_1(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l1 += cur_t - ex_t
        ex_t = cur_t

        x, shortcut, shortcut2 = resnet_2(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l2 += cur_t - ex_t
        ex_t = cur_t

        x, shortcut, shortcut2 = resnet_3(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l3 += cur_t - ex_t
        ex_t = cur_t

        x, shortcut, shortcut2 = resnet_4(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l4 += cur_t - ex_t
        ex_t = cur_t
        
        x, shortcut, shortcut2 = resnet_5(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l5 += cur_t - ex_t
        ex_t = cur_t
        
        x, shortcut, shortcut2 = resnet_6(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l6 += cur_t - ex_t
        ex_t = cur_t
        
        x, shortcut, shortcut2 = resnet_7(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l7 += cur_t - ex_t
        ex_t = cur_t
        
        x, shortcut, shortcut2 = resnet_8(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l8 += cur_t - ex_t
        ex_t = cur_t
        
        x, shortcut, shortcut2 = resnet_9(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l9 += cur_t - ex_t
        ex_t = cur_t
        
        x, shortcut, shortcut2 = resnet_10(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l10 += cur_t - ex_t
        ex_t = cur_t
        
        x, shortcut, shortcut2 = resnet_11(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l11 += cur_t - ex_t
        ex_t = cur_t
        
        x, shortcut, shortcut2 = resnet_12(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l12 += cur_t - ex_t
        ex_t = cur_t
        
        x, shortcut, shortcut2 = resnet_13(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l13 += cur_t - ex_t
        ex_t = cur_t
        
        x, shortcut, shortcut2 = resnet_14(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l14 += cur_t - ex_t
        ex_t = cur_t

        x, shortcut, shortcut2 = resnet_15(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l15 += cur_t - ex_t
        ex_t = cur_t
        
        
        x, shortcut, shortcut2 = resnet_16(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l16 += cur_t - ex_t
        ex_t = cur_t
        
        
        x, shortcut, shortcut2 = resnet_17(x, shortcut, shortcut2)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l17 += cur_t - ex_t
        ex_t = cur_t
                
        predict = tf.argmax(x, 1)
        answer = test.reshape(-1)
        correct += tf.reduce_sum(tf.cast(predict == answer, tf.float32))

    print("resnet accuracy: {}%".format(correct/10))
    print("resnet-in took {} ms".format(l_in*1000/max))
    print("resnet-1 took {} ms".format(l1*1000/max))
    print("resnet-2 took {} ms".format(l2*1000/max))
    print("resnet-3 took {} ms".format(l3*1000/max))
    print("resnet-4 took {} ms".format(l4*1000/max))
    print("resnet-5 took {} ms".format(l5*1000/max))
    print("resnet-6 took {} ms".format(l6*1000/max))
    print("resnet-7 took {} ms".format(l7*1000/max))
    print("resnet-8 took {} ms".format(l8*1000/max))
    print("resnet-9 took {} ms".format(l9*1000/max))
    print("resnet-10 took {} ms".format(l10*1000/max))
    print("resnet-11 took {} ms".format(l11*1000/max))
    print("resnet-12 took {} ms".format(l12*1000/max))
    print("resnet-13 took {} ms".format(l13*1000/max))
    print("resnet-14 took {} ms".format(l14*1000/max))
    print("resnet-15 took {} ms".format(l15*1000/max))
    print("resnet-16 took {} ms".format(l16*1000/max))
    print("resnet-17 took {} ms".format(l17*1000/max))



    