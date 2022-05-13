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
    # _, (x_test, y_test) = keras.datasets.cifar10.load_data()
    # x_test = x_test.reshape(10000, 32, 32, 3).astype('float32') / 255

    # loading weights
    alexnet_in = AlexNet_layer(name='AlexNet-in', layer_list=PARTITION_INFOS["AlexNet-in"])
    alexnet_1 = AlexNet_layer(name='AlexNet-1', layer_list=PARTITION_INFOS["AlexNet-1"])
    alexnet_2 = AlexNet_layer(name='AlexNet-2', layer_list=PARTITION_INFOS["AlexNet-2"])
    alexnet_out = AlexNet_layer(name='AlexNet-out', layer_list=PARTITION_INFOS["AlexNet-out"])
    
    # loading weights
    vgg_1 = VGGNet_layer(name='VGG-1', layer_list=PARTITION_INFOS['VGG-1'])
    vgg_2 = VGGNet_layer(name='VGG-2', layer_list=PARTITION_INFOS['VGG-2'])
    vgg_3 = VGGNet_layer(name='VGG-3', layer_list=PARTITION_INFOS['VGG-3'])
    nin_1 = NiN_layer(name='NIN-1',  layer_list=PARTITION_INFOS['NiN-1'])
    nin_2 = NiN_layer(name='NIN-2',  layer_list=PARTITION_INFOS['NiN-2'])

    resnet_1_10 = ResNet_layer(name='ResNet-in',  layer_list=PARTITION_INFOS['ResNet-CNN_1-10'])
    resnet_11 = ResNet_layer(name='ResNet-CNN_11_2',  layer_list=PARTITION_INFOS['ResNet-CNN_11_2'])
    resnet_12 = ResNet_layer(name='ResNet-CNN_12_1',  layer_list=PARTITION_INFOS['ResNet-CNN_12_1'])
    resnet_13 = ResNet_layer(name='ResNet-CNN_13_2',  layer_list=PARTITION_INFOS['ResNet-CNN_13_2'])
    resnet_14 = ResNet_layer(name='ResNet-CNN_14_1',  layer_list=PARTITION_INFOS['ResNet-CNN_14_1'])
    resnet_15 = ResNet_layer(name='ResNet-CNN_15_2',  layer_list=PARTITION_INFOS['ResNet-CNN_15_2'])
    resnet_16 = ResNet_layer(name='ResNet-CNN_16_1',  layer_list=PARTITION_INFOS['ResNet-CNN_16_1'])
    resnet_17 = ResNet_layer(name='ResNet-CNN_17',  layer_list=PARTITION_INFOS['ResNet-CNN_17'])

    # for cudnn load
    alexnet_in(alexnet_in.get_random_input())
    alexnet_1(alexnet_1.get_random_input())
    alexnet_2(alexnet_2.get_random_input())
    alexnet_out(alexnet_out.get_random_input())
    vgg_1(vgg_1.get_random_input())
    vgg_2(vgg_2.get_random_input())
    vgg_3(vgg_3.get_random_input())
    nin_1(nin_1.get_random_input())
    nin_2(nin_2.get_random_input())
    resnet_1_10(resnet_1_10.get_random_input())
    resnet_11 = resnet_11(resnet_11.get_random_input())
    resnet_12 = resnet_12(resnet_12.get_random_input())
    resnet_13 = resnet_13(resnet_13.get_random_input())
    resnet_14 = resnet_14(resnet_14.get_random_input())
    resnet_15 = resnet_15(resnet_15.get_random_input())
    resnet_16 = resnet_16(resnet_16.get_random_input())
    resnet_17 = resnet_17(resnet_17.get_random_input())


    batch_size = 1
    max_times = 100
    l1, l2, l3, l4 = 0, 0, 0, 0
    inputs = alexnet_in.get_random_input()

    for i in range(max_times):
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
        x = alexnet_out((x_1, x_2))
        x = x.numpy()
        l4 += time.time() - t

    print("alexnet-in took {} ms".format(l1 / max_times * 1000))
    print("alexnet-1 took {} ms".format(l2 / max_times * 1000))
    print("alexnet-2 took {} ms".format(l3 / max_times * 1000))
    print("alexnet-out took {} ms".format(l4 / max_times * 1000))

    l1, l2, l3 = 0, 0, 0
    inputs = vgg_1.get_random_input()
    for i in range(max_times):

        t = time.time()
        x = vgg_1(inputs)
        x = x.numpy()
        l1 += time.time() - t

        t = time.time()
        x = vgg_2(x)
        x = x.numpy()
        l2 += time.time() - t

        t = time.time()
        x = vgg_3(x)
        x = x.numpy()
        l3 += time.time() - t

    print("vgg-1 took {} ms".format(l1*1000/max_times))
    print("vgg-2 took {} ms".format(l2*1000/max_times))
    print("vgg-3 took {} ms".format(l3*1000/max_times))

    l1, l2 = 0, 0
    inputs = nin_1.get_random_input()
    for i in range(max_times):
        t = time.time()
        x = nin_1(inputs)
        x = x.numpy()
        l1 += time.time() - t

        t = time.time()
        x = nin_2(x)
        x = x.numpy()
        l2 += time.time() - t

    print("nin-1 took {} ms".format(l1*1000/max_times))
    print("nin-2 took {} ms".format(l2*1000/max_times))

    l1, l2, l3, l4 = 0, 0, 0, 0,
    l5, l6, l7, l8 = 0, 0, 0, 0, 
    inputs = resnet_1_10.get_random_input()
    for i in range(max_times):

        ex_t = time.time()
        x, shortcut, shortcut2 = resnet_1_10(inputs)
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()        
        cur_t = time.time()
        l1 += cur_t - ex_t
        ex_t = cur_t

        x, shortcut, shortcut2 = resnet_11((x, shortcut, shortcut2))
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l2 += cur_t - ex_t
        ex_t = cur_t

        x, shortcut, shortcut2 = resnet_12((x, shortcut, shortcut2))
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l3 += cur_t - ex_t
        ex_t = cur_t

        x, shortcut, shortcut2 = resnet_13((x, shortcut, shortcut2))
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l4 += cur_t - ex_t
        ex_t = cur_t

        x, shortcut, shortcut2 = resnet_14((x, shortcut, shortcut2))
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l5 += cur_t - ex_t
        ex_t = cur_t
        
        x, shortcut, shortcut2 = resnet_15((x, shortcut, shortcut2))
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l6 += cur_t - ex_t
        ex_t = cur_t
        
        x, shortcut, shortcut2 = resnet_16((x, shortcut, shortcut2))
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l7 += cur_t - ex_t
        ex_t = cur_t        
        x, shortcut, shortcut2 = resnet_17((x, shortcut, shortcut2))
        x = x.numpy()
        if shortcut is not None and type(shortcut) is not np.ndarray:
            shortcut = shortcut.numpy()
        if shortcut2 is not None and type(shortcut2) is not np.ndarray:
            shortcut2 = shortcut2.numpy()
        cur_t = time.time()
        l8 += cur_t - ex_t
        ex_t = cur_t

    print("resnet accuracy: {}%".format(correct/10))
    print("resnet-in took {} ms".format(l_in*1000/max_times))
    print("resnet-1 took {} ms".format(l1*1000/max_times))
    print("resnet-2 took {} ms".format(l2*1000/max_times))
    print("resnet-3 took {} ms".format(l3*1000/max_times))
    print("resnet-4 took {} ms".format(l4*1000/max_times))
    print("resnet-5 took {} ms".format(l5*1000/max_times))
    print("resnet-6 took {} ms".format(l6*1000/max_times))
    print("resnet-7 took {} ms".format(l7*1000/max_times))
    print("resnet-8 took {} ms".format(l8*1000/max_times))
    print("resnet-9 took {} ms".format(l9*1000/max_times))
    print("resnet-10 took {} ms".format(l10*1000/max_times))
    print("resnet-11 took {} ms".format(l11*1000/max_times))
    print("resnet-12 took {} ms".format(l12*1000/max_times))
    print("resnet-13 took {} ms".format(l13*1000/max_times))
    print("resnet-14 took {} ms".format(l14*1000/max_times))
    print("resnet-15 took {} ms".format(l15*1000/max_times))
    print("resnet-16 took {} ms".format(l16*1000/max_times))
    print("resnet-17 took {} ms".format(l17*1000/max_times))



    