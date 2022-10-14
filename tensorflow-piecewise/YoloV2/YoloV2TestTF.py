from YoloV2Model import *
from YoloV2Loss import loss
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
    test_dataset = tfds.load('voc/2007', split='test[75%:]', shuffle_files=True, download=True)
    test_dataset = (test_dataset.map(get_image_and_objects)
            .map(bbox_to_matrix)
            .map(resize_image)
            .map(scale_boxes_to_yolo_grid)
            .map(boxes_to_yx_hw_pairs)
            .map(normalize_image_zero_one))

    # loading weights
    model = YoloV2(name='YoloV2')
    model.stage1_conv1.load_weights('./YoloV2_stage1_conv1_weights')
    model.stage1_conv2.load_weights('./YoloV2_stage1_conv2_weights')
    model.stage1_conv3.load_weights('./YoloV2_stage1_conv3_weights')
    model.stage1_conv4.load_weights('./YoloV2_stage1_conv4_weights')
    model.stage1_conv5.load_weights('./YoloV2_stage1_conv5_weights')
    model.stage1_conv6.load_weights('./YoloV2_stage1_conv6_weights')
    model.stage1_conv7.load_weights('./YoloV2_stage1_conv7_weights')
    model.stage1_conv8.load_weights('./YoloV2_stage1_conv8_weights')
    model.stage1_conv9.load_weights('./YoloV2_stage1_conv9_weights')
    model.stage1_conv10.load_weights('./YoloV2_stage1_conv10_weights')
    model.stage1_conv11.load_weights('./YoloV2_stage1_conv11_weights')
    model.stage1_conv12.load_weights('./YoloV2_stage1_conv12_weights')
    model.stage1_conv13.load_weights('./YoloV2_stage1_conv13_weights')
    model.stage2_a_conv1.load_weights('./YoloV2_stage2_a_conv1_weights')
    model.stage2_a_conv2.load_weights('./YoloV2_stage2_a_conv2_weights')
    model.stage2_a_conv3.load_weights('./YoloV2_stage2_a_conv3_weights')
    model.stage2_a_conv4.load_weights('./YoloV2_stage2_a_conv4_weights')
    model.stage2_a_conv5.load_weights('./YoloV2_stage2_a_conv5_weights')
    model.stage2_a_conv6.load_weights('./YoloV2_stage2_a_conv6_weights')
    model.stage2_a_conv7.load_weights('./YoloV2_stage2_a_conv7_weights')
    model.stage2_b_conv.load_weights('./YoloV2_stage2_b_conv_weights')
    model.stage3_conv1.load_weights('./YoloV2_stage3_conv1_weights')
    model.stage3_conv2.load_weights('./YoloV2_stage3_conv2_weights')
    model.anchor_head.load_weights('./YoloV2_anchor_head_weights')

    # for cudnn load
    model(np.zeros((1,416,416,3)))

    num_images = 1000
    correct, total_took = 0, 0
    total_loss = []
    for i, (image, true_boxes) in enumerate(test_dataset.take(num_images)):
        image = tf.reshape(image, (1,416,416,3))
        true_boxes = tf.reshape(true_boxes, (1,5,4))
        
        t = time.time()
        x = model(image)
        
        temp_took = time.time() - t
        temp_loss = loss(x, true_boxes, model)
        print("#{} took: {:.3f}ms loss: {:.3f}".format(i+1, temp_took*1000, temp_loss))
        total_took += temp_took
        total_loss.append(temp_loss)
    print("average_loss: {:.3f}".format(sum(total_loss)/num_images))
    print("average_took {:.3f} ms".format(total_took))