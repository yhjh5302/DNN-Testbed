from YoloV2Model import *

if __name__ == '__main__':
    # load dataset
    epoch = 1
    batch_size = 16
    train_dataset, test_dataset = tfds.load('voc/2012', split=['train', 'test'], shuffle_files=True, download=True)
    train_dataset = (train_dataset.map(get_image_and_objects)
              .map(bbox_to_matrix)
              .map(resize_image)
              .map(scale_boxes_to_yolo_grid)
              .map(boxes_to_yx_hw_pairs)
              .map(normalize_image_zero_one)
              .shuffle(256)
              .batch(batch_size))
    test_dataset = (test_dataset.map(get_image_and_objects)
            .map(bbox_to_matrix)
            .map(resize_image)
            .map(scale_boxes_to_yolo_grid)
            .map(boxes_to_yx_hw_pairs)
            .map(normalize_image_zero_one))

    # model training
    model = YoloV2(name='YoloV2')

    optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    for e in range(epoch):
        print("Epoch #{}".format(e))
        epoch_losses = []
        for i, (x, y_true) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y_pred = model(x)
                loss_tensor = loss(y_pred, y_true, model)
                epoch_losses.append(loss_tensor)
                print("%3d/%3d Curr loss %.3f" % (i, len(train_dataset), sum(epoch_losses)/len(epoch_losses)))
                grads = tape.gradient(loss_tensor, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                if i == 30:
                    break

    # saving weights
    model.stage1_conv1.save_weights('./YoloV2_stage1_conv1_weights', save_format='tf')
    model.stage1_conv2.save_weights('./YoloV2_stage1_conv2_weights', save_format='tf')
    model.stage1_conv3.save_weights('./YoloV2_stage1_conv3_weights', save_format='tf')
    model.stage1_conv4.save_weights('./YoloV2_stage1_conv4_weights', save_format='tf')
    model.stage1_conv5.save_weights('./YoloV2_stage1_conv5_weights', save_format='tf')
    model.stage1_conv6.save_weights('./YoloV2_stage1_conv6_weights', save_format='tf')
    model.stage1_conv7.save_weights('./YoloV2_stage1_conv7_weights', save_format='tf')
    model.stage1_conv8.save_weights('./YoloV2_stage1_conv8_weights', save_format='tf')
    model.stage1_conv9.save_weights('./YoloV2_stage1_conv9_weights', save_format='tf')
    model.stage1_conv10.save_weights('./YoloV2_stage1_conv10_weights', save_format='tf')
    model.stage1_conv11.save_weights('./YoloV2_stage1_conv11_weights', save_format='tf')
    model.stage1_conv12.save_weights('./YoloV2_stage1_conv12_weights', save_format='tf')
    model.stage1_conv13.save_weights('./YoloV2_stage1_conv13_weights', save_format='tf')
    model.stage2_a_conv1.save_weights('./YoloV2_stage2_a_conv1_weights', save_format='tf')
    model.stage2_a_conv2.save_weights('./YoloV2_stage2_a_conv2_weights', save_format='tf')
    model.stage2_a_conv3.save_weights('./YoloV2_stage2_a_conv3_weights', save_format='tf')
    model.stage2_a_conv4.save_weights('./YoloV2_stage2_a_conv4_weights', save_format='tf')
    model.stage2_a_conv5.save_weights('./YoloV2_stage2_a_conv5_weights', save_format='tf')
    model.stage2_a_conv6.save_weights('./YoloV2_stage2_a_conv6_weights', save_format='tf')
    model.stage2_a_conv7.save_weights('./YoloV2_stage2_a_conv7_weights', save_format='tf')
    model.stage2_b_conv.save_weights('./YoloV2_stage2_b_conv_weights', save_format='tf')
    model.stage3_conv1.save_weights('./YoloV2_stage3_conv1_weights', save_format='tf')
    model.stage3_conv2.save_weights('./YoloV2_stage3_conv2_weights', save_format='tf')
    model.anchor_head.save_weights('./YoloV2_anchor_head_weights', save_format='tf')