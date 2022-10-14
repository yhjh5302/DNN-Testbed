import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

''' https://github.com/FlorisHoogenboom/yolo-v2-tf-2/blob/master/Demo.ipynb '''

class AnchorLayer(keras.Model):
    def __init__(self, grid_height, grid_width, anchors, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.anchors = anchors
        self.n_classes = n_classes
        self.boxes_conv = keras.layers.Conv2D(
            len(self.anchors) * 4,
            (1, 1),
            strides=(1, 1),
            padding='same',
            name='boxes',
            activation='linear',
        )
        self.confidence_conv = keras.layers.Conv2D(
            len(self.anchors) * 1,
            (1, 1),
            strides=(1, 1),
            padding='same',
            name='conf',
            activation='sigmoid',
        )
        self.classes_conv = keras.layers.Conv2D(
            self.n_classes,
            (1, 1),
            strides=(1, 1),
            padding='same',
            name='probas',
            activation='softmax'
        )

    def compute_boxes(self, input):
        unaligned_preds = self.boxes_conv(input)
        preds = tf.reshape(
            unaligned_preds,
            (-1, self.grid_height, self.grid_width, len(self.anchors), 4)
        )
        yx_base = self.base_anchor_boxes[..., 0:2]
        hw_base = self.base_anchor_boxes[..., 2:4]
        # Substract 1/2 from the result of applying sigmoid to get a
        # value in the range (-0.5, 0.5). Hence, they are fluctuating around
        # the grid centers
        yx = yx_base + (tf.sigmoid(preds[..., 0:2]) - 0.5)
        # Clip the WH values at an arbitrary boundary such that we prevent
        # them from exploding to infinity during training
        # TODO: Fix the arbitrary clipping bound
        hw = hw_base * tf.clip_by_value(tf.exp(preds[..., 2:4]), 0.00001, 130)
        return tf.concat([yx, hw], axis=-1)

    def compute_confidences(self, input):
        unaligned_confs = self.confidence_conv(input)
        confs = tf.reshape(
            unaligned_confs,
            (-1, self.grid_height, self.grid_width, len(self.anchors), 1)
        )
        return confs

    def compute_classes(self, input):
        unaligned_classes = self.classes_conv(input)
        classes = tf.tile(
            unaligned_classes[:, :, :, None, :],
            [1, 1, 1, len(self.anchors), 1]
        )
        return classes

    @property
    def base_anchor_boxes(self):
        h, w = self.grid_height, self.grid_width
        y_centroids = tf.tile(
            tf.range(h, dtype='float')[:, None],
            [1, w]
        ) + 0.5
        x_centroids = tf.tile(
            tf.range(w, dtype='float')[None, :],
            [h, 1]
        ) + 0.5
        yx_centroids = tf.concat(
            [y_centroids[..., None, None], x_centroids[..., None, None]],
            axis=-1
        )
        yx_centroids = tf.tile(yx_centroids, [1, 1, len(self.anchors), 1])
        hw = tf.tile(
            tf.constant(self.anchors)[None, None, ...],
            [h, w, 1, 1]
        )
        return tf.concat([yx_centroids, hw], axis=-1)

    def call(self, inputs, training=None, mask=None):
        return tf.concat([
            self.compute_boxes(inputs),
            self.compute_confidences(inputs),
            self.compute_classes(inputs)
        ], axis=-1)

    @property
    def n_anchors(self):
        return len(self.anchors)

class YoloV2(keras.Model):
    def __init__(self, name=None):
        super(YoloV2, self).__init__(name=name)
        self.IMAGE_H, self.IMAGE_W = 416, 416
        self.GRID_H, self.GRID_W  = 13, 13  # grid size = image size / 32
        self.LAMBDA_NO_OBJECT = 1.0
        self.LAMBDA_OBJECT    = 5.0
        self.LAMBDA_COORD     = 1.0
        self.LAMBDA_CLASS     = 1.0
        self.ANCHORS = [
            [2., 2.],
            [2., 3.],
            [3., 2.],
            [6., 6.],
            [3., 8.],
            [8., 3.],
        ]
        self.BOX = 10
        self.block_size = 2
        self.num_classes = 20

        self.stage1_conv1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage1_conv1')
        self.stage1_maxpl1 = keras.layers.MaxPool2D(pool_size=(2,2))
        self.stage1_conv2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage1_conv2')
        self.stage1_maxpl2 = keras.layers.MaxPool2D(pool_size=(2,2))
        self.stage1_conv3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage1_conv3')
        self.stage1_conv4 = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage1_conv4')
        self.stage1_conv5 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage1_conv5')
        self.stage1_maxpl3 = keras.layers.MaxPool2D(pool_size=(2,2))
        self.stage1_conv6 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage1_conv6')
        self.stage1_conv7 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage1_conv7')
        self.stage1_conv8 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage1_conv8')
        self.stage1_maxpl4 = keras.layers.MaxPool2D(pool_size=(2,2))
        self.stage1_conv9 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage1_conv9')
        self.stage1_conv10 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage1_conv10')
        self.stage1_conv11 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage1_conv11')
        self.stage1_conv12 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage1_conv12')
        self.stage1_conv13 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage1_conv13')

        self.stage2_a_maxpl = keras.layers.MaxPool2D(pool_size=(2,2))
        self.stage2_a_conv1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=1024, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage2_a_conv1')
        self.stage2_a_conv2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage2_a_conv2')
        self.stage2_a_conv3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=1024, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage2_a_conv3')
        self.stage2_a_conv4 = keras.models.Sequential([
            keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage2_a_conv4')
        self.stage2_a_conv5 = keras.models.Sequential([
            keras.layers.Conv2D(filters=1024, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage2_a_conv5')
        self.stage2_a_conv6 = keras.models.Sequential([
            keras.layers.Conv2D(filters=1024, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage2_a_conv6')
        self.stage2_a_conv7 = keras.models.Sequential([
            keras.layers.Conv2D(filters=1024, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage2_a_conv7')
        self.stage2_b_conv = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage2_b_conv')

        self.stage3_conv1 = keras.models.Sequential([
            keras.layers.Conv2D(filters=1024, kernel_size=(3,3), strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
        ], name = 'stage3_conv1')
        self.stage3_conv2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=self.BOX * (4 + 1 + self.num_classes), kernel_size=(1,1), strides=1, padding='same'),
        ], name = 'stage3_conv2')
        self.anchor_head = AnchorLayer(grid_height=self.GRID_H, grid_width=self.GRID_W, anchors=self.ANCHORS, n_classes=self.num_classes)
        self.anchor_heads = [self.anchor_head]

    def call(self, inputs):
        output = tf.image.resize(inputs, size=(self.IMAGE_H, self.IMAGE_W), method='nearest')
        output = self.stage1_conv1(output)
        output = self.stage1_maxpl1(output)
        output = self.stage1_conv2(output)
        output = self.stage1_maxpl2(output)
        output = self.stage1_conv3(output)
        output = self.stage1_conv4(output)
        output = self.stage1_conv5(output)
        output = self.stage1_maxpl3(output)
        output = self.stage1_conv6(output)
        output = self.stage1_conv7(output)
        output = self.stage1_conv8(output)
        output = self.stage1_maxpl4(output)
        output = self.stage1_conv9(output)
        output = self.stage1_conv10(output)
        output = self.stage1_conv11(output)
        output = self.stage1_conv12(output)
        output = self.stage1_conv13(output)

        residual = output
        output_1 = self.stage2_a_maxpl(output)
        output_1 = self.stage2_a_conv1(output_1)
        output_1 = self.stage2_a_conv2(output_1)
        output_1 = self.stage2_a_conv3(output_1)
        output_1 = self.stage2_a_conv4(output_1)
        output_1 = self.stage2_a_conv5(output_1)
        output_1 = self.stage2_a_conv6(output_1)
        output_1 = self.stage2_a_conv7(output_1)
        output_2 = self.stage2_b_conv(residual)

        batch, height, width, depth = keras.backend.int_shape(output_2)
        batch = -1
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        y = keras.backend.reshape(output_2, (batch, reduced_height, self.block_size, reduced_width, self.block_size, depth))
        z = keras.backend.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
        t = keras.backend.reshape(z, (batch, reduced_height, reduced_width, depth * self.block_size **2))
        output = keras.layers.concatenate([t, output_1])

        output = self.stage3_conv1(output)
        output = self.stage3_conv2(output)
        anchor_output = self.anchor_head(output)
        return [anchor_output]

def get_image_and_objects(record):
    return record['image'], record['objects']

def bbox_to_matrix(image, objects, MAX_TRAIN_BOXES=5):
    bbox = objects['bbox']
    n_boxes_to_pad = tf.maximum(MAX_TRAIN_BOXES - tf.shape(bbox)[0], 0)
    bbox_matrix = tf.pad(bbox, [[0, n_boxes_to_pad], [0, 0]])
    bbox_matrix = bbox_matrix[:MAX_TRAIN_BOXES] # Drop any extra train boxes
    return image, bbox_matrix

def resize_image(image, bbox_matrix, INPUT_H=416, INPUT_W=416):
    resized_image = tf.image.resize(image, (INPUT_H, INPUT_W))
    return resized_image, bbox_matrix

def scale_boxes_to_yolo_grid(image, bbox_matrix, YOLO_SCALE_FACTOR=13):
    return image, bbox_matrix * YOLO_SCALE_FACTOR

def boxes_to_yx_hw_pairs(image, bbox_matrix):
    yx = bbox_matrix[:, 0:2]
    hw = bbox_matrix[:, 2:4] - bbox_matrix[:, 0:2]
    return image, tf.concat([yx + 0.5 * hw, hw], axis=-1)

def normalize_image_zero_one(image, bbox_matrix):
    return image / 255, bbox_matrix