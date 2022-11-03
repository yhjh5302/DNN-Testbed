from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import time
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (YoloV3, YoloV3Tiny)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf', 'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', '../../Data/AIC22_Track1_MTMC_Tracking/train/S03/c013/vdo.avi', 'path to video file or number for webcam)')
flags.DEFINE_string('roi_mask', '../../Data/AIC22_Track1_MTMC_Tracking/train/S03/c013/roi.jpg', 'path to image file)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    yolo.predict(np.zeros((1, 416, 416, 3)))
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)
        fps = vid.get(cv2.CAP_PROP_FPS)
        delay = int(600/fps)
        roi_mask = cv2.imread(FLAGS.roi_mask, cv2.IMREAD_UNCHANGED)
        roi_mask = cv2.resize(roi_mask, (854, 480), interpolation=cv2.INTER_CUBIC)

    out = None
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    kernel = None
    backgroundObject = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=128, detectShadows=False)
    while vid.isOpened():
        _, frame = vid.read()

        if frame is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue
        frame = cv2.resize(frame, (854, 480), interpolation=cv2.INTER_CUBIC)
        detected = False

        # calculate the foreground mask
        took = time.time()
        foreground_mask = cv2.bitwise_and(frame, frame, mask=roi_mask)
        foreground_mask = backgroundObject.apply(foreground_mask)
        _, foreground_mask = cv2.threshold(foreground_mask, 250, 255, cv2.THRESH_BINARY)
        foreground_mask = cv2.erode(foreground_mask, kernel, iterations=1)
        foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=10)
        print("mask {:.5f} ms".format((time.time() - took) * 1000))

        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxedFrame = frame.copy()
        # loop over each contour found in the frame.
        for cnt in contours:
            # We need to be sure about the area of the contours i.e. it should be higher than 256 to reduce the noise.
            if cv2.contourArea(cnt) > 256:
                detected = True
                # Accessing the x, y and height, width of the objects
                x, y, w, h = cv2.boundingRect(cnt)
                # Here we will be drawing the bounding box on the objects
                cv2.rectangle(boxedFrame, (x , y), (x + w, y + h),(0, 0, 255), 2)
                # Then with the help of putText method we will write the 'detected' on every object with a bounding box
                cv2.putText(boxedFrame, 'Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
        
        # inference
        yoloFrame = frame.copy()
        if detected:
            img_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, FLAGS.size)

            t1 = time.time()
            boxes, scores, classes, nums = yolo.predict(img_in)
            t2 = time.time()
            times.append(t2-t1)
            times = times[-20:]

            yoloFrame = draw_outputs(yoloFrame, (boxes, scores, classes, nums), class_names)
            yoloFrame = cv2.putText(yoloFrame, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            if FLAGS.output:
                out.write(yoloFrame)
            cv2.imshow("yoloFrame", yoloFrame)
            print("detect {:.5f} ms".format((time.time() - took) * 1000))

        # show_all_frames = np.hstack((frame, foregroundPart, boxedFrame))
        foregroundPart = cv2.bitwise_and(frame, frame, mask=foreground_mask)
        cv2.imshow('foregroundPart', foregroundPart)
        cv2.imshow('boxedFrame', boxedFrame)

        if cv2.waitKey(delay) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
