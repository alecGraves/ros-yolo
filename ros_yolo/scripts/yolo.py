#!/usr/bin/env python
# YOLO ros object, uses the subscriber to evaluate images
from __future__ import print_function

import os
import sys
import rospy
import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras import backend as K
import argparse

import context

from yad2k.utils.draw_boxes import draw_boxes
from yad2k.models.keras_yolo import yolo_body, yolo_eval, yolo_head
from subscriber import videosub


class yolo(object):
    '''
    YOLOv2 class integrated with YAD2K and ROS
    '''
    def __init__(self, anchors_path, classes_path, model_path, score_threshold=0.3, iou_threshold=0.6, max_detections=15):  
        # Load classes and anchors.
        with open(classes_path) as f:
                self.class_names = f.readlines()
        self.class_names = [c.strip() for c in self.class_names]

        with open(anchors_path) as f:
            self.anchors = f.readline()
            self.anchors = [float(x) for x in self.anchors.split(',')]
            self.anchors = np.array(self.anchors).reshape(-1, 2)

        # Load model and set up computation graph.
        self.sess = K.get_session()

        self.yolo_body = load_model(model_path)

        self.yolo_body.summary()

        self.yolo_outputs = yolo_head(self.yolo_body.output, self.anchors, len(self.class_names))

        self.input_image_shape = K.placeholder(shape=(2, ))

        self.boxes, self.scores, self.classes = yolo_eval(
            self.yolo_outputs,
            self.input_image_shape,
            max_boxes=max_detections,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold)

        print('yolo object created')

    def pred(self, image_s):
        # Make predictions for one image, in (1, height, width, channel) format
        assert len(image_s.shape) == 4 # image must have 4 dims ready to be sent into the graph
        out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_body.input: image_s,
                    self.input_image_shape: [image_s.shape[1], image_s.shape[2]],
                    K.learning_phase(): 0
                })
        return out_boxes, out_scores, out_classes

    def display(self, out_boxes, out_scores, out_classes, image, name):
        if len(out_boxes) == 0:
            cv2.imshow(name, np.floor(image[0] * 255 + 0.5).astype('uint8'))
        else:
            image = draw_boxes(image[0], out_boxes, out_classes, self.class_names, scores=out_scores)
            cv2.imshow(name, image)
        cv2.waitKey(10)

def _main(args):
    anchors_path= os.path.expanduser(args.anchors_path)
    classes_path= os.path.expanduser(args.classes_path)
    model_path = os.path.expanduser(args.model_path)

    rospy.init_node("yoloNode")

    yo = yolo(anchors_path, classes_path, model_path, .4, .4, 100)

    vid1 = videosub(args.first_topic)
    if not (args.second_topic is None):
        vid2 = videosub(args.second_topic)

    rate = rospy.Rate(15)
    image = None
    while not rospy.is_shutdown():
        # Grab new images from the subscribers
        if vid1.newImgAvailable:
            image = vid1.getProcessedImage()
            boxes, scores, classes = yo.pred(image)
            yo.display(boxes, scores, classes, image, vid1.topic) # Display
        if (not (args.second_topic is None)) and vid2.newImgAvailable:
            image = vid2.getProcessedImage()
            boxes, scores, classes = yo.pred(image)
            yo.display(boxes, scores, classes, image, vid2.topic)

        rate.sleep()

if __name__ == '__main__':
    # Configure default paths
    filepath = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.abspath(os.path.join(filepath, '..', '..', 'YAD2K'))
    
    # Args
    argparser = argparse.ArgumentParser(
        description="launch a yolo model, subscribed to topic '/sensors/camera0/jpegbuffer'")

    argparser.add_argument(
        '-f',
        '--first_topic',
        help="First topic to subscribe to. defaults to '/sensors/camera0/jpegbuffer'",
        default='/sensors/camera0/jpegbuffer')

    argparser.add_argument(
        '-s',
        '--second_topic',
        help='Second topic to subscribe to. Leave blank if None.',
        default=None)

    argparser.add_argument(
        '-c',
        '--classes_path',
        help='path to classes file, defaults to YAD2K/model_data/aerial_classe.txt',
        default=os.path.join(filepath, 'model_data', 'aerial_classes.txt'))

    argparser.add_argument(
        '-a',
        '--anchors_path',
        help='path to anchors file, defaults to YAD2K/model_data/yolo_anchors.txt',
        default=os.path.join(filepath, 'model_data', 'yolo_anchors.txt'))
    
    argparser.add_argument(
        '-m',
        '--model_path',
        help='path to model file, defaults to YAD2K/trained_stage_2_best.h5',
        default=os.path.join(filepath, 'model_data', 'retrained.h5'))

    args = argparser.parse_args()

    _main(args)

