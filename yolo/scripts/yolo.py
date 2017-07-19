#!/usr/bin/env python
# YOLO ros object, uses the subscriber to evaluate images
from __future__ import print_function

import os
import sys
import rospy
import cv2
import json
import argparse
import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.models import model_from_json
from keras import backend as K
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import context

from yad2k.utils.draw_boxes import draw_boxes
from retrain_yolo import create_model
from yad2k.models.keras_yolo import yolo_head_np, yolo_eval
from subscriber import videosub


class yolo(object):
    '''
    YOLOv2 class integrated with YAD2K and ROS
    '''
    def __init__(self, paths, score_threshold=0.3, iou_threshold=0.6, max_detections=15):  
        # Load classes and anchors.
        json_path, anchors_path, classes_path, weights_path = paths

        with open(classes_path) as f:
                self.class_names = f.readlines()
        self.class_names = [c.strip() for c in self.class_names]

        with open(anchors_path) as f:
            self.anchors = f.readline()
            self.anchors = [float(x) for x in self.anchors.split(',')]
            self.anchors = np.array(self.anchors).reshape(-1, 2)

        self.image_input = Input(shape=(None, None, 3))

        # yolo_json_file = open(json_path, 'r') # no longer loading model directly, using create_model
        # yolo_json = yolo_json_file.read()
        # yolo_json_file.close()

        self.yolo_model, garbage = create_model(self.anchors, self.class_names, False, 0, json_path)
        
        self.yolo_model.load_weights(weights_path)

        self.yolo_model.summary()

        self.max_boxes=max_detections,
        self.score_threshold=score_threshold,
        self.iou_threshold=iou_threshold

        print('yolo object created')

    def pred(self, image_s, display_shape=(640, 480)):
        # Make predictions for images in (num_images, height, width, channel) format
        assert len(image_s.shape) == 4 # image must have 4 dims ready to be sent into the graph
        # TODO allow multiple images at once.
        assert image_s.shape[0] == 1
        features = self.yolo_model.predict(image_s)

        proc = yolo_head_np(features, self.anchors, len(self.class_names))

        out_boxes, out_scores, out_classes = yolo_eval(proc, display_shape, max_boxes=self.max_boxes,
            score_threshold=self.score_threshold, iou_threshold=self.iou_threshold)

        return out_boxes, out_scores, out_classes

def draw(out_boxes, out_scores, out_classes, image, name, class_names, display=True):
    if len(out_boxes) != 0:
        np.swapaxes(image, 0, 1) # opencv convention is to access image as (row, column)
        image = draw_boxes(image, out_boxes, out_classes, class_names, scores=out_scores, rectify=False)
        np.swapaxes(image, 0, 1)
    if display:
        cv2.imshow(name, image)
        cv2.waitKey(3)
    return image

def to_json(img_shape, boxes, scores, classes):
    # image_shape : the shape of the image used by yolo
    # boxes: An `array` of shape (num_boxes, 4) containing box corners as
    #     (y_min, x_min, y_max, x_max).
    # `scores`: A `list` of scores for each box.
    # classes: A `list` of indicies into `class_names`.
    # return: json string
    json_list = []
    for i in range(len(boxes)):
        center_x = (boxes[i][1] + boxes[i][3])/2/img_shape[1]
        center_y = (boxes[i][0] + boxes[i][2])/2/img_shape[0]
        confidence = scores[i]
        _class = classes[i]
        json_list.append([int(_class), float(center_x), float(center_y), float(confidence)])

    json_list = json.dumps(json_list)
    return json_list

def to_multiarray(img_shape, boxes, scores, classes):
    # image_shape : the shape of the image used by yolo
    # boxes: An `array` of shape (num_boxes, 4) containing box corners as
    #     (y_min, x_min, y_max, x_max).
    # `scores`: A `list` of scores for each box.
    # classes: A `list` of indicies into `class_names`.
    # return: multiarray
    data = []
    for i in range(len(boxes)):
        y_min = boxes[i][0]/img_shape[0]
        x_min = boxes[i][1]/img_shape[1]
        y_max = boxes[i][2]/img_shape[0]
        x_max = boxes[i][3]/img_shape[1]
        confidence = scores[i]
        _class = classes[i]
        data.append(float(_class))
        data.append(float(y_min))
        data.append(float(x_min))
        data.append(float(y_max))
        data.append(float(x_max))
        data.append(float(confidence))
    message = Float32MultiArray()
    message.layout.dim.append(MultiArrayDimension("boxes", len(boxes), 1))
    message.layout.dim.append(MultiArrayDimension("box_data", 6, 1))
    message.data = data

    return message


def _main(args):
    anchors_path= os.path.expanduser(args.anchors_path)
    classes_path= os.path.expanduser(args.classes_path)
    weights_path = os.path.expanduser(args.weights_path)
    json_path = os.path.expanduser(args.json_path)

    rospy.init_node("yoloNode")

    yo = yolo((json_path, anchors_path, classes_path, weights_path), .4, .4, 100)

    if args.mode.lower()=='json':
        box_type = String
    else:
        box_type = Float32MultiArray

    bridge = CvBridge()

    vid1 = videosub(args.first_topic, (96, 96))
    boxes_pub_1 = rospy.Publisher("/yolo/first/boxes", box_type, queue_size=10)
    proc_image_pub_1 = rospy.Publisher("/yolo/first/proc_image", Image, queue_size=2)

    if args.second_topic is not None:
        vid2 = videosub(args.second_topic)
        boxes_pub_2 = rospy.Publisher("/yolo/second/boxes", box_type, queue_size=10)
        proc_image_pub_2 = rospy.Publisher("/yolo/second/proc_image", Image, queue_size=2)

    
    rate = rospy.Rate(15)
    image = None
    while not rospy.is_shutdown():
        # Grab new images from the subscribers
        if vid1.newImgAvailable:
            image, image_data = vid1.getProcessedImage()
            boxes, scores, classes = yo.pred(image_data, image.shape[0:2])
            print(image.shape[:2])
            if args.display: #display the image
                processed_image = draw(boxes, scores, classes, image, vid1.topic, yo.class_names) # Display

            if args.mode.lower()=='json': # message is JSON
                message = to_json(image.shape[0:2], boxes, scores, classes)
            else: # message is multiarray
                message = to_multiarray(image.shape[0:2], boxes, scores, classes)
            boxes_pub_1.publish(message)

            if args.publish: # Publish the processed image
                if not args.display:
                    processed_image = draw(boxes, scores, classes, image, vid1.topic, yo.class_names, False)
                try:
                    proc_image_pub_1.publish(bridge.cv2_to_imgmsg(processed_image, "bgr8"))
                except CvBridgeError as e:
                    print(e)


        if args.second_topic is not None and vid2.newImgAvailable: #again, for second topic 
            image, image_data = vid2.getProcessedImage()
            boxes, scores, classes = yo.pred(image_data, image.shape[0:2])

            if args.display:
                processed_image = draw(boxes, scores, classes, image, vid1.topic, yo.class_names) # Display

            if args.mode.lower()=='json':
                message = to_json(image.shape[0:2], boxes, scores, classes)
            else: # multiarray
                message = to_multiarray(image.shape[0:2], boxes, scores, classes)
            boxes_pub_2.publish(message)

            if args.publish: # Publish the processed image
                if not args.display:
                    processed_image = draw(boxes, scores, classes, image, vid1.topic, yo.class_names, False)
                try:
                    proc_image_pub_2.publish(bridge.cv2_to_imgmsg(processed_image, "bgr8"))
                except CvBridgeError as e:
                    print(e)

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
        help="(optional)First topic to subscribe to. defaults to '/sensors/camera0/jpegbuffer'",
        default='/sensors/camera0/jpegbuffer')

    argparser.add_argument(
        '-s',
        '--second_topic',
        help='(optional) Second topic to subscribe to. Leave blank if None.',
        default=None)

    argparser.add_argument(
        '-d',
        '--display',
        action='store_true',
        help='use this flag to display an opencv image during computation')

    argparser.add_argument(
        '-p',
        '--publish',
        action='store_true',
        help='use this flag to publish image with class boxes.')
    
    argparser.add_argument(
        '-m',
        '--mode',
        help='''mode: "json" for json publisher, else: multiarray publisher.
        Defaults to json. Note: json publisher only publishes [classNum, centerX(%of image), cneterY(%of image), Confidence]
        per box. Multiarray Publishes [classNum, y_min%, x_min%, y_max%, x_max%, confidence]''',
        default='json')

    argparser.add_argument(
        '-c',
        '--classes_path',
        help='(optional)path to classes file, defaults to YAD2K/model_data/aerial_classe.txt',
        default=os.path.join(filepath, 'model_data', 'aerial_classes.txt'))

    argparser.add_argument(
        '-a',
        '--anchors_path',
        help='(optional)path to anchors file, defaults to YAD2K/model_data/yolo_anchors.txt',
        default=os.path.join(filepath, 'model_data', 'yolo_anchors.txt'))
    
    argparser.add_argument(
        '-w',
        '--weights_path',
        help='(optional) path to model weights file, defaults to YAD2K/trained_stage_2_best.h5',
        default=os.path.join(filepath, 'trained_stage_2_best.h5'))

    argparser.add_argument(
        '-j',
        '--json_path',
        help='(optional) path to model json file, defaults to YAD2K/model_data/yolo.json',
        default=os.path.join(filepath, 'model_data', 'yolo.json'))

    args = argparser.parse_args()

    _main(args)

