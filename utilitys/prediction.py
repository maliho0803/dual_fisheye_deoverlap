# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import tensorflow as tf
import cv2, os, time
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_iou(box1, box2):
    h1_1, h2_1, w1_1, w2_1 = box1
    box1_s = (h2_1 - h1_1) * (w2_1 - w1_1)
    h1_2, h2_2, w1_2, w2_2 = box2
    box2_s = (h2_2 - h1_2) * (w2_2 - w1_2)

    if h1_1 > h2_2 or h2_1 < h1_2:
        h1 = 0
        h2 = 0
    else:
        h1 = max(h1_1, h1_2)
        h2 = min(h2_1, h2_2)

    if w1_1 > w2_2 or w2_1 < w1_2:
        w1 = 0
        w2 = 0
    else:
        w1 = max(w1_1, w1_2)
        w2 = min(w2_1, w2_2)

    box_s = (h2 - h1) * (w2 - w1)

    return [h1, h2, w1, w2], (box_s / (box1_s + box2_s - box_s))


def get_detect_result(img_list, confidence_threshold, sess, tensor_dict, image_np_tensor):
    tmp_classes_list = []
    tmp_boxes_list = []
    tmp_score_list = []

    output_dict = sess.run(tensor_dict, feed_dict={image_np_tensor: img_list})

    for idx in range(len(img_list)):
        detection_classes = list(output_dict['detection_classes'][idx].astype(np.uint8))
        detection_boxes = list(output_dict['detection_boxes'][idx])
        detection_scores = list(output_dict['detection_scores'][idx])

        img = img_list[idx].copy()

        boxes = []
        classes = []
        scores = []

        while True:
            idx = np.argmax(detection_scores)

            y1, x1, y2, x2 = detection_boxes[idx]
            t_box = [y1, y2, x1, x2]
            tc = int(detection_classes[idx])
            ts = float(detection_scores[idx])

            del detection_classes[idx]
            del detection_boxes[idx]
            del detection_scores[idx]

            if detection_boxes == []:
                break
            if ts < confidence_threshold:
                break

            boxes.append(t_box)
            classes.append(tc)
            scores.append(ts)

            for box_idx in range(len(detection_boxes)):
                y1, x1, y2, x2 = detection_boxes[box_idx]
                iou_area, iou_score = get_iou(t_box, [y1, y2, x1, x2] )

                if detection_classes[box_idx] == tc:
                    detection_scores[box_idx] = np.exp(-(iou_score * iou_score)/0.5)*detection_scores[box_idx]
                else:
                    if iou_score > 0.8:
                        detection_scores[box_idx] = (0.5+np.exp(-(iou_score * iou_score)/0.5)) /1.5 *detection_scores[box_idx]

        tmp_classes_list.append(classes)
        tmp_boxes_list.append(boxes)
        tmp_score_list.append(scores)

    return tmp_classes_list, tmp_boxes_list, tmp_score_list
