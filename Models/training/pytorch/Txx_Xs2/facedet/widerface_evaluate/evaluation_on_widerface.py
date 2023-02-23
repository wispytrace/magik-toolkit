#!/usr/bin/ python3
# -*- coding: utf-8 -*-
# @Time    : 2019-10-17
# @Author  : vealocia
# @FileName: evaluation_on_widerface.py

import math
import os
import sys

import cv2
sys.path.append('../')
from vision_quantize.ssd.config.fd_config import define_img_size

input_img_size = 320  # define input size ,default optional(128/160/320/480/640/1280)
define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision_quantize.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
import numpy as np
label_path = "../models/voc-model-labels.txt"

net_type = "slim"          # inference faster,lower precision

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = "cuda:0"
# test_device = "cpu"
candidate_size = 800
threshold = 0.1

val_image_root = "/home/ssd/sfyan/data/WIDER_val/images"  # path to widerface valuation image root
val_result_txt_save_root = "./widerface_evaluation/"  # result directory

if net_type == 'slim':
    #model_path = "../models/pretrained/version-slim-320.pth"
    # model_path = "../models/pretrained/version-slim-640.pth" 
    model_path = "../models/4bit_clip2_model/models/slim-Epoch-202-Loss-3.006300452903465.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
net.load(model_path)

counter = 0
for parent, dir_names, file_names in os.walk(val_image_root):
    for file_name in file_names:
        if not file_name.lower().endswith('jpg'):
            continue
        im = cv2.imread(os.path.join(parent, file_name), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  
        #im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) #sfyan change
        #im = np.reshape(im, (im.shape[0], im.shape[1], 1))

        boxes, labels, probs = predictor.predict(im, candidate_size / 2, threshold)

        event_name = parent.split('/')[-1]
        if not os.path.exists(os.path.join(val_result_txt_save_root, event_name)):
            os.makedirs(os.path.join(val_result_txt_save_root, event_name))
        fout = open(os.path.join(val_result_txt_save_root, event_name, file_name.split('.')[0] + '.txt'), 'w')
        fout.write(file_name.split('.')[0] + '\n')
        fout.write(str(boxes.size(0)) + '\n')
        for i in range(boxes.size(0)):
            bbox = boxes[i, :]
            fout.write('%d %d %d %d %.03f' % (math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2] - bbox[0]), math.ceil(bbox[3] - bbox[1]), probs[i] if probs[i] <= 1 else 1) + '\n')
        fout.close()
        counter += 1
        print('[%d] %s is processed.' % (counter, file_name))

# note: with score_threshold = 0.11 and hard_nms, MAP of 320-input model on widerface val set is: 0.785/0.695/0.431
