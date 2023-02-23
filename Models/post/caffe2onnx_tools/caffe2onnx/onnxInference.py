#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    > File Name: onnxInference.py
    > Author: dzhang
    > Mail: dong.zhang@ingenic.com 
    > Created Time: Tue 27 Oct 2020 03:32:35 PM CST
"""


import os
import numpy as np
import onnxruntime
import onnx
import sys
import glob
import cv2
import numpy as np
from PIL import Image
import torchvision.models as models
import torch
from torchvision import datasets, transforms as T

class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
 
    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed
 
    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        self.output_name = []
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        res = self.onnx_session.run(self.output_name, input_feed=input_feed)[0]
        return res

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


onnx_model_path = "/home/czhang/onnx/clip_onnx_model.onnx"
model = ONNXModel(onnx_model_path)

test_dir = '/home/czhang/data/test_cifar10/'
mean=[0, 0, 0]
std=[1, 1, 1]

num = 0
right = 0	

for cls in os.listdir(test_dir):
    img_dir = test_dir + '/' + cls
    for img in os.listdir(img_dir):
        imagepath = test_dir + '/' + cls + '/' + img
        image = cv2.imread(imagepath)
        image = image[...,::-1]
        image = image.astype(np.float32) / 255
        image = image.transpose(2,0,1)
        image = image[np.newaxis,...]
        res = model.forward(image)
        res = softmax(res)
        if np.argmax(res)==int(cls):
            right +=1
        num +=1
        if num%100 == 0:
            print('current num {} : {} '.format(num, right*1.0/num))

print('current num {} : {} '.format(num, right*1.0/num))

    # imageNet(model)
