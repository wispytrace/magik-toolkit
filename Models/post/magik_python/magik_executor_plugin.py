#!/usr/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : magik_executor_plugin.py
## Authors    : lqwang
## Create Time: 2022-03-16:09:39:54
## Description:
## 
##

import os
import logging
import numpy as np
import cv2

from magik_executor.magik_executor import *

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

if __name__=="__main__":
    path = "../yolov5s/venus_sample_yolov5s/save-magik/model_quant.mgk"
    mr = MagikRoutine(path, dump_file=True)
    img = cv2.imread("./fall_1054_sys.jpg")
    img = letterbox(img, 640)[0]
    img = img[:, :, ::-1]
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    print(img.shape)
    mr.set_input(img, mr.inames()[0])
    print(mr.inames()[0])
    mr.run()
    for i in mr.onames():
        print(mr.get_output(i).shape)
