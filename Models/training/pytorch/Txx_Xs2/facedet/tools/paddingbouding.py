import cv2
import numpy as np
import os
import sys

path = "/home/xjyu/Ultra-Light-Fast-Generic-Face-Detector-1MB-gray/imgs/5042test"
listdir = os.listdir(path)
for file_path in listdir:
    img_path = os.path.join(path, file_path)
    print(img_path)
    orig_image = cv2.imread(img_path)
    
    Black = [0,0,0]
    constant = cv2.copyMakeBorder(orig_image,120,120,120,120,cv2.BORDER_CONSTANT,value=Black)
    print(orig_image.shape)
    print(constant.shape)
    cv2.imwrite(file_path[:-4]+'padding.png',constant)
    

