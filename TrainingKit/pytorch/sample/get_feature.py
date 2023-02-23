import os
import time
import argparse
from datetime import datetime

import torch
import torch.optim as optim

from torchvision import datasets, transforms

from network import *
import numpy
import cv2
import numpy as np

model = Network_T40(is_quantize = True, BITA = 8, BITW = 8).cuda()
model.load_state_dict(torch.load('ckpt/net-w8a8/checkpoint.w8a8-4'))
model.eval()

img = cv2.imread('test1.bmp')
nimg = img
nimg = nimg/255.
nimg = np.transpose(nimg, (2,0,1))
nimg = torch.tensor(nimg).float().reshape(1,3,32,32)
outputs = model(nimg.cuda())
print (outputs.data)
_, predicted = torch.max(outputs.data, 1)
print(predicted)
cv2.imshow('iii', img)
cv2.waitKey(0)
