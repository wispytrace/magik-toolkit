import cv2
from conf import settings
import numpy as np 
import torch
from utils import get_test_dataloader,get_network
import argparse
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-num_classes', type=int, default=100, help='num_classes')
    args = parser.parse_args()
    
    image_one = cv2.imread("./test_71.jpg")
    image_one = cv2.cvtColor(image_one, cv2.COLOR_BGR2RGB)
    #image_one = cv2.resize(image_one,(32, 32))

    image_one = image_one/255.
    image_one = np.transpose(image_one, (2,0,1))
    image_one = torch.tensor(image_one)
    image_one = image_one.unsqueeze(0).type(torch.FloatTensor).cuda()

    net = get_network(args).cuda()

    state_dict = torch.load(args.weights)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.eval()

    output = net(image_one)
    _, pred = output.max(1)
    # print("output ---> ",output)
    print("pred ---> ",pred)
