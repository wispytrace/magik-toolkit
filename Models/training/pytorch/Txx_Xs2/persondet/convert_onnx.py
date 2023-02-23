import torch
import torchvision
from models import *
from torch.autograd import Variable
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="weights/4-5bit/backup3.pt", help="path to model")
    parser.add_argument('--cfg', type=str, default='cfg/persondet.cfg', help='*.cfg path')
    
    cfg = parser.parse_args()
    
    trained_model = Darknet(cfg.cfg, (416, 416)).cuda()
    print('train_model: ', trained_model)
    trained_model.load_state_dict(torch.load(cfg.model)['model'], False)

    dummy_input = Variable(torch.randn(1,3,416,416)).cuda()
    dst_path = cfg.model.replace(cfg.model.split('.')[-1], "onnx")
    
#    torch.onnx.export(trained_model, dummy_input, dst_path, verbose=True, enable_onnx_checker=False, do_constant_folding=False)
    torch.onnx.export(trained_model, dummy_input, dst_path, verbose=False, enable_onnx_checker=False, do_constant_folding=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    
    print("======convert over=======save in: %s\n"%dst_path)


if __name__ == '__main__':
    main()
