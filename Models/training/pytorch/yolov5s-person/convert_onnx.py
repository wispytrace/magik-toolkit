import torch
import torchvision
from models.yolo import Model
from torch.autograd import Variable
import argparse
import os
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/train/yolov5s-person-4bit.pt", help="path to model")
    #parser.add_argument("--cfg", type=str, default="./yolov5.yaml", help="path to model")
    args = parser.parse_args()

    chkpt = torch.load(args.weights)
    model = Model(chkpt['model'].yaml, ch=3, nc=1).cuda()
    state_dict = chkpt['ema' if chkpt.get('ema') else 'model'].float().state_dict()  # to FP32
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    # model = chkpt['ema' if chkpt.get('ema') else 'model'].float().cuda()

    dummy_input = Variable(torch.randn(1, 3, 416, 416)).cuda()
    dst_path = args.weights.replace(args.weights.split('.')[-1], "onnx")
    torch.onnx.export(model, dummy_input, dst_path, verbose=False, enable_onnx_checker=False, do_constant_folding=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    
    print("======convert over=======save in: %s\n"%dst_path)


if __name__ == '__main__':
    main()
