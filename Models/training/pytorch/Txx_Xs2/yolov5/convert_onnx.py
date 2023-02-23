import torch
import torchvision
from models.yolo import Model
from torch.autograd import Variable
import argparse
import os
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="ckpt/net-w8a8/checkpoint.w8a8-0", help="path to model")
    parser.add_argument("--cfg", type=str, default="./yolov5.yaml", help="path to model")
    args = parser.parse_args()
    
    model = Model(args.cfg, ch=3, nc=4).cuda()
    # print('train_model: ', trained_model)
    
    #dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23454', rank=0,
    #                        world_size=1)  # distributed backend
    chkpt = torch.load(args.weights)
    model.load_state_dict(chkpt['ema' if chkpt.get('ema') else 'model'].float().state_dict(), strict=True)

    dummy_input = Variable(torch.randn(2, 3, 128, 128)).cuda()
    dst_path = args.weights.replace(args.weights.split('.')[-1], "onnx")
    #exit()
    torch.onnx.export(model, dummy_input, dst_path, verbose=True, enable_onnx_checker=False, do_constant_folding=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    
    print("======convert over=======save in: %s\n"%dst_path)


if __name__ == '__main__':
    main()

##python convert_onnx.py --weights --cfg
