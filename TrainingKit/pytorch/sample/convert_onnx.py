import torch
import torchvision
from network import *
from torch.autograd import Variable
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ckpt/net-w8a8/checkpoint.w8a8-4", help="path to model")
    parser.add_argument("--Abits", type=int, default="8", help="path to model")
    parser.add_argument("--Wbits", type=int, default="8", help="path to model")
    cfg = parser.parse_args()

    trained_model = Network_T40(is_quantize = True, BITA = cfg.Abits, BITW = cfg.Wbits).cuda()
    # print('train_model: ', trained_model)
    trained_model.load_state_dict(torch.load(cfg.model), True)

    trained_model.eval()
    dummy_input = Variable(torch.randn(1, 3, 32, 32)).cuda()
    dst_path = cfg.model.replace(cfg.model.split('.')[-1], "onnx")

    torch.onnx.export(trained_model, dummy_input, dst_path, verbose=False, enable_onnx_checker=False, do_constant_folding=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    print("======convert over=======save in: %s\n"%dst_path)


if __name__ == '__main__':
    main()

##python convert_onnx.py --Abits 8 --Wbits 8 --model ./ckpt/net-w8a8/checkpoint.w8a8-0
