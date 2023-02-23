from utils import get_network
import torch
import argparse
from torch.autograd import Variable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-num_classes', type=int, default=100, help='num_classes')
    args = parser.parse_args()

    net = get_network(args).cuda()
    
    state_dict = torch.load(args.weights)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.eval()
    
    dummy_input = Variable(torch.randn(2, 3, 32, 32)).cuda()
    dst_path = args.weights.replace(args.weights.split('.')[-1], "onnx")
    
    torch.onnx.export(net, dummy_input, dst_path, verbose=False, enable_onnx_checker=False, do_constant_folding=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    
    print("======convert over=======save in: %s\n"%dst_path)


if __name__ == '__main__':
    main()

##python convert_onnx.py --Abits 8 --Wbits 8 --model ./ckpt/net-w8a8/checkpoint.w8a8-0 
