"""
This code is used to convert the pytorch model into an onnx format model.
"""
import sys

import torch.onnx

#from vision.ssd.config.fd_config import define_img_size

from vision_quantize.ssd.config.fd_config import define_img_size
input_img_size = 320  # define input size ,default optional(128/160/320/480/640/1280)
define_img_size(input_img_size)
from vision_quantize.ssd.mb_tiny_fd import create_mb_tiny_fd
net_type = "slim"  # inference faster,lower precision

label_path = "models/voc-model-labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

if net_type == 'slim':
    model_path = "./models/4bit_clip2_model/slim-Epoch-208-Loss-3.0186328684842145.pth"
    net = create_mb_tiny_fd(num_classes, is_test=True)
else:
    print("unsupport network type.")
    sys.exit(1)
net.load(model_path)
net.eval()
net.to("cuda")


#net.fuse()
model_name = model_path.split("/")[-1].split(".")[0]
model_path = f"models/onnx/{model_name}.onnx"


dummy_input = torch.randn(1, 3, 240, 320).to("cuda")
torch.onnx.export(net, dummy_input, model_path, verbose=True, input_names=['input'], output_names=['scores', 'boxes'], enable_onnx_checker=False)
