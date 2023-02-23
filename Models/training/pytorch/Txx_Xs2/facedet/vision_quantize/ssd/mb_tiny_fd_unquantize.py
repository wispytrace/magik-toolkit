from torch.nn import Conv2d, Sequential, ModuleList, ReLU
import torch
#from vision.nn.mb_tiny import Mb_Tiny    #sfyan change## RGB ##  Ori
from vision_quantize.nn.mb_tiny_grey import Mb_Tiny  #sfyan change## GRAY ##
from vision_quantize.ssd.config import fd_config as config
from vision_quantize.ssd.predictor import Predictor
from vision_quantize.ssd.ssd import SSD
import sys
sys.path.append("./QuantizationTrainingPlugin/python")
import ops
IS_QUANTIZE = True
BITA = 4#32
BITW = 4#32
CLIP_MAX_VALUE = 2.0
TARGET_DEVICE = 'Txx'
# ---------------------------------------
def Depthwise(in_channels, 
                stride = 1,
                padding = 0,
                is_quantize =False, 
                BITW = 32,
                BITA = 32, ):

    return  ops.DepthwiseConv2D(in_channels, stride=stride,
                                activation_fn=None,
                                enable_batch_norm=False, 
                                enable_bias=True, quantize=is_quantize, padding=padding, 
                                weight_bitwidth = BITW, input_bitwidth=BITA, output_bitwidth=BITA, 
                                clip_max_value = CLIP_MAX_VALUE, target_device=TARGET_DEVICE)
# ----------------------------------------

def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )

def SeperableConv2d_Quantize(in_channels, out_channels, kernel_size=1, stride=1, padding=0,output_bitwidth=32,activation_fn=None,last_layer=True,quantize_last_feature = False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        #Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
        #        groups=in_channels, stride=stride, padding=padding),
        # ReLU(),
        Depthwise(in_channels=in_channels,stride=stride,padding=padding,is_quantize=IS_QUANTIZE,BITW=BITW,BITA=BITA),
        #Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        ops.Conv2D(in_channels, out_channels, stride=1, 
                    kernel_h=1,kernel_w=1,
                    activation_fn = activation_fn,
                    enable_batch_norm=False, 
                    padding=0,
                    enable_bias=True, 
                    quantize=IS_QUANTIZE, 
                    weight_bitwidth = BITW, 
                    input_bitwidth=BITA, 
                    output_bitwidth=output_bitwidth, 
                    clip_max_value =CLIP_MAX_VALUE, 
                    last_layer = last_layer, 
                    quantize_last_feature = quantize_last_feature,
                    target_device=TARGET_DEVICE)
    )


def create_mb_tiny_fd(num_classes, is_test=False, device="cuda"):
    base_net = Mb_Tiny(2)
    base_net_model = base_net.model  # disable dropout layer

    source_layer_indexes = [
        8,
        11,
        13
    ]
    extras = ModuleList([
        Sequential(
            # Conv2d(in_channels=base_net.base_channel * 16, out_channels=base_net.base_channel * 4, kernel_size=1),
            # ReLU(),
             ops.Conv2D(base_net.base_channel * 16, 
                    out_channels=base_net.base_channel * 4, 
                    stride=1, 
                    kernel_h=1,kernel_w=1,
                    activation_fn = None,
                    enable_batch_norm=False, padding=0,
                    enable_bias=True, quantize=IS_QUANTIZE, 
                    weight_bitwidth = BITW, input_bitwidth=BITA, 
                    output_bitwidth=BITA, clip_max_value = CLIP_MAX_VALUE, 
                    target_device=TARGET_DEVICE),
                    
            # SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=base_net.base_channel * 16, kernel_size=3, stride=2, padding=1),
            # ReLU()
            SeperableConv2d_Quantize(in_channels=base_net.base_channel * 4, 
                            out_channels=base_net.base_channel * 16,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            activation_fn= None,
                            last_layer=False,
                            quantize_last_feature=True,
                            output_bitwidth=BITA)
        )
    ])
    #ori 
    regression_headers = ModuleList([ 
        SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=3 * 4, kernel_size=3, padding=1), #ori 
        SeperableConv2d(in_channels=base_net.base_channel * 8, out_channels=2 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=2 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=base_net.base_channel * 16, out_channels=3 * 4, kernel_size=3, padding=1)
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=3 * num_classes, kernel_size=3, padding=1), #ori
        SeperableConv2d(in_channels=base_net.base_channel * 8, out_channels=2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=2 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=base_net.base_channel * 16, out_channels=3 * num_classes, kernel_size=3, padding=1)
    ])

    return SSD(num_classes, base_net_model, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config, device=device)


def create_mb_tiny_fd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean_test,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
