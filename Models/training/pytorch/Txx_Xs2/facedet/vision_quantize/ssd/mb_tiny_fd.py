from torch.nn import Conv2d, Sequential, ModuleList, ReLU
import torch
from vision_quantize.nn.mb_tiny import Mb_Tiny    #sfyan change## RGB ##  Ori
from vision_quantize.ssd.config import fd_config as config
from vision_quantize.ssd.predictor import Predictor
from vision_quantize.ssd.ssd import SSD
import torch.nn as nn
import sys

from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python import ops

IS_QUANTIZE = True
BITA = 4#32
BITW = 4#32
SPECIAL_BIT = 4  #4bit >128  5bit<128
CLIP_MAX_VALUE = 2.0
#CLIP_MAX_VALUE = 1
#CLIP_MAX_VALUE = 2.0
print("BI",BITA)
TARGET_DEVICE = 'Txx'
# ---------------------------------------
def Depthwise(in_channels, 
              stride = 1,
              padding = 0,
              is_quantize =False, 
              bitw = 32,
              bita = 32,
              clip_vale =CLIP_MAX_VALUE ):

    return  ops.DepthwiseConv2D(in_channels, stride=stride,
                                activation_fn=None if is_quantize else nn.ReLU6(),
                                enable_batch_norm=False, 
                                enable_bias=True,
                                quantize=is_quantize,
                                padding=padding, 
                                weight_bitwidth = bitw if in_channels <= 128 else SPECIAL_BIT, #
                                input_bitwidth= bita, 
                                output_bitwidth= bita,
                                clip_max_value = clip_vale,
                                target_device=TARGET_DEVICE)
# ----------------------------------------


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,output_bitwidth=32,activation_fn=None,last_layer=True,quantize_last_feature = False,clip_dept=CLIP_MAX_VALUE,clip_con2d=CLIP_MAX_VALUE):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        #Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
        #        groups=in_channels, stride=stride, padding=padding),
        # ReLU(),
        Depthwise(in_channels=in_channels,stride=stride,padding=padding,is_quantize=IS_QUANTIZE,bitw=BITW,bita=BITA,clip_vale=clip_dept),
        
        ops.Conv2D(in_channels, out_channels, stride=1, 
                   kernel_h=1,kernel_w=1,
                   activation_fn = nn.ReLU6() if activation_fn is not None else None,
                   enable_batch_norm=False, 
                   padding=0,
                   enable_bias=True, 
                   quantize=IS_QUANTIZE,
                   weight_bitwidth = BITW if in_channels <=128 else SPECIAL_BIT, 
                   
                   input_bitwidth = BITA, 
                   output_bitwidth = 32 if last_layer else output_bitwidth,
                   clip_max_value =clip_con2d, 
                   last_layer = last_layer, 
                   quantize_last_feature = quantize_last_feature,
                   target_device=TARGET_DEVICE,)
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
             ops.Conv2D(base_net.base_channel * 16, 
                        out_channels=base_net.base_channel * 4, 
                        stride=1, 
                        kernel_h=1,kernel_w=1,
                        activation_fn = None if IS_QUANTIZE else nn.ReLU6(),
                        enable_batch_norm=False,
                        padding=0,
                        enable_bias=True,
                        quantize=IS_QUANTIZE, 
                        weight_bitwidth = BITW if base_net.base_channel * 16 <= 128 else SPECIAL_BIT,
                        input_bitwidth=BITA,  
                        output_bitwidth=BITA,
                        clip_max_value = CLIP_MAX_VALUE, 
                        target_device=TARGET_DEVICE),  # 32                    
            SeperableConv2d(in_channels=base_net.base_channel * 4, 
                            out_channels=base_net.base_channel * 16,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            activation_fn= None if IS_QUANTIZE else nn.ReLU6(),
                            last_layer=False,
                            quantize_last_feature=True,
                            output_bitwidth=BITA ,
                            clip_dept=CLIP_MAX_VALUE,
                            clip_con2d=CLIP_MAX_VALUE) # 33 34
        )
    ])
    #ori 
    regression_headers = ModuleList([ 
        SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=3 * 4, kernel_size=3, padding=1,clip_dept=CLIP_MAX_VALUE), #ori  #17
        SeperableConv2d(in_channels=base_net.base_channel * 8, out_channels=2 * 4, kernel_size=3, padding=1,clip_dept=CLIP_MAX_VALUE), #25
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=2 * 4, kernel_size=3, padding=1,clip_dept=CLIP_MAX_VALUE), #31
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=3 * 4, kernel_size=3, padding=1,clip_dept=CLIP_MAX_VALUE)
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=3 * num_classes, kernel_size=3, padding=1,clip_dept=CLIP_MAX_VALUE), #ori #16
        SeperableConv2d(in_channels=base_net.base_channel * 8, out_channels=2 * num_classes, kernel_size=3, padding=1,clip_dept=CLIP_MAX_VALUE), #24
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=2 * num_classes, kernel_size=3, padding=1,clip_dept=CLIP_MAX_VALUE),#30
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=3 * num_classes, kernel_size=3, padding=1,clip_dept=CLIP_MAX_VALUE)
        
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
