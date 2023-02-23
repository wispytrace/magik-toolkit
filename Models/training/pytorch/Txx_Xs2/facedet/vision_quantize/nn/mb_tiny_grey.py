import torch.nn as nn
import torch.nn.functional as F


IS_QUANTIZE = True
BITA = 4#32
BITW = 4#32
SPECIAL_BIT = 4  #4bit >128  5bit<128
CLIP_MAX_VALUE = 2.0
#CLIP_MAX_VALUE = 1.5
#CLIP_MAX_VALUE = 2.0
TARGET_DEVICE = 'Txx'
import sys
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python import ops

class Mb_Tiny(nn.Module):

    def __init__(self, num_classes=2):
        super(Mb_Tiny, self).__init__()
        self.base_channel = 8 * 2

        def conv_dw(inp, oup, stride,clip_dept=CLIP_MAX_VALUE,clip_conv=CLIP_MAX_VALUE):
            return nn.Sequential(
                Depthwise(inp,stride=stride,padding=1,is_quantize=IS_QUANTIZE,bitw=BITW,bita=BITA,clip_vale=clip_dept),
                ConvBlock(inp,oup,kernel_size=1,stride=1,padding=0,is_quantize=IS_QUANTIZE,bitw=BITW,bita=BITA,clip_vale=clip_conv)
            )
        def ConvBlock(in_channels, 
                out_channels, 
                stride = 1,
                kernel_size = 3,
                is_quantize =False, 
                bitw = 32,
                bita = 32, 
                padding = 0,
                clip_vale= CLIP_MAX_VALUE):

            return  ops.Conv2D(in_channels, out_channels, stride=stride, 
                               kernel_h=kernel_size,kernel_w=kernel_size,
                               activation_fn=None if is_quantize else nn.ReLU6(),
                               enable_batch_norm=True, 
                               enable_bias=False,
                               quantize=is_quantize, 
                               padding=padding,
                               weight_bitwidth = bitw if in_channels <= 128 else SPECIAL_BIT,
                               #input_bitwidth=bita if in_channels <= 128 else SPECIAL_BIT,
                               input_bitwidth=bita, 
                               #output_bitwidth=bita if out_channels <= 128 else SPECIAL_BIT,
                               output_bitwidth=bita,
                               clip_max_value = clip_vale,
                               target_device=TARGET_DEVICE)
        # ---------------------------------------
        def Depthwise(in_channels, 
                        stride = 1,
                        padding = 0,
                        is_quantize =False, 
                        bitw = 32,
                        bita = 32,
                        clip_vale = CLIP_MAX_VALUE):

            return  ops.DepthwiseConv2D(in_channels, stride=stride,
                                        activation_fn=None if is_quantize else nn.ReLU6(),
                                        enable_batch_norm=True, 
                                        enable_bias=False, quantize=is_quantize, padding=padding, 
                                        weight_bitwidth = bitw if in_channels <= 128 else SPECIAL_BIT,#8
                                        input_bitwidth=bita, 
                                        output_bitwidth=bita,
                                        clip_max_value = clip_vale,
                                        target_device=TARGET_DEVICE)
        # ----------------------------------------
        self.model = nn.Sequential(
            
            ops.Preprocess(128., 128.), 
            ops.Conv2D(3, self.base_channel, stride=2, #1
                                activation_fn=None if IS_QUANTIZE else nn.ReLU6(),
                                enable_batch_norm=True, 
                                enable_bias=False, quantize=IS_QUANTIZE,
                                first_layer=True,
                                padding=1, weight_bitwidth = BITW, input_bitwidth=8, 
                                output_bitwidth=BITA, clip_max_value = 4 if 0 else CLIP_MAX_VALUE, target_device=TARGET_DEVICE),

            conv_dw(self.base_channel, self.base_channel * 2, 1, CLIP_MAX_VALUE,CLIP_MAX_VALUE), # 2 3
            conv_dw(self.base_channel * 2, self.base_channel * 2, 2, CLIP_MAX_VALUE,CLIP_MAX_VALUE), # 4 5 80*60  120*90
            conv_dw(self.base_channel * 2, self.base_channel * 2, 1, CLIP_MAX_VALUE,CLIP_MAX_VALUE), # 6 7 
            conv_dw(self.base_channel * 2, self.base_channel * 4, 2, CLIP_MAX_VALUE,CLIP_MAX_VALUE), # 8 9 40*30  60*45
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1, CLIP_MAX_VALUE,CLIP_MAX_VALUE), # 10 11
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1, CLIP_MAX_VALUE,CLIP_MAX_VALUE), # 12 13
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1, CLIP_MAX_VALUE,CLIP_MAX_VALUE), # 14 15
            conv_dw(self.base_channel * 4, self.base_channel * 8, 2, CLIP_MAX_VALUE,CLIP_MAX_VALUE), # 18 19 20*15  30*23
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1, CLIP_MAX_VALUE,CLIP_MAX_VALUE), #20 21
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1, CLIP_MAX_VALUE,CLIP_MAX_VALUE), #22 23
            conv_dw(self.base_channel * 8, self.base_channel * 16, 2, CLIP_MAX_VALUE,CLIP_MAX_VALUE),  # 26 27 10*8  15*12
            conv_dw(self.base_channel * 16, self.base_channel * 16, 1,CLIP_MAX_VALUE,CLIP_MAX_VALUE) , # 28 29
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        print('hello hello')
        x = self.model(x)
        x = F.avg_pool2d(x, 7) ##avg-pool  7*7cov
        x = x.view(-1, 1024)  
        x = self.fc(x)
        return x

