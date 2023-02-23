import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
import warnings

from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python import ops

IS_QUANTIZE = 1
BITW = 8
if BITW==8:
    BITA = 8
    WEIGHT_FACTOR = 3.0
    CLIP_MAX_VALUE = 6.0
elif BITW==4:
    BITA = 4
    WEIGHT_FACTOR = 3.0
    CLIP_MAX_VALUE = 4.0
elif BITW==2:
    BITA = 2
    WEIGHT_FACTOR = 2.0
    CLIP_MAX_VALUE = 2.0
else:
    BITA = 32
    WEIGHT_FACTOR = 3.
    CLIP_MAX_VALUE = 6.
TARGET_DEVICE = "Txx"

def preprocess():
    return ops.Preprocess(0., 255., target_device = TARGET_DEVICE)

def qConv(in_channels, out_channels, kernel_size=None, stride=1, pad=0, groups=1, dilation=1, bias=False, bn=True, act=False, first=False, last=False):
    assert(groups==1)
    assert(dilation==1)
    act_fn = nn.ReLU(inplace=True) if not IS_QUANTIZE and act else None
    if isinstance(kernel_size, tuple):
        h = int(kernel_size[0])
        w = int(kernel_size[1])
    else:
        h = w = int(kernel_size)
    return ops.Conv2D(in_channels,
                      out_channels,
                      kernel_h = h,
                      kernel_w = w,
                      stride = stride,
                      activation_fn = act_fn,
                      enable_batch_norm = bn,
                      enable_bias = bias,
                      quantize = IS_QUANTIZE,
                      first_layer = first,
                      padding = pad,
                      weight_bitwidth = BITW,
                      input_bitwidth = BITA,
                      output_bitwidth = BITA,
                      clip_max_value = CLIP_MAX_VALUE,
                      weight_factor = WEIGHT_FACTOR,
                      target_device = TARGET_DEVICE)

def qConv_last(in_channels, out_channels, kernel_size=1, stride=1, pad=0, groups=1, dilation=1, bias=False, bn=True, act=False):
    if act:
        act_fn = nn.ReLU(inplace=True)
    return ops.Conv2D(in_channels,
                      out_channels,
                      kernel_h = kernel_size,
                      kernel_w = kernel_size,
                      stride = stride,
                      activation_fn = act_fn,
                      enable_batch_norm = bn,
                      enable_bias = bias,
                      quantize = IS_QUANTIZE,
                      padding = pad,
                      weight_bitwidth = BITW,
                      input_bitwidth = BITA,
                      output_bitwidth = 32,
                      last_layer = True,
                      clip_max_value = CLIP_MAX_VALUE,
                      weight_factor = WEIGHT_FACTOR,
                      target_device = TARGET_DEVICE)


def dwConv(in_channels, kernel_size=None, stride=1, pad=0, groups=1, dilation=1, bias=False, bn=True, act=False, first=False, last=False):
    assert(groups==1)
    assert(dilation==1)
    act_fn = nn.ReLU(inplace=True) if not IS_QUANTIZE and act else None
    if isinstance(kernel_size, tuple):
        h = int(kernel_size[0])
        w = int(kernel_size[1])
    else:
        h = w = int(kernel_size)
    return ops.DepthwiseConv2D(in_channels,
                      kernel_h = h,
                      kernel_w = w,
                      stride = stride,
                      activation_fn = act_fn,
                      enable_batch_norm = bn,
                      enable_bias = bias,
                      quantize = IS_QUANTIZE,
                      padding = pad,
                      weight_bitwidth = BITW,
                      input_bitwidth = BITA,
                      output_bitwidth = BITA,
                      clip_max_value = CLIP_MAX_VALUE,
                      weight_factor = WEIGHT_FACTOR,
                      target_device = TARGET_DEVICE)

def maxpool(kernel_size, stride=2, padding=0):
    return ops.Maxpool2D(kernel_h=kernel_size, kernel_w=kernel_size, stride=stride, padding=padding, target_device = TARGET_DEVICE)

def avgpool(in_channels, kernel_size, stride=2, padding=0):
    return ops.Avgpool2D(in_channels,
                         kernel_h=kernel_size,
                         kernel_w=kernel_size,
                         stride=stride,
                         padding=padding,
                         quantize = IS_QUANTIZE,
                         input_bitwidth = BITA,
                         output_bitwidth = BITA,
                         target_device = TARGET_DEVICE)

def add(channels):
    return ops.Shortcut(channels,
                        quantize = IS_QUANTIZE,
                        input_bitwidth = BITA,
                        output_bitwidth = BITA,
                        target_device = TARGET_DEVICE)

def adaptiveAvgpool(channels, keepdim = False, last_layer=False):
    return ops.AdaptiveAvgpool2D(channels,
                                 keepdim = keepdim,
                                 quantize = IS_QUANTIZE,
                                 input_bitwidth = BITA,
                                 output_bitwidth = 32 if last_layer else BITA,
                                 last_layer = last_layer,
                                 target_device = TARGET_DEVICE)

def maxpool2D(kernel_h=2, kernel_w=2, stride=2):
    return ops.Maxpool2D(kernel_h=kernel_h,
                         kernel_w=kernel_w,
                         stride=stride,
                         padding=0,
                         target_device = TARGET_DEVICE)

def flatten(shape_list):
    return ops.Flatten(shape_list, target_device = TARGET_DEVICE)

def qFullConnect(in_channels, out_channels, pad=0, bias=True, bn=False, act=False, last=False):
    act_fn = nn.ReLU(inplace=True) if not IS_QUANTIZE and act else None
    if(last):
        assert(act_fn == None)
    return ops.FullConnected(in_channels,
                             out_channels,
                             activation_fn = act_fn,
                             enable_batch_norm = bn,
                             enable_bias = bias,
                             quantize = IS_QUANTIZE,
                             last_layer = last,
                             weight_bitwidth = BITW,
                             input_bitwidth = BITA,
                             output_bitwidth = 32 if last else BITA,
                             clip_max_value = CLIP_MAX_VALUE,
                             weight_factor = WEIGHT_FACTOR,
                             target_device = TARGET_DEVICE)

def route():
    return ops.Route(target_device=TARGET_DEVICE)

def channel_split(split):
    return ops.Channel_Split(split, target_device = TARGET_DEVICE)

def channel_shuffle(groups):
    return ops.Channel_Shuffle(groups, target_device = TARGET_DEVICE)
