import torch
import torch.nn as nn
import sys
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python import ops

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, is_quantize, first_layer, input_bitwidth, BITW, BITA, auto=True):
        super(ConvBlock, self).__init__()
        self.conv = ops.Conv2D(in_channels, out_channels, stride=stride, kernel_h=kernel_size, kernel_w=kernel_size, activation_fn=None, enable_batch_norm=True, enable_bias=True, quantize=is_quantize, padding=padding, first_layer=first_layer, weight_bitwidth = BITW, input_bitwidth=input_bitwidth, output_bitwidth=BITA, clip_max_value = 4.0, target_device="Txx", auto_bitwidth=auto)

    def forward(self, input):
        out = self.conv(input)
        return out

class DWConvBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, is_quantize, input_bitwidth, BITW, BITA, auto=True):
        super(DWConvBlock, self).__init__()
        self.conv = ops.DepthwiseConv2D(in_channels, stride=stride, kernel_h=kernel_size, kernel_w=kernel_size, activation_fn=None, enable_batch_norm=True, quantize=is_quantize, padding=padding, weight_bitwidth = BITW, input_bitwidth=input_bitwidth, output_bitwidth=BITA, clip_max_value=4.0, target_device="Txx", auto_bitwidth=auto)

    def forward(self, input):
        out = self.conv(input)
        return out

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_quantize, input_bitwidth, BITW, BITA):
        super(DWConv, self).__init__()
        self.DWConv_1 = DWConvBlock(in_channels, 3, stride, 1, is_quantize, input_bitwidth, BITW, BITA)
        self.conv1x1_1 = ConvBlock(in_channels, out_channels, 1, 1, 0, is_quantize, False, input_bitwidth, BITW, BITA)

    def forward(self, x):
        out = self.DWConv_1(x)
        out = self.conv1x1_1(out)
        return out

class Network(nn.Module):
    def __init__(self, is_quantize=False, BITA = 32, BITW = 32):
        super(Network, self).__init__()

        self.preprocess = ops.Preprocess(0, 255., target_device = "Txx")
        self.conv0 = ConvBlock(3, 16, 3, 1, 1, is_quantize, True, 8, BITW, BITA)
        self.maxpool = ops.Maxpool2D(target_device="Txx")
        self.conv1 = DWConv(16, 32, 3, 1, is_quantize, BITA, BITW, BITA)
        self.route = ops.Route(target_device="Txx")
        self.conv2 = DWConv(48, 32, 3, 2, is_quantize, BITA, BITW, BITA)
        self.conv3 = DWConv(32, 64, 3, 1, is_quantize, BITA, BITW, BITA)
        self.unpool = ops.Unpool2D(2,2,target_device="Txx")
        self.split = ops.Channel_Split([32, 32])
        self.conv4 = DWConv(32, 64, 3, 1, is_quantize, BITA, BITW, BITA)
        self.conv5 = DWConv(32, 64, 3, 1, is_quantize, BITA, BITW, BITA)
        self.shortcut = ops.Shortcut(64, quantize=is_quantize, input_bitwidth=BITA, output_bitwidth = BITA, target_device="Txx")
        self.conv6 = DWConv(64, 128, 3, 2, is_quantize, BITA, BITW, BITA)
        self.conv7 = DWConv(128, 128, 3, 1, is_quantize, BITA, BITW, BITA)
        self.conv8 = DWConv(128, 256, 3, 2, is_quantize, BITA, BITW, BITA)
        self.conv9 = DWConv(256, 256, 3, 1, is_quantize, BITA, BITW, BITA)
        self.conv10 = DWConv(256, 256, 3, 2, is_quantize, BITA, BITW, BITA)
        self.flatten = ops.Flatten([-1, 1*1*256])

        self.fc1 = ops.FullConnected(1*1*256, 128, enable_batch_norm=True, quantize=is_quantize, activation_fn=None, last_layer=False, weight_bitwidth = BITW, input_bitwidth=BITA, output_bitwidth=BITA, clip_max_value=4.0, target_device="Txx")
        self.fc2 = ops.FullConnected(128, 10, enable_bias=True, quantize=is_quantize, activation_fn=None, last_layer=True, weight_bitwidth = BITW, input_bitwidth=BITA, output_bitwidth=32, clip_max_value=4.0, target_device="Txx")
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        pinput = self.preprocess(input)
        conv0 = self.conv0(pinput)
        max1 = self.maxpool(conv0)
        conv1 = self.conv1(max1)
        concat1 = self.route([max1, conv1])
        conv2 = self.conv2(concat1)
        conv3 = self.conv3(conv2)
        unpool1 = self.unpool(conv3)
        split = self.split(unpool1)
        conv4 = self.conv4(split[0])
        conv5 = self.conv5(split[1])
        concat2 = self.shortcut([conv4, conv5])
        conv6 = self.conv6(concat2)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)
        conv9 = self.conv9(conv8)
        conv10 = self.conv10(conv9)
        max2 = self.maxpool(conv10)
        res3 = self.flatten(max2)
        fc1 = self.fc1(res3)
        fc2 = self.fc2(fc1)
        fc = self.sigmoid(fc2)

        return fc


class ConvBlock_T40(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, is_quantize, first_layer, input_bitwidth, BITW, BITA, auto=True):
        super(ConvBlock_T40, self).__init__()
        # self.conv = nn.Sequential(
        #                 nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
        #                 padding=padding, bias=False),
        #                 nn.BatchNorm2d(out_channels),
        #                 nn.ReLU6(inplace=True) #nn.ReLU6(inplace=True) if is_quantize else None
        # )

        self.conv = ops.Conv2D(in_channels, out_channels, stride=stride, kernel_h=kernel_size,
                                kernel_w=kernel_size, activation_fn=None, enable_batch_norm=True, enable_bias=False,
                                quantize=is_quantize, padding=padding, first_layer=first_layer, weight_bitwidth = BITW,
                                input_bitwidth=input_bitwidth, output_bitwidth=BITA, clip_max_value = 6.0, target_device="T40",
                                auto_bitwidth=False)

    def forward(self, input):
        out = self.conv(input)
        return out

class Network_T40(nn.Module):
    def __init__(self, is_quantize=False, BITA = 32, BITW = 32):
        super(Network_T40, self).__init__()
        self.preprocess = ops.Preprocess(0, 255., target_device="T40")

        # self.bn = nn.BatchNorm2d(3)
        self.bn = ops.BatchNorm(3, None, is_quantize, True, 8, BITA, target_device="T40") #first_layer

        self.conv0 = ConvBlock_T40(3, 32, 3, 1, 1, is_quantize, False, 8, BITW, BITA)

        # self.maxpool = nn.MaxPool2d((2, 2), stride=2, padding=0)
        self.maxpool = ops.Maxpool2D(target_device="T40")

        self.conv1 = ConvBlock_T40(32, 32, 3, 1, 1, is_quantize, False, BITA, BITW, BITA)
        self.route = ops.Route(target_device="T40")
        self.conv2 = ConvBlock_T40(64, 32, 3, 2, 1, is_quantize, False, BITA, BITW, BITA)
        self.conv3 = ConvBlock_T40(32, 64, 3, 1, 1, is_quantize, False, BITA, BITW, BITA)

        # self.unpool = torch.nn.Upsample(scale_factor=(2, 2))
        self.unpool = ops.Unpool2D(2, 2,target_device="T40",quantize=is_quantize)

        self.conv4 = ConvBlock_T40(64, 64, 3, 1, 1, is_quantize, False, BITA, BITW, BITA)
        self.conv5 = ConvBlock_T40(64, 64, 3, 1, 1, is_quantize, False, BITA, BITW, BITA)
        self.shortcut = ops.Shortcut(64, quantize=is_quantize, input_bitwidth=BITA, output_bitwidth = BITA, clip_max_value = 2.0, target_device="T40")

        # self.bn1 = nn.Sequential(
        #                nn.BatchNorm2d(64),
        #                nn.PReLU(64)
        # )
        self.bn1 = ops.BatchNorm(64, ops.PReLU(64), is_quantize, False, BITA, BITA, target_device="T40")

        self.conv6 = ConvBlock_T40(64, 128, 3, 2, 1, is_quantize, False, BITA, BITW, BITA)
        self.conv7 = ConvBlock_T40(128, 128, 3, 1, 1, is_quantize, False, BITA, BITW, BITA)
        self.conv8 = ConvBlock_T40(128, 256, 3, 2, 1, is_quantize, False, BITA, BITW, BITA)
        self.conv9 = ConvBlock_T40(256, 256, 3, 1, 1, is_quantize, False, BITA, BITW, BITA)
        self.conv10 = ConvBlock_T40(256, 256, 3, 2, 1, is_quantize, False, BITA, BITW, BITA)

        # self.rnn = torch.nn.LSTM(input_size=256, hidden_size=256, bidirectional=True, batch_first=True)
        #self.rnn = ops.LSTM(256, 256, bidirectional=True, batch_first=True, use_squeeze=True, quantize=is_quantize, input_bitwidth=BITA, output_bitwidth=BITA, weight_bitwidth=BITW, target_device="T40")

        self.flatten = ops.Flatten([-1, 256*1], target_device="T40")

        # self.fc1 = nn.Sequential(
        #                nn.Linear(2*256, 128, bias=True),
        #                nn.BatchNorm1d(128),
        #                nn.ReLU6(inplace=True)
        # )
        self.fc1 = ops.FullConnected(256*1, 128, enable_batch_norm=True, quantize=is_quantize, activation_fn=None, last_layer=False, weight_bitwidth = BITW, input_bitwidth=BITA, output_bitwidth=BITA, clip_max_value=6.0, target_device="T40")

        # self.fc2 = nn.Linear(128, 10, bias=True)
        self.fc2 = ops.FullConnected(128, 10, enable_bias=True, quantize=is_quantize, activation_fn=None, last_layer=True, weight_bitwidth = BITW, input_bitwidth=BITA, output_bitwidth=32, clip_max_value=6.0, target_device="T40") #last_layer

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        pinput = self.preprocess(input)
        pbn = self.bn(pinput)
        conv0 = self.conv0(pbn)
        max1 = self.maxpool(conv0)
        conv1 = self.conv1(max1)

        # concat1 = torch.cat([max1, conv1], 1)
        concat1 = self.route([max1, conv1])

        conv2 = self.conv2(concat1)
        conv3 = self.conv3(conv2)
        unpool1 = self.unpool(conv3)
        conv4 = self.conv4(unpool1)
        max2 = self.maxpool(conv4)
        conv5 = self.conv5(max2)

        # concat2 = max2 + conv5
        concat2 = self.shortcut([max2, conv5])

        concat2 = self.bn1(concat2)
        conv6 = self.conv6(concat2)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)
        conv9 = self.conv9(conv8)
        conv10 = self.conv10(conv9)

        # conv10 = torch.reshape(conv10, [-1, 1, 256])
        #rnn, _ = self.rnn(conv10)

        # res3 = torch.reshape(rnn, (-1, 512))
        res3 = self.flatten(conv10)

        fc1 = self.fc1(res3)
        fc2 = self.fc2(fc1)
        fc = self.sigmoid(fc2)

        return fc2
