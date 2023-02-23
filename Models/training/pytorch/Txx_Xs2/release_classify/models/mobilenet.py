from torch import nn
from models.basic import *

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, first=False):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if first:
            super(ConvBNReLU, self).__init__(
                qConv(in_planes, out_planes, kernel_size, stride=stride, pad=padding, bias=False, bn=True, act=True, first=first)
            )
        else:
            super(ConvBNReLU, self).__init__(
                dwConv(in_planes, kernel_size, stride=stride, pad=padding, bias=False, bn=True, act=True),
                qConv(in_planes, out_planes, 1, stride=1, pad=0, bias=False, bn=True, act=True)
            )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            dwConv(hidden_dim, 3, stride=stride, pad=1, bias=False, bn=True, act=True),
            #ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            # norm_layer(oup),
            qConv(hidden_dim, oup, 1, bias=False, bn=True, act=False)
        ])
        self.conv = nn.Sequential(*layers)

        self.add = ops.Shortcut(out_channels=oup,
                                quantize = IS_QUANTIZE,
                                clip_max_value = CLIP_MAX_VALUE,
                                input_bitwidth = BITA,
                                output_bitwidth = BITA,
                                target_device = TARGET_DEVICE)

    def forward(self, x):
        if self.use_res_connect:
            # return x + self.conv(x)
            return self.add((x,self.conv(x)))
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=100,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 512

        self.preprocess = preprocess()

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [2, 32, 2, 2],
                [2, 32, 3, 2],
                [2, 64, 4, 2],
                [2, 96, 3, 1],
                [2, 160, 3, 2],
                [2, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer, first=True)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.flatten = flatten([-1, self.last_channel*1*1])
        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(0.2),
            # nn.Linear(self.last_channel, num_classes),
            qFullConnect(self.last_channel, num_classes, bias=True, bn=False, act=False, last=True)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, ops.Conv2D):
                nn.init.kaiming_normal_(m.Conv2d.weight, mode='fan_out')
                if m.Conv2d.bias is not None:
                    nn.init.zeros_(m.Conv2d.bias)
                if m.enable_batch_norm:
                    nn.init.ones_(m.BatchNorm2d.batch_norm.weight)
                    nn.init.zeros_(m.BatchNorm2d.batch_norm.bias)
            elif isinstance(m, ops.FullConnected):
                nn.init.normal_(m.Linear.weight, 0, 0.01)
                if m.Linear.bias is not None:
                    nn.init.zeros_(m.Linear.bias)
                if m.enable_batch_norm:
                    nn.init.ones_(m.BatchNorm1d.batch_norm.weight)
                    nn.init.zeros_(m.BatchNorm1d.batch_norm.bias)

    def _forward_impl(self, x):
        x = self.preprocess(x)
        x = self.features(x)
        #x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    return model
