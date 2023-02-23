import torch.nn.functional as F

from utils.parse_config import *
from utils.utils import *
import sys
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python import ops


def create_modules(module_defs, img_size):
    # Constructs module list of layer blocks from module configuration in module_defs

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    ##quantize
    target_device = hyperparams['target_device']
    bita = int(hyperparams['bita'])
    bitw = int(hyperparams['bitw'])
    is_quantize = bool(hyperparams['is_quantize'])

    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            size = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            
            ##activation just support relu6
            if mdef['activation'] == 'relu6':
                act_fn = None if is_quantize else nn.ReLU6()
            elif mdef['activation'] == 'None':
                act_fn = None
            else:
                print('convolutional no support activation funtion!!')
                exit()

            modules.add_module('Conv2d', ops.Conv2D(in_channels=output_filters[-1],
                                                    out_channels=filters,
                                                    stride=stride,
                                                    kernel_h = size,
                                                    kernel_w = size,
                                                    activation_fn = act_fn,
                                                    enable_batch_norm=bool(bn),
                                                    enable_bias=not bool(bn),
                                                    quantize=is_quantize,
                                                    weight_bitwidth=bitw,
                                                    first_layer=True if bool(mdef['first_layer']) else False,
                                                    input_bitwidth=8 if bool(mdef['first_layer']) else bita,
                                                    output_bitwidth=32 if bool(mdef['last_layer']) else bita,
                                                    padding=(size - 1) // 2 if mdef['pad'] else 0,
                                                    weight_factor=float(mdef['weight_factor']),
                                                    clip_max_value=float(mdef['clip_max_value']),
                                                    last_layer=True if bool(mdef['last_layer']) else False,
                                                    quantize_last_feature=False,
                                                    target_device=target_device))
            if not bn:
                routs.append(i)  # detection output (goes into yolo layer)  ##only conv 1x1 should be detection output
            
        elif mdef['type'] == 'conv_dw':
            if bool(mdef['first_layer']) or bool(mdef['last_layer']):
                print('depthwise not support first or last layer!')
                exit()
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            size = mdef['size']
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            ##activation just support relu6
            if mdef['activation'] == 'relu6':
                act_fn = None if is_quantize else nn.ReLU6()
            elif mdef['activation'] == 'None':
                act_fn = None
            else:
                print('no support activation funtion!!')
                exit()
            

            modules.add_module('Conv2d_dw', ops.DepthwiseConv2D(in_channels=output_filters[-1], stride=stride,
                                                                kernel_h=size,kernel_w=size,
                                                                activation_fn=act_fn,
                                                                enable_batch_norm=bool(bn), 
                                                                enable_bias=not bool(bn), quantize=is_quantize, 
                                                                padding=(size - 1) // 2 if mdef['pad'] else 0, 
                                                                weight_bitwidth=bitw,
                                                                input_bitwidth=bita,
                                                                output_bitwidth=bita,
                                                                clip_max_value=float(mdef['clip_max_value']),
                                                                weight_factor=float(mdef['weight_factor']),
                                                                target_device=target_device))
            
            modules.add_module('Conv2d1x1', ops.Conv2D(in_channels=output_filters[-1], 
                                                       out_channels=filters, stride=1, 
                                                       kernel_h=1,kernel_w=1,
                                                       activation_fn=act_fn,
                                                       enable_batch_norm=bool(bn), 
                                                       enable_bias=not bool(bn), quantize=is_quantize, 
                                                       padding=0, 
                                                       weight_bitwidth = bitw, 
                                                       input_bitwidth=bita, 
                                                       output_bitwidth=bita,
                                                       clip_max_value=float(mdef['clip_max_value']), 
                                                       weight_factor=float(mdef['weight_factor']), target_device=target_device))

        elif mdef['type'] == 'unpooling':
            kernel_h = mdef['stride']
            kernel_w = mdef['stride']
            modules.add_module('Unpooling', ops.Unpool2D(kernel_h=kernel_h, kernel_w=kernel_w,target_device = target_device))

        elif mdef['type'] == 'maxpool':
            size = mdef['size']
            stride = mdef['stride']
            modules = ops.Maxpool2D(kernel_h=size, kernel_w=size, stride=stride, target_device=target_device)
           

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            
            modules = ops.Shortcut( out_channels = filters,
                                    quantize = is_quantize,
                                    quantize_last_feature = False,
                                    last_layer = False,
                                    input_bitwidth = bita,
                                    output_bitwidth = bita,
                                    target_device = target_device)

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            l = mdef['from'] if 'from' in mdef else []
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
                                nc=mdef['classes'],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=l)  # output layers

            try:
                bo = -4.5  #  obj bias
                bc = math.log(1 / (modules.nc - 0.99))  # cls bias: class probability is sigmoid(p) = 1/nc

                j = l[yolo_index] if 'from' in mdef else -1
                bias_ = module_list[j].Conv2d.Conv2d.bias  # shape(255,)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                bias[:, 4] += bo - bias[:, 4].mean()  # obj
                bias[:, 5:] += bc - bias[:, 5:].mean()  # cls, view with utils.print_model_biases(model)
                module_list[j].Conv2d.Conv2d.bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints

    def forward(self, p, img_size, out):
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)

        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    def __init__(self, cfg, img_size=(416, 416)):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        self.yolo_layers = get_yolo_layers(self)

        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info()  # print model description

    def forward(self, x, verbose=False):
        img_size = x.shape[-2:]
        yolo_out, out = [], []
        if verbose:
            str1 = ''
            print('0', x.shape)
        
        x = ops.Preprocess(0, 255.)(x)
        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'unpooling', 'maxpool','conv_dw']:
                x = module(x)
            elif mtype == 'shortcut':  # sum
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    s = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str1 = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, s)])
                x = module([x] + [out[j] for j in self.module_defs[i]["from"]])
            elif mtype == 'route':  # concat
                layers = mdef['layers']
                if verbose:
                    l = [i - 1] + layers  # layers
                    s = [list(x.shape)] + [list(out[i].shape) for i in layers]  # shapes
                    str1 = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, s)])
                if len(layers) == 1:
                    x = out[layers[0]]
                else:
                    x = ops.Route()([out[i] for i in layers])
                        
            elif mtype == 'yolo':
                yolo_out.append(module(x, img_size, out))
            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), mtype), list(x.shape), str)
                str1 = ''
        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


