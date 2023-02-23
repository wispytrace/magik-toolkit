import glob
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from . import torch_utils  # , google_utils

matplotlib.rc('font', **{'size': 11})

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)

def xyxy2xywh(x):
    # Transform box coordinates from [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right) to [x, y, w, h] 
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  
    s = [len(unique_classes), tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU: 
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU: 
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU: 
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)

def smooth_BCE(eps=0.1):  
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def compute_loss(p, targets, model):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vec = build_targets(model, targets)
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red)

    cp, cn = smooth_BCE(eps=0.0)

    # Compute losses
    np, ng = 0, 0  # number grid points, targets
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj
        np += tobj.numel()

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ng += nb
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchor_vec[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou computation
            lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  # giou loss
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn)  # targets
                t[range(nb), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE
                # lcls += CE(ps[:, 5:], tcls[i])  # CE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    if red == 'sum':
        bs = tobj.shape[0]  # batch size
        lobj *= 3 / (6300 * bs) * 2  # 3 / np * 2
        if ng:
            lcls *= 3 / ng / model.nc
            lbox *= 3 / ng

    loss = lbox + lobj + lcls
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(model, targets):
    # targets = [image, class, x, y, w, h]

    nt = targets.shape[0]
    tcls, tbox, indices, av = [], [], [], []
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    reject, use_all_anchors = True, True
    for i in model.yolo_layers:
        # get number of grid points and anchor vec for this yolo layer
        if multi_gpu:
            ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec

        # iou of targets-anchors
        t, a = targets, []
        gwh = t[:, 4:6] * ng
        if nt:
            iou = wh_iou(anchor_vec, gwh)  # iou(3,n) = wh_iou(anchor_vec(3,2), gwh(n,2))

            if use_all_anchors:
                na = anchor_vec.shape[0]  # number of anchors
                a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)
                t = targets.repeat([na, 1])
                gwh = gwh.repeat([na, 1])
            else:  # use best anchor only
                iou, a = iou.max(0)  # best iou and anchor

            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            if reject:
                j = iou.view(-1) > model.hyp['iou_t']  # iou threshold hyperparameter
                t, a, gwh = t[j], a[j], gwh[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))

        # Box
        gxy -= gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        av.append(anchor_vec[a])  # anchor vec

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' % (model.nc, model.nc - 1, c.max())

    return tcls, tbox, indices, av


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, multi_label=True):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Box constraints
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    method = 'merge'
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * len(prediction)

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply conf constraint
        x = x[x[:, 4] > conf_thres]

        # Apply width-height constraint
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)

        # Apply finite constraint
        if not torch.isfinite(x).all():
            x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # if method == 'fast_batch':
        #    x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5]  # classes
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        if method == 'merge':  # Merge NMS (boxes merged using weighted mean)
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
            if n < 1E4:  # update boxes
                # weights = (box_iou(boxes, boxes).tril_() > iou_thres) * scores.view(-1, 1)  # box weights
                # weights /= weights.sum(0)  # normalize
                # x[:, :4] = torch.mm(weights.T, x[:, :4])
                weights = (box_iou(boxes[i], boxes) > iou_thres) * scores[None]  # box weights
                x[i, :4] = torch.mm(weights / weights.sum(1, keepdim=True), x[:, :4])  # boxes(i,4) = w(i,n) * boxes(n,4)
        elif method == 'vision':
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        elif method == 'fast':  
            iou = box_iou(boxes, boxes).triu_(diagonal=1)  # upper triangular iou matrix
            i = iou.max(0)[0] < iou_thres

        output[xi] = x[i]
    return output


def get_yolo_layers(model):
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def print_model_biases(model):
    # prints the bias neurons preceding each yolo layer
    print('\nModel Bias Summary: %8s%18s%18s%18s' % ('layer', 'regression', 'objectness', 'classification'))
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for l in model.yolo_layers:  # print pretrained biases
        try:
            if multi_gpu:
                na = model.module.module_list[l].na  # number of anchors
                b = model.module.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
            else:
                na = model.module_list[l].na
                b = model.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
            print(' ' * 20 + '%8g %18s%18s%18s' % (l, '%5.2f+/-%-5.2f' % (b[:, :4].mean(), b[:, :4].std()),
                                                   '%5.2f+/-%-5.2f' % (b[:, 4].mean(), b[:, 4].std()),
                                                   '%5.2f+/-%-5.2f' % (b[:, 5:].mean(), b[:, 5:].std())))
        except:
            pass

def kmean_anchors(path='../coco/train2017.txt', n=9, img_size=(608, 608)):
    # from utils.utils import *; _ = kmean_anchors()
    # Produces a list of target kmeans suitable for use in *.cfg files
    from utils.datasets import LoadImagesAndLabels
    thr = 0.20  # IoU threshold

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        iou = wh_iou(wh, torch.Tensor(k))
        max_iou = iou.max(1)[0]
        bpr, aat = (max_iou > thr).float().mean(), (iou > thr).float().mean() * n  # best possible recall, anch > thr
        print('%.2f iou_thr: %.3f best possible recall, %.2f anchors > thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, IoU_all=%.3f/%.3f-mean/best, IoU>thr=%.3f-mean: ' %
              (n, img_size, iou.mean(), max_iou.mean(), iou[iou > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    def fitness(k):  # mutation fitness
        iou = wh_iou(wh, torch.Tensor(k))  # iou
        max_iou = iou.max(1)[0]
        return max_iou.mean()  # product

    # Get label wh
    wh = []
    dataset = LoadImagesAndLabels(path, augment=True, rect=True, cache_labels=True)
    nr = 1 if img_size[0] == img_size[1] else 10  # number augmentation repetitions
    for s, l in zip(dataset.shapes, dataset.labels):
        wh.append(l[:, 3:5] * (s / s.max()))  # image normalized to letterbox normalized wh
    wh = np.concatenate(wh, 0).repeat(nr, axis=0)  # augment 10x
    wh *= np.random.uniform(img_size[0], img_size[1], size=(wh.shape[0], 1))  # normalized to pixels (multi-scale)
    wh = wh[(wh > 2.0).all(1)]  # remove below threshold boxes (< 2 pixels wh)

    use_darknet = False
    if use_darknet and n == 9:
        k = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
    else:
        # Kmeans calculation
        from scipy.cluster.vq import kmeans
        print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
        s = wh.std(0)  # sigmas for whitening
        k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
        k *= s
    wh = torch.Tensor(wh)
    k = print_results(k)

    # Evolve
    npr = np.random
    f, sh, ng, mp, s = fitness(k), k.shape, 1000, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    for _ in tqdm(range(ng), desc='Evolving anchors'):
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)  # 98.6, 61.6
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            print_results(k)
    k = print_results(k)

    return k

def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    w = [0.0, 0.01, 0.99, 0.00]  # weights for [P, R, mAP, F1]@0.5 or [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_images(imgs, targets, paths=None, fname='images.png'):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()
    # targets = targets[targets[:, 1] == 21]  # plot only one class

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    bs = min(bs, 16)  # limit plot to 16 images
    ns = np.ceil(bs ** 0.5)  # number of subplots

    for i in range(bs):
        boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
        plt.axis('off')
        if paths is not None:
            s = Path(paths[i]).name
            plt.title(s[:min(len(s), 40)], fontdict={'size': 8})  # limit to 40 characters
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close()

