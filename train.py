# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 15:07
@Author        : Tianxiaomo
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import time
import logging
import os
import sys
import math
import argparse
from collections import deque
import datetime

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict

from dataset import Yolo_dataset
from cfg import Cfg
from models import Yolov4
from tool.darknet2pytorch import Darknet

from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator

"""shell
python train.py -g "0,1,2,3,4,5,6,7" -dir . -train_label_path ./data/train.txt
"""


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    计算两组边界框两两之间的交集并集比（IoU），输出数组Res大小为[N, K]，N表示bbox_a的长度，K表示bbox_b的长度
    比如Res[3,5]表示bbox_a第3个边界框和bbox_b第5个边界框之间的IoU

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # 边界框数值格式，是按照[x_left_top, y_left_top, x_right_bottom, y_right_bottom]还是按照[x_center, y_center, w, h]
    if xyxy:
        # 计算交集的左上角坐标
        # intersection top left
        # bboxes_a[:, None, :2]，大小为[N, 1, 2]
        # bboxes_b[:, :2]，大小为[K, 2]
        # torch.max(...)，大小为[N, K, 2]
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # 计算交集的右下角坐标
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # 计算并集的左上角和右下角坐标
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        # bboxes_a／高
        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        # bboxes_b／高
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        # bboxes_a[:, 2:]表示边界框右下角的坐标 bboxes_a[:, :2]表示边界框左上角的坐标
        # bboxes_a[:, 2:] - bboxes_a[:, :2]得到bboxes_a的宽和高，大小为[N, 2]
        # torch.prod(...)：每行相乘，得到最后的每个边界框的面积，大小为[N]
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        # 同理，大小为[K]
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    # 创建掩码：计算并集的左上角坐标小于右下角坐标的情况，也就是这两个边界框之间存在并集
    # [N, K, 2] < [N, K, 2] -> [N, K, 2] -> [N, K]
    en = (tl < br).type(tl.type()).prod(dim=2)
    # 计算并集面积 [N, K]
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    # 计算交集面积
    # [N, 1] + [K] - [N, K] = [N, K]
    area_u = area_a[:, None] + area_b - area_i
    # IoU = 交集并集比
    iou = area_i / area_u

    # 针对IoU的优化，包括GIoU / DIoU / CIoU，后续再涉及
    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


class Yolo_loss(nn.Module):
    """
    YOLO网络损失
    关键点：
    1. 锚点框、真值框和预测框之间的计算

    在计算target的时候，首先是计算真值框和锚点框的IoU
    """

    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        """
        YOLO损失计算
        Args:
            n_classes: 数据集类别数
            n_anchors: 锚点框个数
            device:
            batch: 批量大小
        """
        super(Yolo_loss, self).__init__()
        self.device = device
        # 步长，图像宽高压缩比率，用于预测不同大小的预测框
        # 步长越大，说明压缩的越大，那么预测框越大
        # 步长越小，说明压缩的越小，那么预测框越小
        self.strides = [8, 16, 32]
        # 默认图像大小设置为608
        image_size = 608
        # 数据集类别数
        self.n_classes = n_classes
        # 每个网格的锚点个数，默认为3个
        self.n_anchors = n_anchors

        # 在YOLO损失计算过程中，同时需要考虑锚点框、真值标签框和预测框，其中在损失函数和推理函数中均会计算锚点框和预测框
        # 此处存在重复代码，后期可以进行优化
        #
        # 保存每个锚点框的宽/高，共设置了9个锚点框
        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        # 将锚点掩码划分为3组，应用在不同的特征数据上
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        # 置信度阈值设置为0.5
        # 在YOLOv4计算中，将预测框与真值标签框的IoU阈值设置为0.5（而不是YOLOv3中的0.7）
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        # 共使用3个特征图进行预测框计算
        for i in range(3):
            # 基于不同缩放比例计算不同特征数据上的锚点框坐标和长宽
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            #
            # 每个特征图使用不一样大小的锚点框，也就是说，不同特征图负责不同大小的预测框计算
            # 大小为[3, 2] 3表示每个特征图上每个网格使用的锚点框数量 2表示锚点框的宽/高
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            # 大小为[9, 4] 9表示在各个特征图上使用的锚点框总数 4表示预测框坐标 x1y1x2y2 其中
            # x1y1表示左上角坐标，x2y2表示右下角坐标，赋值为锚点框长宽
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)

            # calculate pred - xywh obj cls
            # 特征图大小
            fsize = image_size // self.strides[i]
            # 计算特征图上网格的左上角x0
            # [fsize] -> [batch, 3, fsize, 1]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            # 计算特征图上网格的左上角y0
            # [fsize] -> [batch, 3, fsize, 1] -> [batch, 3, 1, fsize]
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            # 锚点宽　[3] -> [batch, fsize, fsize, 3] -> [batch, 3, fsize, fsize]
            anchor_w = torch.from_numpy(masked_anchors[:, 0]) \
                .repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(device)
            # 锚点高　[3] -> [batch, fsize, fsize, 3] -> [batch, 3, fsize, fsize]
            anchor_h = torch.from_numpy(masked_anchors[:, 1]) \
                .repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(device)

            # 保存每个特征图对应的
            # 锚点框
            self.masked_anchors.append(masked_anchors)
            # ???这个估计是为了和真值标签框进行计算
            self.ref_anchors.append(ref_anchors)
            # 每个网格的左上角坐标x/y以及对应的锚点框长宽
            # 这也等同于x1y1x2y2设置
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self,
                     # 预测框　[B, n_anchors, F_H, F_W, xywh]
                     pred,
                     # 真值框 [B, N_Truths, 5]
                     # [B, n_truths, xywh+cls_id]
                     labels,
                     # 批量大小
                     batchsize,
                     # 特征图大小
                     fsize,
                     # 4+1+self.n_classes
                     n_ch,
                     # 第几个特征图（或者说第几个YOLO层）计算的特征数据）
                     output_id):
        """
        损失计算，需要的是预测结果和每个预测结果对应的真值标签
        """
        # target assignment
        # [B, 3, F_H, F_W, 4+num_classes]
        # 针对边界框坐标以及分类概率计算掩码
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        # [B, 3, F_H, F_W]
        # 针对置信度的掩码
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        # [B, 3, F_H, F_W, 2]
        # 针对xy的缩放因子
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        # [B, 3, F_H, F_W, 5 + self.n_classes]
        # target，大小和预测一致
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)
        # 从上面的定义中，可以发现有一些预测框是不会参与损失计算的

        # labels = labels.cpu().data
        # 并不是每幅图像都有足够数量的真值标签框
        # 计算每幅图像的真值框个数
        # [B, N_Truths, xywh+cls_id] -> [B, N_Truths] -> [B]
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        # 从这个定义中，应该是xyxy设置，然后转换成xywh格式，并且缩放到指定倍数
        # (w + x) / stride / 2
        # (x1 + x2) / 2 = x_center
        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        # (h + y) / stride / 2
        # (y1 + y2) / 2 = y_center
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        # (w - x) / stride
        # (x2 - x1) = w
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        # (h - y) / stride
        # (y2 - y1) = h
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        # 这个目的是计算网格
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        # 单独计算每幅图像的真值标签
        for b in range(batchsize):
            # 第b幅图像的真值框个数
            n = int(nlabel[b])
            if n == 0:
                # 如果该幅图像没有对应的真值框，那么跳过，也就是说target[...]=0
                continue

            # 创建真值框列表
            truth_box = torch.zeros(n, 4).to(self.device)
            # 填充w / h
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            # 去除对应的网格下标，也就是说，真值框所在的网格位置
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            # 计算真值框和预测框之间的IoU，YOLOv4使用CIoU
            #
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True)

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            # pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_ious = bboxes_iou(pred[b].reshape(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        """

        Args:
            xin:　[x2, x10, x18]
                x2: [B, (4 + 1 + n_classes) * 3, H/8, W/8]
                x10: [B, (4 + 1 + n_classes) * 3, H/16, W/16]
                x18: [B, (4 + 1 + n_classes) * 3, H/32, W/32]
            labels:　[num_bboxes, 5] 5表示xywh+cls_id

        Returns:

        """
        # loss: 整体损失
        # loss_xy: 坐标预测损失
        # loss_wh: 长宽预测损失
        # loss_obj: 目标置信度损失
        # loss_cls: 分类置信度损失
        # loss_l2: 并没有参与训练，作为展示使用
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            # 改变特征数据形状，符合后续计算
            # [B, C, F_H, F_W] -> [B, 3, n_ch, F_H, F_W]
            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            # [B, 3, n_ch, F_H, F_W] -> [B, 3, F_H, F_W, n_ch]
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            # 针对坐标xy以及、预测框置信度和分类概率进行逻辑激活
            # sigmoid([B, 3, F_H, F_W, [:2]+[4:n_ch]])
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            # 预测框pred [B, 3, F_H, F_W, 4]
            # B表示图像批量大小
            # 3表示每个网格拥有3个预测框
            # F_H/F_W表示特征数据的高/宽
            # 4表示xywh，预测框中心点坐标以及宽高
            pred = output[..., :4].clone()
            # 对于中心点坐标，其计算方式为每个网格的左上角坐标 + pred[x,y]
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            # 对于预测框高宽，其计算方式为锚点宽高 * exp(w/h)
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            # 计算target以及掩码（某一些预测框不参与损失计算）
            #
            #
            #
            # 掩码的目的在于屏蔽某些目标的损失计算
            # obj_mask: 屏蔽目标置信度的损失计算
            # tgt_mask: 屏蔽边界框坐标以及分类概率的损失计算
            # tgt_scale: 尺度问题
            #
            # 通过结合labels（真值标签框）、锚点框和预测框坐标，得到符合条件的target参与损失计算
            # 注意：output.shape == target.shape
            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            # 对于预测框中心点坐标，使用二元交叉熵损失
            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            # 对于预测框长宽，使用均值平方损失
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            # 对于目标置信度，使用二元交叉熵损失
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
            # 对于分类置信度，使用二元交叉熵损失
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum')
            # 计算平均损失，用于日志记录
            loss_l2 += F.mse_loss(input=output, target=target, reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        # 每一次提取数据包含了图像以及对应bbox列表
        # img: np.ndarray
        images.append([img])
        # box: List[N_boxes, 5]
        bboxes.append([box])
    # [N, H, W, C] -> [N, C, H, W] / 255.0
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    # [N, N_boxes, 5]
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def train(model, device, config, epochs=5, batch_size=1, save_cp=True, log_step=20, img_scale=0.5):
    logging.info(f"train file: {config.train_label}")
    logging.info(f"val file: {config.val_label}")
    train_dataset = Yolo_dataset(config.train_label, config, train=True)
    val_dataset = Yolo_dataset(config.val_label, config, train=False)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 为什么要除以subdivisions，???
    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate)

    val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=8,
                            pin_memory=True, drop_last=True, collate_fn=val_collate)

    writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}')
    # writer.add_images('legend',
    #                   torch.from_numpy(train_dataset.label2colorlegend2(cfg.DATA_CLASSES).transpose([2, 0, 1])).to(
    #                       device).unsqueeze(0))
    max_itr = config.TRAIN_EPOCHS * n_train
    # global_step = cfg.TRAIN_MINEPOCH * n_train
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained:
    ''')

    # learning rate setup
    def burnin_schedule(i):
        # 学习率按照iteration进行迭代，而不是按照epoch进行学习率调整
        # 前burn_in步，执行Linear Warmup
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        # 前steps[0]，保持学习率不变
        elif i < config.steps[0]:
            factor = 1.0
        # [step[0], step[1]]，下降学习率0.1
        elif i < config.steps[1]:
            factor = 0.1
        # step[1]之后，下降学习率0.01
        else:
            factor = 0.01
        return factor

    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            # 最终学习率 = 初始学习率 / 批量大小
            lr=config.learning_rate / config.batch,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    criterion = Yolo_loss(device=device, batch=config.batch // config.subdivisions, n_classes=config.classes)
    # scheduler = ReduceLROnPlatea
    # u(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    save_prefix = 'Yolov4_epoch'
    saved_models = deque()
    model.train()
    for epoch in range(epochs):
        # model.train()
        epoch_loss = 0
        epoch_step = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=50) as pbar:
            for i, batch in enumerate(train_loader):
                global_step += 1
                epoch_step += 1
                images = batch[0]
                bboxes = batch[1]

                # 原始图像：[N, C, H, W]
                images = images.to(device=device, dtype=torch.float32)
                # 真值边界框：[N, 4]
                bboxes = bboxes.to(device=device)

                # 预测边界框
                bboxes_pred = model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                # loss = loss / config.subdivisions
                loss.backward()

                epoch_loss += loss.item()

                if global_step % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if global_step % (log_step * config.subdivisions) == 0:
                    writer.add_scalar('train/Loss', loss.item(), global_step)
                    writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                    writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                    writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                    writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                    writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                    writer.add_scalar('lr', scheduler.get_lr()[0] * config.batch, global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                                        'loss_wh': loss_wh.item(),
                                        'loss_obj': loss_obj.item(),
                                        'loss_cls': loss_cls.item(),
                                        'loss_l2': loss_l2.item(),
                                        'lr': scheduler.get_lr()[0] * config.batch
                                        })
                    logging.debug('Train step_{}: loss : {},loss xy : {},loss wh : {},'
                                  'loss obj : {}，loss cls : {},loss l2 : {},lr : {}'
                                  .format(global_step, loss.item(), loss_xy.item(),
                                          loss_wh.item(), loss_obj.item(),
                                          loss_cls.item(), loss_l2.item(),
                                          scheduler.get_lr()[0] * config.batch))

                pbar.update(images.shape[0])

            if cfg.use_darknet_cfg:
                eval_model = Darknet(cfg.cfgfile, inference=True)
            else:
                eval_model = Yolov4(cfg.pretrained, n_classes=cfg.classes, inference=True)
            # eval_model = Yolov4(yolov4conv137weight=None, n_classes=config.classes, inference=True)
            if torch.cuda.device_count() > 1:
                eval_model.load_state_dict(model.module.state_dict())
            else:
                eval_model.load_state_dict(model.state_dict())
            eval_model.to(device)
            evaluator = evaluate(eval_model, val_loader, config, device)
            del eval_model

            stats = evaluator.coco_eval['bbox'].stats
            writer.add_scalar('train/AP', stats[0], global_step)
            writer.add_scalar('train/AP50', stats[1], global_step)
            writer.add_scalar('train/AP75', stats[2], global_step)
            writer.add_scalar('train/AP_small', stats[3], global_step)
            writer.add_scalar('train/AP_medium', stats[4], global_step)
            writer.add_scalar('train/AP_large', stats[5], global_step)
            writer.add_scalar('train/AR1', stats[6], global_step)
            writer.add_scalar('train/AR10', stats[7], global_step)
            writer.add_scalar('train/AR100', stats[8], global_step)
            writer.add_scalar('train/AR_small', stats[9], global_step)
            writer.add_scalar('train/AR_medium', stats[10], global_step)
            writer.add_scalar('train/AR_large', stats[11], global_step)

            if save_cp:
                try:
                    # os.mkdir(config.checkpoints)
                    os.makedirs(config.checkpoints, exist_ok=True)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                save_path = os.path.join(config.checkpoints, f'{save_prefix}{epoch + 1}.pth')
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                logging.info(f'Checkpoint {epoch + 1} saved !')
                saved_models.append(save_path)
                if len(saved_models) > config.keep_checkpoint_max > 0:
                    model_to_remove = saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except:
                        logging.info(f'failed to remove {model_to_remove}')

    writer.close()


@torch.no_grad()
def evaluate(model, data_loader, cfg, device, logger=None, **kwargs):
    """ finished, tested
    """
    # cpu_device = torch.device("cpu")
    model.eval()
    # header = 'Test:'

    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"], bbox_fmt='coco')

    for images, targets in data_loader:
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(model_input)

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # outputs = outputs.cpu().detach().numpy()
        res = {}
        # for img, target, output in zip(images, targets, outputs):
        for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
            img_height, img_width = img.shape[:2]
            # boxes = output[...,:4].copy()  # output boxes in yolo format
            boxes = boxes.squeeze(2).cpu().detach().numpy()
            boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            boxes[..., 0] = boxes[..., 0] * img_width
            boxes[..., 1] = boxes[..., 1] * img_height
            boxes[..., 2] = boxes[..., 2] * img_width
            boxes[..., 3] = boxes[..., 3] * img_height
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # confs = output[...,4:].copy()
            confs = confs.cpu().detach().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = np.max(confs, axis=1).flatten()
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def get_args(**kwargs):
    """
    基础配置 + 命令行参数配置 = 最终配置
    注意：基础配置和命令行参数配置中部分重合
    """
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
    #                     help='Batch size', dest='batchsize')
    # 学习率设置，默认为0.001
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    # 训练GPU，默认为-1，表示使用cpu
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    # 数据根路径，默认为None，使用基础配置
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    # 预训练权重，默认为None，不使用
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
    # 类别数，默认使用coco数据集，大小为80
    parser.add_argument('-classes', type=int, default=80, help='dataset classes')
    parser.add_argument('-train_label_path', dest='train_label', type=str, default='train.txt', help="train label path")
    # 优化器，默认使用adam
    parser.add_argument(
        '-optimizer', type=str, default='adam',
        help='training optimizer',
        dest='TRAIN_OPTIMIZER')
    # IoU类型，默认使用iou
    parser.add_argument(
        '-iou-type', type=str, default='iou',
        help='iou type (iou, giou, diou, ciou)',
        dest='iou_type')
    parser.add_argument(
        '-keep-checkpoint-max', type=int, default=10,
        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
        dest='keep_checkpoint_max')
    args = vars(parser.parse_args())

    # for k in args.keys():
    #     cfg[k] = args.get(k)
    cfg.update(args)

    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """

    def get_date_str():
        # 年-月-日_时-分-秒
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    # 日志格式：时间 文件名[line:行数] 日志级别: 具体内容
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        # 默认放置在~/temp/log路径下
        log_dir = '~/temp/log/'
    if log_file is None:
        # 日志文件命名：log_<日期>.txt
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        # 是否同步打印到文件和控制台窗口
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def _get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')


if __name__ == "__main__":
    # 初始化日志模块
    logging = init_logger(log_dir='log')
    # Cfg是配置文件，有需要可以手动调整
    # Cfg.cfgfile = os.path.join('cfg', 'yolov4_zj.cfg')
    # Cfg.batch = 256
    cfg = get_args(**Cfg)
    print(cfg)
    # 设置有效GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    # 判断是否使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 使用darknet网络配置还是自定义YOLoV4，默认为True
    if cfg.use_darknet_cfg:
        model = Darknet(cfg.cfgfile)
        model.print_network()
    else:
        model = Yolov4(cfg.pretrained, n_classes=cfg.classes)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:
        train(model=model,
              config=cfg,
              epochs=cfg.TRAIN_EPOCHS,
              device=device, )
    except KeyboardInterrupt:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), 'INTERRUPTED.pth')
        else:
            torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
