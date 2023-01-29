import torch.nn as nn
import torch.nn.functional as F
from tool.torch_utils import *


def yolo_forward(output, conf_thresh, num_classes, anchors, num_anchors, scale_x_y, only_objectness=1,
                 validation=False):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    batch = output.size(0)
    H = output.size(2)
    W = output.size(3)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)

        bxy_list.append(output[:, begin: begin + 2])
        bwh_list.append(output[:, begin + 2: begin + 4])
        det_confs_list.append(output[:, begin + 4: begin + 5])
        cls_confs_list.append(output[:, begin + 5: end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(batch, num_anchors * H * W)

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(batch, num_anchors, num_classes, H * W)
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes] 
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, num_classes)

    # Apply sigmoid(), exp() and softmax() to slices
    bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    bwh = torch.exp(bwh)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.sigmoid(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, W - 1, W), axis=0).repeat(H, 0), axis=0),
                            axis=0)
    grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, H - 1, H), axis=1).repeat(W, 1), axis=0),
                            axis=0)
    # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
    # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        bx = bxy[:, ii: ii + 1] + torch.tensor(grid_x, device=device,
                                               dtype=torch.float32)  # grid_x.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        by = bxy[:, ii + 1: ii + 2] + torch.tensor(grid_y, device=device,
                                                   dtype=torch.float32)  # grid_y.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii: ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1: ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)

    ########################################
    #   Figure out bboxes from slices     #
    ########################################

    # Shape: [batch, num_anchors, H, W]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = torch.cat(bh_list, dim=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw = torch.cat((bx, bw), dim=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh = torch.cat((by, bh), dim=1)

    # normalize coordinates to [0, 1]
    bx_bw /= W
    by_bh /= H

    # Shape: [batch, num_anchors * H * W, 1]
    bx = bx_bw[:, :num_anchors].view(batch, num_anchors * H * W, 1)
    by = by_bh[:, :num_anchors].view(batch, num_anchors * H * W, 1)
    bw = bx_bw[:, num_anchors:].view(batch, num_anchors * H * W, 1)
    bh = by_bh[:, num_anchors:].view(batch, num_anchors * H * W, 1)

    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
    boxes = torch.cat((bx1, by1, bx2, by2), dim=2).view(batch, num_anchors * H * W, 1, 4)
    # boxes = boxes.repeat(1, 1, num_classes, 1)

    # boxes:     [batch, num_anchors * H * W, 1, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    det_confs = det_confs.view(batch, num_anchors * H * W, 1)
    confs = cls_confs * det_confs

    # boxes: [batch, num_anchors * H * W, 1, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return boxes, confs


def yolo_forward_dynamic(output,
                         conf_thresh,
                         num_classes,
                         anchors,
                         num_anchors,
                         scale_x_y,
                         only_objectness=1,
                         validation=False):
    # Output would be invalid if it does not satisfy this assert
    assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    # batch = output.size(0)
    # H = output.size(2)
    # W = output.size(3)

    # x_center/y_center
    bxy_list = []
    # w_box/h_box
    bwh_list = []
    # 检测置信度
    det_confs_list = []
    # 分类置信度
    cls_confs_list = []

    # 首先单独收集各自的数据（坐标／宽高／检测置信度／分类置信度）
    for i in range(num_anchors):
        # 假设有num_anchors个锚点，每个锚点占据(4+1+num_classes)个数据
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)

        # 输出向量的前两位表示预测框的左上角坐标（x0, y0）
        # bxy_list.append([Batch, 2, F_H, F_W])
        bxy_list.append(output[:, begin: begin + 2])
        # 输出向量的第3到4位表示预测框的宽和高（w, h）
        # bwh_list.append([Batch, 2, F_H, F_W])
        bwh_list.append(output[:, begin + 2: begin + 4])
        # 输出向量的第5位表示预测框的置信度
        # det_confs_list.append([Batch, F_H, F_W])
        det_confs_list.append(output[:, begin + 4: begin + 5])
        # 输出向量剩余的数据表示该预测框的分类概率
        # cls_confs_list.append([Batch, num_classes, F_H, F_W])
        cls_confs_list.append(output[:, begin + 5: end])

    # 转换成torch格式
    #
    # len(bxy_list) == num_anchors
    # bxy_list[0].shape == [Batch, 2, F_H, F_W]
    # bxy.shape = [Batch, 2*num_anchors, F_H, F_W]
    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3))

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(output.size(0), num_anchors, num_classes, output.size(2) * output.size(3))
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes] 
    cls_confs = cls_confs.permute(0, 1, 3, 2) \
        .reshape(output.size(0), num_anchors * output.size(2) * output.size(3), num_classes)

    # 对于坐标而言，使用sigmoid进行归一化，同时按照尺度进行偏移
    # Apply sigmoid(), exp() and softmax() to slices
    #
    # sigmoid(bxy)取值范围在(0, 1)之间
    # scale_x_y = 1
    # bxy = torch.sigmoid(bxy)
    bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    # 针对宽高，使用指数进行缩放
    # bwh取值范围大于1（当bwh == 0时，exp(bwh) == 1）
    bwh = torch.exp(bwh)
    # 同样的，对于检测置信度和分类置信度，仍旧使用sigmoid进行归一化
    # 边界框预测置信度在(0, 1)之间
    det_confs = torch.sigmoid(det_confs)
    # 分类概率预测在(0, 1)之间
    cls_confs = torch.sigmoid(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    # 计算网格坐标
    # grid_x.shape =
    # [F_W] -> [1, F_W] -> [F_H, F_W] -> [1, F_H, F_W] -> [1, 1, F_H, F_W]
    # [F_W] = [0, 1, 2, ..., (F_W - 1)]
    # [F_H, F_W]表示F_H行，每行向量大小为[F_W]
    grid_x = np.expand_dims(
        np.expand_dims(
            np.expand_dims(np.linspace(0, output.size(3) - 1, output.size(3)), axis=0)
                .repeat(output.size(2), 0), axis=0),
        axis=0)
    # grid_y.shape =
    # [F_H] -> [F_H, 1] -> [F_H, F_W] -> [1, F_H, F_W] -> [1, 1, F_H, F_W]
    # [F_H] = [0, 1, 2, ..., (F_H - 1)]
    # [F_H, F_W]表示F_W列，每列向量大小为[F_H]
    grid_y = np.expand_dims(np.expand_dims(
        np.expand_dims(np.linspace(0, output.size(2) - 1, output.size(2)), axis=1).repeat(output.size(3), 1), axis=0),
        axis=0)
    # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
    # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        # 每个网格包含了num_anchors个锚点框
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        # 针对每个网格的锚点框，计算预测框
        ii = i * 2
        # Shape: [batch, 1, H, W]
        # 网格中每个锚点框的预测x0，其值相对于网格坐标
        bx = bxy[:, ii: ii + 1] + torch.tensor(grid_x, device=device,
                                               dtype=torch.float32)  # grid_x.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        # 网格中每个锚点框的预测y0，其值相对于网格坐标
        by = bxy[:, ii + 1: ii + 2] + torch.tensor(grid_y, device=device,
                                                   dtype=torch.float32)  # grid_y.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        # 网格中每个锚点框的预测w，其值相对于锚点框宽度
        bw = bwh[:, ii: ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        # 网格中每个锚点框的预测h，其值相对于锚点框高度
        bh = bwh[:, ii + 1: ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)

    ########################################
    #   Figure out bboxes from slices     #
    ########################################

    # Shape: [batch, num_anchors, H, W]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = torch.cat(bh_list, dim=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw = torch.cat((bx, bw), dim=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh = torch.cat((by, bh), dim=1)

    # 基于特征图空间尺寸，将预测边界框宽高归一化
    # normalize coordinates to [0, 1]
    bx_bw /= output.size(3)
    by_bh /= output.size(2)

    # Shape: [batch, num_anchors * H * W, 1]
    # 拉伸到相同形状大小
    bx = bx_bw[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    by = by_bh[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    bw = bx_bw[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    bh = by_bh[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)

    # 从这里可以看出，预测框的bx/by表示预测框的中心点
    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
    # 共Batch幅图像，每幅图像有num_anchors * F_H * F_W个预测框
    boxes = torch.cat((bx1, by1, bx2, by2), dim=2) \
        .view(output.size(0), num_anchors * output.size(2) * output.size(3), 1, 4)
    # boxes = boxes.repeat(1, 1, num_classes, 1)

    # boxes:     [batch, num_anchors * H * W, 1, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    # [batch, num_anchors * H * W] -> [batch, num_anchors * H * W, 1]
    det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    # [batch, num_anchors * H * W, 1] * [batch, num_anchors * H * W, num_classes]
    # 最终置信度计算 = 检测置信度 * 分类置信度
    confs = cls_confs * det_confs

    # boxes: [batch, num_anchors * H * W, 1, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    # 返回预测边界框，以及对应置信度
    return boxes, confs


class YoloLayer(nn.Module):
    '''
    YOLO层，应该和YOLOv3保持一致，关键在于实现是否有优化
    Yolo layer

    ＠model_out:
        while inference,is post-processing inside or outside the model
        true : outside
    '''

    def __init__(self, anchor_mask=[], num_classes=0, anchors=[], num_anchors=1, stride=32, model_out=False):
        """
        Args:
            anchor_mask: [0, 1, 2]
            num_classes: n_classes
            anchors: [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
            num_anchors: 9
            stride: 8
            model_out:
        """
        super(YoloLayer, self).__init__()
        # 锚点掩码，使用第几个锚点框
        self.anchor_mask = anchor_mask
        # 数据集类别数
        self.num_classes = num_classes
        # 全部锚点列表
        self.anchors = anchors
        # 全部锚点个数
        self.num_anchors = num_anchors
        # 锚点步长，等同于锚点列表长度 // 锚点个数。没啥必要
        self.anchor_step = len(anchors) // num_anchors
        #
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

        self.model_out = model_out

    def forward(self, output, target=None):
        # output:
        # 特征数据，大小为[Batch, OUTPUT_CHANNELS, F_H, F_W]
        # 其中OUTPUT_CHANNELS = (4 + 1 + num_classes) * 3
        # output等同于所有预测框的输出
        if self.training:
            # 训练阶段直接输出，不执行预测边界框的计算
            return output

        masked_anchors = []
        for m in self.anchor_mask:
            # 获取该特征层使用的锚点框列表
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        # 缩放到指定倍数
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        # 动态YOLO层前向操作
        return yolo_forward_dynamic(output, self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask),
                                    scale_x_y=self.scale_x_y)
