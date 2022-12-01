import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tool.region_loss import RegionLoss
from tool.yolo_layer import YoloLayer
from tool.config import *
from tool.torch_utils import *


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class MaxPoolDark(nn.Module):
    """
    MaxPool2D + Pad操作，保持操作前后空间感受野大小不变
    池化层空间尺寸计算公式：
    H_out = (H_in + 2 * P - Dilate * (kernel - 1) -1) / S + 1
    Pad计算公式：
    H_out = H_in + Pad_top + Pad_bottom
    W_out = W_in + Pad_left + Pad_right
    """

    def __init__(self, size=2, stride=1):
        super(MaxPoolDark, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        '''
        darknet output_size = (input_size + p - k) / s +1
        p : padding = k - 1
        k : size
        s : stride
        torch output_size = (input_size + 2*p -k) / s +1
        p : padding = k//2
        '''
        p = self.size // 2
        if ((x.shape[2] - 1) // self.stride) != ((x.shape[2] + 2 * p - self.size) // self.stride):
            padding1 = (self.size - 1) // 2
            padding2 = padding1 + 1
        else:
            padding1 = (self.size - 1) // 2
            padding2 = padding1
        if ((x.shape[3] - 1) // self.stride) != ((x.shape[3] + 2 * p - self.size) // self.stride):
            padding3 = (self.size - 1) // 2
            padding4 = padding3 + 1
        else:
            padding3 = (self.size - 1) // 2
            padding4 = padding3
        x = F.max_pool2d(F.pad(x, (padding3, padding4, padding1, padding2), mode='replicate'),
                         self.size, stride=self.stride)
        return x


class Upsample_expand(nn.Module):
    """
    上采样操作，增加空间大小，不改变通道数
    H_out = H_in * stride
    W_out = W_in * stride
    """

    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        # 步长控制扩充倍数
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)

        x = x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1). \
            expand(x.size(0), x.size(1), x.size(2), self.stride, x.size(3), self.stride).contiguous(). \
            view(x.size(0), x.size(1), x.size(2) * self.stride, x.size(3) * self.stride)

        return x


class Upsample_interpolate(nn.Module):
    """
    应用最近邻插值算法上采样图像
    """

    def __init__(self, stride):
        super(Upsample_interpolate, self).__init__()
        # 步长控制扩充倍数
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)

        out = F.interpolate(x, size=(x.size(2) * self.stride, x.size(3) * self.stride), mode='nearest')
        return out


class Reorg(nn.Module):
    """
    重新组织特征数据，减少空间尺寸，增加通道数。目的？？？
    """

    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        # [B, C, H, W] -> [B, C, H/S, S, W/S, S] -> [B, C, H/S, W/S, HS, WS]
        x = x.view(B, C, H / hs, hs, W / ws, ws).transpose(3, 4).contiguous()
        # [B, C, H/S, W/S, HS, WS] -> [B, C, H/S * W/S, HS * WS] -> [B, C, HS * WS, H/S * W/S]
        x = x.view(B, C, H / hs * W / ws, hs * ws).transpose(2, 3).contiguous()
        # [B, C, HS * WS, H/S * W/S] -> [B, C, HS * WS, H/S, W/S] -> [B, HS * WS, C, H/S, W/S]
        x = x.view(B, C, hs * ws, H / hs, W / ws).transpose(1, 2).contiguous()
        # [B, HS * WS, C, H/S, W/S] -> [B, HS * WS * C, H/S, W/S]
        x = x.view(B, hs * ws * C, H / hs, W / ws)
        return x


class GlobalAvgPool2d(nn.Module):
    """
    全局平均池化层
    """

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        # [N, C, H, W] -> [N, C, 1, 1]
        x = F.avg_pool2d(x, (H, W))
        # [N, C, 1, 1] -> [N, C]
        x = x.view(N, C)
        return x


# for route, shortcut and sam
class EmptyModule(nn.Module):
    """
    空模块，运行过程中根据模块性质（路由／一致性连接以及sam）进行功能填充
    """

    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile, inference=False):
        super(Darknet, self).__init__()
        # 推理模式还是训练模式
        self.inference = inference
        self.training = not self.inference

        # 通过配置文件生成对应blocks，默认使用`/path/to/cfg/yolov4.cfg`
        self.blocks = parse_cfg(cfgfile)
        # 获取输入数据的宽和高
        # 这个应该是darknet指定。对于Pytorch定义模型而言，可以输入任意大小（理论上）
        # blocks[0]是[net]块，该网络通用配置
        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        # 基于blocks参数创建对应模型
        self.models = self.create_network(self.blocks)  # merge conv, bn, leaky
        self.loss = self.models[len(self.models) - 1]

        if self.blocks[(len(self.blocks) - 1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        # 4位大小header ???
        self.header = torch.IntTensor([0, 0, 0, 0])
        # seen ???
        self.seen = 0

    def forward(self, x):
        # 去除第一个块[net]，后续block会参与计算，保存每个block的输出
        ind = -2
        self.loss = None
        outputs = dict()
        out_boxes = []
        # 依据blocks进行计算
        for block in self.blocks:
            ind = ind + 1
            # if ind > 0:
            #    return x

            if block['type'] == 'net':
                # 跳过[net]模块
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'connected']:
                # 卷积层、最大池化层、reorg?、上采样层、平均池化层、softmax层、connected?
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if 'groups' not in block.keys() or int(block['groups']) == 1:
                        # 没有分组
                        x = outputs[layers[0]]
                        outputs[ind] = x
                    else:
                        groups = int(block['groups'])
                        group_id = int(block['group_id'])
                        _, b, _, _ = outputs[layers[0]].shape
                        x = outputs[layers[0]][:, b // groups * group_id:b // groups * (group_id + 1)]
                        outputs[ind] = x
                elif len(layers) == 2:
                    # 按通道维度连接多个特征数据
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                    outputs[ind] = x
                elif len(layers) == 4:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x3 = outputs[layers[2]]
                    x4 = outputs[layers[3]]
                    x = torch.cat((x1, x2, x3, x4), 1)
                    outputs[ind] = x
                else:
                    print("rounte number > 2 ,is {}".format(len(layers)))

            elif block['type'] == 'shortcut':
                # Identity Mapping 恒等映射
                # 负数，从后往前数
                from_layer = int(block['from'])
                # 激活函数类型
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                # 启示层输出特征
                x1 = outputs[from_layer]
                # 上一层输出特征
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'sam':
                from_layer = int(block['from'])
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 * x2
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'yolo':
                # if self.training:
                #     pass
                # else:
                #     boxes = self.models[ind](x)
                #     out_boxes.append(boxes)
                boxes = self.models[ind](x)
                out_boxes.append(boxes)
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))

        if self.training:
            return out_boxes
        else:
            return get_region_boxes(out_boxes)

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()

        # 上一层滤波器个数，也就是上一层输出特征的通道数
        prev_filters = 3
        # 记录每一层输出通道数
        out_filters = []
        # 累积滤波器步长，作用母鸡，估计作用在特征连接或者相加的情况，保持空间大小一致
        prev_stride = 1
        # 记录每一层计算后的累计滤波器步长
        out_strides = []
        # 卷积编码，注意：这个conv_id包含了Conv-BN-Act
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                # 块net保存了网络的通用属性，当前仅需使用输入数据的通道数
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                # 块convolutional保存了Conv-BN-Act的属性
                # 对于卷积层，需要设置滤波器个数，内核大小，步长以及是否填充
                # 对于BN层，设置是否跟随
                # 对于激活层，设置激活层类型
                #
                # 第几个卷积层
                conv_id = conv_id + 1
                # 是否跟随归一化层　0表示没有BN层，1表示增加BN层
                batch_normalize = int(block['batch_normalize'])
                # 滤波器数目
                filters = int(block['filters'])
                # 滤波器内核大小
                kernel_size = int(block['size'])
                # 滤波器步长
                stride = int(block['stride'])
                # 是否进行填充，如果是，表示保持空间感受野大小
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                # 卷积层空间尺寸计算公式：
                # H_out = (H_in + 2 * pad - dilate * (kernel - 1) -1) / stride + 1
                # stride=1 is_pad=1 空间尺寸不变
                # H_out = (H_in + 2 * (kernel - 1) // 2 - 1 * (kernel - 1) - 1) / 1 + 1 = (H_in - 1) / 1 + 1 = H_in
                # stride=2 is_pad=1 空间尺寸减半
                # H_out = (H_in + 2 * (kernel - 1) // 2 - 1 * (kernel - 1) - 1) / 2 + 1 = (H_in - 2) / 2 + 1 = H_in / 2
                # 激活函数类型
                activation = block['activation']

                model = nn.Sequential()
                if batch_normalize:
                    # 是否跟随BN层
                    # conv<编号> = nn.Conv2d(...)
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    # bn<编号> = nn.BatchNorm2d(...)
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    # model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                # 激活函数
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                elif activation == 'mish':
                    model.add_module('mish{0}'.format(conv_id), Mish())
                elif activation == 'linear':
                    model.add_module('linear{0}'.format(conv_id), nn.Identity())
                elif activation == 'logistic':
                    model.add_module('sigmoid{0}'.format(conv_id), nn.Sigmoid())
                else:
                    print("No convolutional activation named {}".format(activation))

                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'maxpool':
                # 块maxpool定义了池化层属性，指定池化大小以及步长
                # maxpool通常作用于网络中间池话层
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride == 1 and pool_size % 2:
                    # You can use Maxpooldark instead, here is convenient to convert onnx.
                    # Example: [maxpool] size=3 stride=1
                    model = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=pool_size // 2)
                elif stride == pool_size:
                    # You can use Maxpooldark instead, here is convenient to convert onnx.
                    # Example: [maxpool] size=2 stride=2
                    model = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=0)
                else:
                    model = MaxPoolDark(pool_size, stride)
                # 最大池化层空间尺寸计算公式：
                # H_out = floor((H_in + 2 * pad - dilate * (pool - 1) -1) / stride + 1)
                # pool = 5 stride = 1 padding = pool // 2 保持空间尺寸不变
                # H_out = floor((H_in + 2 * (pool // 2) - 1 * (pool - 1) -1) / 1 + 1) = floor(H_in + 4 - 5 + 1) = H_in
                # pool = 9 stride = 1 padding = pool // 2 保持空间尺寸不变
                # H_out = floor((H_in + 2 * (pool // 2) - 1 * (pool - 1) -1) / 1 + 1) = floor(H_in + 8 - 9 + 1) = H_in
                # pool = 13 stride = 1 padding = pool // 2 保持空间尺寸不变
                # H_out = floor((H_in + 2 * (pool // 2) - 1 * (pool - 1) -1) / 1 + 1) = floor(H_in + 12 - 13 + 1) = H_in

                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'avgpool':
                # 块avgpool定义了池化层属性，指定池化大小以及步长
                # avgpool通常作用于最后一个池话层
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                # Softmax层，应该是作用于最后的分类输出
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                # 成员函数，设计了三种：MSELoss / L1Loss / SmoothL1Loss
                if block['_type'] == 'sse':
                    model = nn.MSELoss(reduction='mean')
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(reduction='mean')
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(reduction='mean')
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'reorg':
                # 母鸡
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                prev_stride = prev_stride * stride
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
            elif block['type'] == 'upsample':
                # 上采样层，扩大图像空间面积
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)
                # stride控制空间尺寸倍增数
                # W_dst = W_src * stride

                models.append(Upsample_expand(stride))
                # models.append(Upsample_interpolate(stride))

            elif block['type'] == 'route':
                # 路由层，作用母鸡
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    # 当前层连接一个层？
                    # 有没有分组？
                    if 'groups' not in block.keys() or int(block['groups']) == 1:
                        prev_filters = out_filters[layers[0]]
                        prev_stride = out_strides[layers[0]]
                    else:
                        prev_filters = out_filters[layers[0]] // int(block['groups'])
                        prev_stride = out_strides[layers[0]] // int(block['groups'])
                elif len(layers) == 2:
                    assert (layers[0] == ind - 1 or layers[1] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 4:
                    assert (layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]] + out_filters[layers[2]] + \
                                   out_filters[layers[3]]
                    prev_stride = out_strides[layers[0]]
                else:
                    print("route error!!!")

                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                # 一致性连接
                ind = len(models)
                # 加法操作，通道数保持不变
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'sam':
                # 作用母鸡
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                # 块connected表示激活函数
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'region':
                # 块region表示损失函数，具体作用以及用法还需要调研
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors) // loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(loss)
            elif block['type'] == 'yolo':
                # YOLO层，需要
                yolo_layer = YoloLayer()
                anchors = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_classes = int(block['classes'])
                self.num_classes = yolo_layer.num_classes
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
                yolo_layer.stride = prev_stride
                yolo_layer.scale_x_y = float(block['scale_x_y'])
                # yolo_layer.object_scale = float(block['object_scale'])
                # yolo_layer.noobject_scale = float(block['noobject_scale'])
                # yolo_layer.class_scale = float(block['class_scale'])
                # yolo_layer.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)
            else:
                print('unknown type %s' % (block['type']))

        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'sam':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))

    # def save_weights(self, outfile, cutoff=0):
    #     if cutoff <= 0:
    #         cutoff = len(self.blocks) - 1
    #
    #     fp = open(outfile, 'wb')
    #     self.header[3] = self.seen
    #     header = self.header
    #     header.numpy().tofile(fp)
    #
    #     ind = -1
    #     for blockId in range(1, cutoff + 1):
    #         ind = ind + 1
    #         block = self.blocks[blockId]
    #         if block['type'] == 'convolutional':
    #             model = self.models[ind]
    #             batch_normalize = int(block['batch_normalize'])
    #             if batch_normalize:
    #                 save_conv_bn(fp, model[0], model[1])
    #             else:
    #                 save_conv(fp, model[0])
    #         elif block['type'] == 'connected':
    #             model = self.models[ind]
    #             if block['activation'] != 'linear':
    #                 save_fc(fc, model)
    #             else:
    #                 save_fc(fc, model[0])
    #         elif block['type'] == 'maxpool':
    #             pass
    #         elif block['type'] == 'reorg':
    #             pass
    #         elif block['type'] == 'upsample':
    #             pass
    #         elif block['type'] == 'route':
    #             pass
    #         elif block['type'] == 'shortcut':
    #             pass
    #         elif block['type'] == 'sam':
    #             pass
    #         elif block['type'] == 'region':
    #             pass
    #         elif block['type'] == 'yolo':
    #             pass
    #         elif block['type'] == 'avgpool':
    #             pass
    #         elif block['type'] == 'softmax':
    #             pass
    #         elif block['type'] == 'cost':
    #             pass
    #         else:
    #             print('unknown type %s' % (block['type']))
    #     fp.close()
