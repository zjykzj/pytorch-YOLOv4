import torch
from torch import nn
import torch.nn.functional as F
from tool.torch_utils import *
from tool.yolo_layer import YoloLayer


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size

        if inference:
            # 在推理阶段，使用

            # B = x.data.size(0)
            # C = x.data.size(1)
            # H = x.data.size(2)
            # W = x.data.size(3)

            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1). \
                expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3),
                       target_size[3] // x.size(3)). \
                contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            # 在训练阶段，使用最近邻插值
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2
        # H_out = (H_in + 2 * Pad - Dilation * (Kernel -1) -1) / Stride + 1 = (H_in - 1) / Stride + 1
        # 保持卷积操作前后特征图空间尺寸不变

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    """
    是否执行残差连接，默认为True
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        # input: [1, 3, 608, 608]
        x1 = self.conv1(input)  # [1, 32, 608, 608]
        x2 = self.conv2(x1)  # [1, 64, 304, 304]
        x3 = self.conv3(x2)  # [1, 64, 304, 304]
        # route -2
        x4 = self.conv4(x2)  # [1, 64, 304, 304]
        # x4: []
        x5 = self.conv5(x4)  # [1, 32, 304, 304][=
        x6 = self.conv6(x5)  # [1, 64, 304, 304]
        # shortcut -3
        x6 = x6 + x4  # [1, 64, 304, 304]

        x7 = self.conv7(x6)  # [1, 64, 304, 304]
        # [route]
        # layers = -1, -7
        # 路由层，相当于连接不同层的特征数据
        x7 = torch.cat([x7, x3], dim=1)  # [1, 128, 304, 304]
        x8 = self.conv8(x7)  # [1, 64, 304, 304]
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

        # 输入输出通道数均为64，遍历两个ConvBNAct
        self.resblock = ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        # 路由层
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')

        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')

        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        # 推理状态还是训练状态
        self.inference = inference

        # 应用不同的ConvBNAct生成不一样的特征数据
        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        # SPP
        # SPP操作，采用三个不同大小的池话层得到
        # H_out = (H_in + 2 * Pad - dilate * (Kernel - 1) - 1) / Stride + 1
        # H_out = (H_in + 2 * (5 // 2) - 1 * (5 - 1) -1) / 1 + 1 = (H_in + 4 - 4 -1) / 1 + 1 = H_in
        # H_out = (H_in + 2 * (9 // 2) - 1 * (9 - 1) -1) / 1 + 1 = (H_in + 8 - 8 - 1) / 1 + 1 = H_in
        # H_out = (H_in + 2 * (13 // 2) - 1 * (13 - 1) - 1) / 1 + 1 =　(H_int + 12 - 12 -1) / 1 + 1 = H_in
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3, inference=False):
        # 首先执行SPP层
        x1 = self.conv1(input)  # [1, 512, 19, 19]
        x2 = self.conv2(x1)  # [1, 1024, 19, 19]
        x3 = self.conv3(x2)  # [1, 512, 19, 19]
        # SPP
        m1 = self.maxpool1(x3)  # [1, 512, 19, 19]
        m2 = self.maxpool2(x3)  # [1, 512, 19, 19]
        m3 = self.maxpool3(x3)  # [1, 512, 19, 19]
        spp = torch.cat([m3, m2, m1, x3], dim=1)  # [1, 2048, 19, 19]
        # SPP end
        # SPP层得到的应该就是最后的卷积层特征向量
        # 后面的就是FPN操作，执行多个卷积操作，然后上采样到上一层特征空间尺寸大小
        # 连接对应特征数据后，再次执行多个卷积操作，然后上采样到上上一层特征空间大小
        # 一共进行了两次上采样操作
        x4 = self.conv4(spp)  # [1, 512, 19, 19]
        x5 = self.conv5(x4)  # [1, 1024, 19, 19]
        x6 = self.conv6(x5)  # [1, 512, 19, 19]
        x7 = self.conv7(x6)  # [1, 256, 19, 19]
        # UP
        # 得到最终的特征数据后，进行上采样，放大到downsample4大小
        up = self.upsample1(x7, downsample4.size(), self.inference)  # [1, 256, 38, 38]
        # R 85
        x8 = self.conv8(downsample4)  # [1, 256, 38, 38]
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)  # [1, 512, 38, 38]

        x9 = self.conv9(x8)  # [1, 256, 38, 38]
        x10 = self.conv10(x9)  # [1, 512, 38, 38]
        x11 = self.conv11(x10)  # [1, 256, 38, 38]
        x12 = self.conv12(x11)  # [1, 512, 38, 38]
        x13 = self.conv13(x12)  # [1, 256, 38, 38]
        x14 = self.conv14(x13)  # [1, 128, 38, 38]

        # UP
        up = self.upsample2(x14, downsample3.size(), self.inference)  # [1, 128, 76, 76]
        # R 54
        x15 = self.conv15(downsample3)  # [1, 128, 76, 76]
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)  # [1, 256, 76, 76]

        x16 = self.conv16(x15)  # [1, 128, 76, 76]
        x17 = self.conv17(x16)  # [1, 256, 76, 76]
        x18 = self.conv18(x17)  # [1, 128, 76, 76]
        x19 = self.conv19(x18)  # [1, 256, 76, 76]
        x20 = self.conv20(x19)  # [1, 128, 76, 76]

        # 最终输出三个特征数据，作用于不同大小的边界框预测
        return x20, x13, x6


class Yolov4Head(nn.Module):
    def __init__(self, output_ch, n_classes, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
            anchor_mask=[0, 1, 2], num_classes=n_classes,
            anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
            num_anchors=9, stride=8)

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo2 = YoloLayer(
            anchor_mask=[3, 4, 5], num_classes=n_classes,
            anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
            num_anchors=9, stride=16)

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo3 = YoloLayer(
            anchor_mask=[6, 7, 8], num_classes=n_classes,
            anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
            num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):
        # Head层负责边界框预测
        # input1: [1, 128, 76, 76]
        # input2: [1, 256, 38, 38]
        # input3: [1, 512, 19, 19]
        # 按照论文描述的，使用PAN架构进行特征增强
        # 从下到上进行特征投影，将加法操作设置为通道连接操作
        # 然后使用YoloLayer执行边界框预测
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)

        if self.inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return get_region_boxes([y1, y2, y3])

        else:
            return [x2, x10, x18]


class Yolov4(nn.Module):
    def __init__(self, yolov4conv137weight=None, n_classes=80, inference=False):
        super().__init__()

        # 输出通道数：坐标（x/y/w/h）+ 坐标框置信度 + 类别数
        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        # neck
        self.neck = Neck(inference)
        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neck)
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)

        # head
        self.head = Yolov4Head(output_ch, n_classes, inference)

        self.num_classes = n_classes

    def forward(self, input):
        # input: [1, 3, 608, 608]
        # 空间尺寸减半，通道数目倍增
        d1 = self.down1(input)  # [1, 64, 304, 304]
        d2 = self.down2(d1)  # [1, 128, 152, 152]
        d3 = self.down3(d2)  # [1, 256, 76, 76]
        d4 = self.down4(d3)  # [1, 512, 38, 38]
        d5 = self.down5(d4)  # [1, 1024, 19, 19]

        # Neck层负责特征融合
        # d5: [1, 128, 19, 19]
        # d4: [1, 512, 38, 38]
        # d3: [1, 256, 76, 76]
        x20, x13, x6 = self.neck(d5, d4, d3)
        # 获取三个特征图，通道数
        # x20: [1, 128, 76, 76]
        # x13: [1, 256, 38, 38]
        # x6: [1, 512, 19, 19]

        # Head层负责边界框预测
        output = self.head(x20, x13, x6)
        # List[2]
        # [1, 22743, 1, 4]
        return output


if __name__ == "__main__":
    import sys
    import cv2

    namesfile = None
    if len(sys.argv) == 6:
        # 类别数
        n_classes = int(sys.argv[1])
        # 权重文件
        weightfile = sys.argv[2]
        # 图像文件
        imgfile = sys.argv[3]
        # 输入高
        height = int(sys.argv[4])
        # 输入宽
        width = int(sys.argv[5])
    elif len(sys.argv) == 7:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        height = int(sys.argv[4])
        width = int(sys.argv[5])
        namesfile = sys.argv[6]
    else:
        print('Usage: ')
        print('  python models.py num_classes weightfile imgfile namefile')

    # 创建Pytorch模型 - 作者自定义的YOLOv4
    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    # pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    # pretrained_dict = {k.replace('neek.', 'neck.'): v for k, v in pretrained_dict.items()}  # strip the names
    # model.load_state_dict(pretrained_dict)

    # GPU推理
    use_cuda = True
    if use_cuda:
        model.cuda()

    # 读取图片
    img = cv2.imread(imgfile)

    # 图像预处理：图像缩放 + 颜色通道转换
    # Inference input size is 416*416 does not mean training size is the same
    # Training size could be 608*608 or even other sizes
    # Optional inference sizes:
    #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
    #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect

    # 第一次检测作为warm up
    for i in range(2):  # This 'for' loop is for speed check
        # Because the first iteration is usually longer
        boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

    # 指定类名文件
    if namesfile == None:
        if n_classes == 20:
            namesfile = 'data/voc.names'
        elif n_classes == 80:
            namesfile = 'data/coco.names'
        else:
            print("please give namefile")

    # 加载类名列表
    class_names = load_class_names(namesfile)
    # 绘制边界框
    plot_boxes_cv2(img, boxes[0], 'predictions.jpg', class_names)
