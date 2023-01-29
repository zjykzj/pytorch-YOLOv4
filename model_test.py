# -*- coding: utf-8 -*-

"""
@date: 2023/1/29 上午11:40
@file: model_test.py
@author: zj
@description: 
"""

import torch.nn as nn

from models import Conv_Bn_Activation, init_seed, DownSample1, DownSample2, DownSample3, DownSample4, DownSample5, Neck


class TinyNet(nn.Module):

    def __init__(self):
        super(TinyNet, self).__init__()
        # self.conv = Conv_Bn_Activation(3, 32, 3, 1, 'mish')
        # self.conv = DownSample1()
        # backbone
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()

        self.neck = Neck()

        self._init()

    def _init(self, ckpt_path=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = self.conv(x)
        # return x
        d1 = self.down1(x)  # [1, 64, 304, 304]
        d2 = self.down2(d1)  # [1, 128, 152, 152]
        d3 = self.down3(d2)  # [1, 256, 76, 76]
        d4 = self.down4(d3)  # [1, 512, 38, 38]
        d5 = self.down5(d4)  # [1, 1024, 19, 19]
        # return d3, d4, d5

        x20, x13, x6 = self.neck(d5, d4, d3)
        return x20, x13, x6


if __name__ == '__main__':
    init_seed()

    model = TinyNet()
    model.eval()

    import torch

    data = torch.randn(1, 3, 640, 640)
    print(data.reshape(-1)[:20])

    # output = model(data)
    # print(output.reshape(-1)[:20])
    x3, x4, x5 = model(data)
    print(x3.reshape(-1)[:20])
