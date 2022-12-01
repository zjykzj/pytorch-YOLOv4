# -*- coding: utf-8 -*-

"""
@date: 2022/12/1 下午2:46
@file: maxpooldark.py
@author: zj
@description: 
"""

import torch

from tool.darknet2pytorch import MaxPoolDark

if __name__ == '__main__':
    m = MaxPoolDark(size=2, stride=1)

    data = torch.randn(1, 3, 224, 224)
    res = m(data)
    print(res.shape)
