# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:09
@Author        : Tianxiaomo
@File          : dataset.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
import random
import sys

import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


def rand_uniform_strong(min, max):
    """
    随机均匀增强
    """
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min


def rand_scale(s):
    """
    随机缩放，放大或者缩小
    """
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


def rand_precalc_random(min, max, random_part):
    """
    也是随机操作，和rand_uniform_strong的差别在于初始值固定
    """
    if max < min:
        swap = min
        min = max
        max = swap
    return (random_part * (max - min)) + min


def fill_truth_detection(bboxes, num_boxes, classes, flip, dx, dy, sx, sy, net_w, net_h):
    """
    边界框操作
    1. 限制真值边界框数目
    2. 执行随机翻转
    3. 依据抖动结果转换边界框坐标
    4. 依据抖动后图像宽/高以及最终输入图像宽高缩放边界框坐标
    Args:
        bboxes: 真值边界框坐标，大小为[N, 5]，５分别表示x0, y0, x1, y1, cls_idx
        num_boxes: 边界框最大数目
        classes: 类别数
        flip: 是否翻转 0表示不翻转 1表示翻转
        dx:　抖动后的图像左上角坐标x
        dy: 抖动后的图像左上角坐标y
        sx: 抖动后的图像宽
        sy: 抖动后的图像高
        net_w: 结果图像宽
        net_h: 结果图像高

    Returns:

    """
    # 如果真值边界框数目为0，那么返回
    if bboxes.shape[0] == 0:
        return bboxes, 10000
    # 随机打乱真值边界框
    np.random.shuffle(bboxes)
    # 原始图像的边界框坐标基于抖动调整坐标系
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    # 设置x0, x1的最大最小值
    # 精度截断
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)
    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)

    # 设置y0，y1的最大最小值
    # 精度截断
    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    # 找出x0==x1，取值为0或者sx 或者y0==y1，取值为0或者xy的边界框
    # 也就是说，边界框经过抖动和截断后变成了一条线
    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    # 移除这种边界框
    for i in out_box:
        list_box.remove(i)
    # 获取剩余边界框
    bboxes = bboxes[list_box]

    # 如果边界框列表为空
    if bboxes.shape[0] == 0:
        return bboxes, 10000

    # 搜索符合类别标签数的边界框
    # 这一部分应该是为了防止数据集创建过程中出现的类别设置出错
    bboxes = bboxes[np.where((bboxes[:, 4] < classes) & (bboxes[:, 4] >= 0))[0]]

    # 最多使用num_boxes个真值边界框
    if bboxes.shape[0] > num_boxes:
        bboxes = bboxes[:num_boxes]

    # 计算边界框的最小宽或者高
    min_w_h = np.array([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]]).min()

    # 转换抖动图像上的边界框坐标到网络输入图像的边界框坐标
    bboxes[:, 0] *= (net_w / sx)
    bboxes[:, 2] *= (net_w / sx)
    bboxes[:, 1] *= (net_h / sy)
    bboxes[:, 3] *= (net_h / sy)

    if flip:
        # 图像进行水平翻转
        # 边界框坐标：[x0, y0, x1, y1]，宽高为w/h
        # 转换成为[w-x1, y0, w-x0, y1]
        temp = net_w - bboxes[:, 0]
        bboxes[:, 0] = net_w - bboxes[:, 2]
        bboxes[:, 2] = temp

    # 返回抖动以及缩放至目标大小之后的边界框列表以及最小宽或者高
    return bboxes, min_w_h


def rect_intersection(a, b):
    # 矩形框交集
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])

    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    return [minx, miny, maxx, maxy]


def image_data_augmentation(mat, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur,
                            truth):
    """
    数据增强，执行
    1. 图像抖动
    2. 颜色抖动
    3. 水平翻转
    4. 高斯噪声
    5. 图像模糊
    Args:
        mat: 输入图像数据
        w: 目标宽
        h: 目标高
        pleft: 抖动后图像左上角宽
        ptop: 抖动后图像左上角高
        swidth: 抖动后图像宽
        sheight:　抖动后图像高
        flip: 是否水平翻转
        dhue: 颜色抖动，色度
        dsat: 颜色抖动，饱和度
        dexp: 颜色抖动，xxx
        gaussian_noise: 是否执行高斯噪声
        blur: 是否执行图像模糊（也可以称之为图像滤波）
        truth: 结果图像的真值边界框坐标列表

    Returns:

    """
    try:
        img = mat
        # 原始图像宽高
        oh, ow, _ = img.shape
        # 抖动后图像左上角坐标以及宽高
        pleft, ptop, swidth, sheight = int(pleft), int(ptop), int(swidth), int(sheight)
        # crop
        src_rect = [pleft, ptop, swidth + pleft, sheight + ptop]  # x1,y1,x2,y2
        img_rect = [0, 0, ow, oh]
        new_src_rect = rect_intersection(src_rect, img_rect)  # 交集

        # 交集图像位于抖动图像中的边界框
        dst_rect = [max(0, -pleft), max(0, -ptop), max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]
        # cv2.Mat sized

        if (src_rect[0] == 0 and src_rect[1] == 0 and src_rect[2] == img.shape[0] and src_rect[3] == img.shape[1]):
            # 原图，没有进行抖动操作
            sized = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        else:
            # 创建抖动图像
            cropped = np.zeros([sheight, swidth, 3])
            # 填充部分像素大小为图像均值
            cropped[:, :, ] = np.mean(img, axis=(0, 1))

            # 赋值
            cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
                img[new_src_rect[1]:new_src_rect[3], new_src_rect[0]:new_src_rect[2]]

            # resize
            # 缩放到输入图像大小
            sized = cv2.resize(cropped, (w, h), cv2.INTER_LINEAR)

        # flip
        if flip:
            # 水平翻转
            # cv2.Mat cropped
            sized = cv2.flip(sized, 1)  # 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)

        # 颜色抖动，作者采用了OpenCV实现
        # 先转换到HSV颜色空间，然后手动调整饱和度、亮度和色度，最后转换成为RGB颜色空间
        # HSV augmentation
        # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
        if dsat != 1 or dexp != 1 or dhue != 0:
            if img.shape[2] >= 3:
                hsv_src = cv2.cvtColor(sized.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
                # Solve: https://github.com/Tianxiaomo/pytorch-YOLOv4/issues/427
                hsv = list(cv2.split(hsv_src))
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                sized = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
            else:
                sized *= dexp

        if blur:
            # 图像滤波
            if blur == 1:
                dst = cv2.GaussianBlur(sized, (17, 17), 0)
                # cv2.bilateralFilter(sized, dst, 17, 75, 75)
            else:
                ksize = (blur / 2) * 2 + 1
                dst = cv2.GaussianBlur(sized, (ksize, ksize), 0)

            if blur == 1:
                # blur　== 1，仅模糊背景
                # 从原始图像中获取未滤波真值边界框数据进行填充
                img_rect = [0, 0, sized.cols, sized.rows]
                for b in truth:
                    left = (b.x - b.w / 2.) * sized.shape[1]
                    width = b.w * sized.shape[1]
                    top = (b.y - b.h / 2.) * sized.shape[0]
                    height = b.h * sized.shape[0]
                    roi(left, top, width, height)
                    roi = roi & img_rect
                    dst[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]] = sized[roi[0]:roi[0] + roi[2],
                                                                          roi[1]:roi[1] + roi[3]]

            sized = dst

        if gaussian_noise:
            # 添加高斯噪声，也是手动实现的
            noise = np.array(sized.shape)
            gaussian_noise = min(gaussian_noise, 127)
            gaussian_noise = max(gaussian_noise, 0)
            cv2.randn(noise, 0, gaussian_noise)  # mean and variance
            sized = sized + noise
    except:
        print("OpenCV can't augment image: " + str(w) + " x " + str(h))
        sized = mat

    return sized


def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    bboxes[:, 0] += xd
    bboxes[:, 2] += xd
    bboxes[:, 1] += yd
    bboxes[:, 3] += yd

    return bboxes


def blend_truth_mosaic(out_img, img, bboxes, w, h, cut_x, cut_y, i_mixup,
                       left_shift, right_shift, top_shift, bot_shift):
    """
    mosic操作，调整图像数据以及真值边界框
    Args:
        out_img: 输出图像，填充为0
        img: 子图像，
        bboxes: 图像预处理后的边界框列表
        w: 结果图像宽，等同于out_img宽
        h: 结果图像高，等同于out_img高
        cut_x: 裁剪图像x
        cut_y: 裁剪图像y
        i_mixup: 第几幅图像
        left_shift:
        right_shift:
        top_shift:
        bot_shift:

    Returns:

    """
    left_shift = min(left_shift, w - cut_x)
    top_shift = min(top_shift, h - cut_y)
    right_shift = min(right_shift, cut_x)
    bot_shift = min(bot_shift, cut_y)

    if i_mixup == 0:
        # 左上角
        # 其大小为[dst_h, dst_w] = [cut_y, cut_x]，起始坐标为[y, x]=[0, 0]
        bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
    if i_mixup == 1:
        # 右上角
        # 其大小为[dst_h, dst_w] = [h - cut_y, w - cut_x]，起始坐标为[y, x]=[0, cut_x]
        bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0)
        # out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]
        out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]
    if i_mixup == 2:
        # 左下角
        # 其大小为[dst_h, dst_w] = [h - cut_y, cut_x]，起始坐标为[y, x]=[cut_y, 0]
        bboxes = filter_truth(bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y)
        out_img[cut_y:, :cut_x] = img[cut_y - bot_shift:h - bot_shift, left_shift:left_shift + cut_x]
    if i_mixup == 3:
        # 右下角
        # 其大小为[dst_h, dst_w] = [h - cut_y, h - cut_x]，起始坐标为[y, x]=[cut_y, cut_x]
        bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bot_shift, w - cut_x, h - cut_y, cut_x, cut_y)
        out_img[cut_y:, cut_x:] = img[cut_y - bot_shift:h - bot_shift, cut_x - right_shift:w - right_shift]

    return out_img, bboxes


def draw_box(img, bboxes):
    for b in bboxes:
        img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return img


class Yolo_dataset(Dataset):
    """
    YOLO数据集类，集成了数据加载和数据转换工作
    """

    def __init__(self, label_path, cfg, train=True):
        super(Yolo_dataset, self).__init__()

        # mixup == 2 执行mosaic操作
        if cfg.mixup == 2:
            print("cutmix=1 - isn't supported for Detector")
            raise
        elif cfg.mixup == 2 and cfg.letter_box:
            print("Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters")
            raise

        self.cfg = cfg
        self.train = train

        # 按行读取标签文件，然后解析出图像路径以及图像对应的标注框和类别下标
        truth = {}
        f = open(label_path, 'r', encoding='utf-8')
        for line in f.readlines():
            # 按行读取，图像名为key，逐个添加bbox
            data = line.split(" ")
            truth[data[0]] = []
            for i in data[1:]:
                truth[data[0]].append([int(float(j)) for j in i.split(',')])

        self.truth = truth
        self.imgs = list(self.truth.keys())

    def __len__(self):
        return len(self.truth.keys())

    def __getitem__(self, index):
        if not self.train:
            # 推理状态下，调用其他函数
            # 推理状态下的预处理函数和训练阶段有差别
            return self._get_val_item(index)
        img_path = self.imgs[index]
        bboxes = np.array(self.truth.get(img_path), dtype=np.float)
        img_path = os.path.join(self.cfg.dataset_dir, img_path)
        use_mixup = self.cfg.mixup
        if random.randint(0, 1):
            # 是否执行mixup，50%概率
            use_mixup = 0

        if use_mixup == 3:
            # only mosaic
            min_offset = 0.2
            # 进行随机裁剪，随机生成裁剪图像的起始坐标
            # 坐标x取值在[0.2*w, 0.8*w]之间，坐标y同理
            cut_x = random.randint(int(self.cfg.w * min_offset), int(self.cfg.w * (1 - min_offset)))
            cut_y = random.randint(int(self.cfg.h * min_offset), int(self.cfg.h * (1 - min_offset)))

        r1, r2, r3, r4, r_scale = 0, 0, 0, 0, 0
        dhue, dsat, dexp, flip, blur = 0, 0, 0, 0, 0
        gaussian_noise = 0

        # 指定结果图像的宽/高
        out_img = np.zeros([self.cfg.h, self.cfg.w, 3])
        # 在输出图像上的真值边界框可以有多个
        out_bboxes = []

        # 如果use_mixup==0，表示不进行mosaic操作
        # 否则use_mixup==3，就是进行mosaic操作，从下面的实现中可以发现，每张图像都会进行预处理操作，包括
        # 空间抖动、颜色抖动、随机翻转、随机模糊、随机噪声、随机缩放
        for i in range(use_mixup + 1):
            # print(f"{use_mixup} {i}")
            if i != 0:
                # i == 0 使用原图
                # i != 0 随机选择一张图像，参与拼接
                img_path = random.choice(list(self.truth.keys()))
                bboxes = np.array(self.truth.get(img_path), dtype=np.float)
                img_path = os.path.join(self.cfg.dataset_dir, img_path)
            # 读取图像，转换成RGB格式
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                continue
            # 原始图像高/宽/通道数
            oh, ow, oc = img.shape
            # 图像抖动
            dh, dw, dc = np.array(np.array([oh, ow, oc]) * self.cfg.jitter, dtype=np.int)

            # 色度、饱和度、还有一个什么东东
            dhue = rand_uniform_strong(-self.cfg.hue, self.cfg.hue)
            dsat = rand_scale(self.cfg.saturation)
            dexp = rand_scale(self.cfg.exposure)

            # 左右填充？？？
            # 填充值可以是正数（增加图像面积），也可以是负数（减少图像面积）
            pleft = random.randint(-dw, dw)
            pright = random.randint(-dw, dw)
            ptop = random.randint(-dh, dh)
            pbot = random.randint(-dh, dh)

            # 随机翻转 50%概率
            flip = random.randint(0, 1) if self.cfg.flip else 0

            if (self.cfg.blur):
                # 随机模糊
                # 0表示不进行模糊操作
                # 1表示模糊背景
                # 2表示模糊整幅图像
                tmp_blur = random.randint(0, 2)  # 0 - disable, 1 - blur background, 2 - blur the whole image
                if tmp_blur == 0:
                    blur = 0
                elif tmp_blur == 1:
                    blur = 1
                else:
                    blur = self.cfg.blur

            # 是否执行高斯噪声 以及 50%概率
            if self.cfg.gaussian and random.randint(0, 1):
                gaussian_noise = self.cfg.gaussian
            else:
                gaussian_noise = 0

            if self.cfg.letter_box:
                # 原始图像的宽高比
                img_ar = ow / oh
                # 结果图像的宽高比
                net_ar = self.cfg.w / self.cfg.h
                # 计算原始图像宽高比和结果图像宽高比的比例
                result_ar = img_ar / net_ar
                # print(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
                if result_ar > 1:  # sheight - should be increased
                    # 这种情况下，原始图像的宽高比大于结果图像的宽高比
                    oh_tmp = ow / net_ar
                    delta_h = (oh_tmp - oh) / 2
                    ptop = ptop - delta_h
                    pbot = pbot - delta_h
                    # print(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
                else:  # swidth - should be increased
                    ow_tmp = oh * net_ar
                    delta_w = (ow_tmp - ow) / 2
                    pleft = pleft - delta_w
                    pright = pright - delta_w
                    # printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);

            # 填充后的宽高
            swidth = ow - pleft - pright
            sheight = oh - ptop - pbot

            # 关键操作一：fill_truth_detection 依据预处理操作设置边界框
            truth, min_w_h = fill_truth_detection(bboxes, self.cfg.boxes, self.cfg.classes, flip, pleft, ptop, swidth,
                                                  sheight, self.cfg.w, self.cfg.h)
            if (min_w_h / 8) < blur and blur > 1:  # disable blur if one of the objects is too small
                # 调整模糊值
                blur = min_w_h / 8

            # 关键操作二：image_data_augmentation 针对图像进行增强操作
            ai = image_data_augmentation(img, self.cfg.w, self.cfg.h, pleft, ptop, swidth, sheight, flip,
                                         dhue, dsat, dexp, gaussian_noise, blur, truth)

            # 相当于标签框处理和图像数据处理分开处理
            # 图像预处理大体可以划分为两部分，一个是空间操作，另一个是颜色操作。对于边界框而言，只有空间操作会影响。

            if use_mixup == 0:
                # 不使用mixup，那么增强图像就是结果图像
                out_img = ai
                out_bboxes = truth
            if use_mixup == 1:
                # 进行mixup
                if i == 0:
                    # 第一幅图像，就是锚点图像，作为其他图像的背景
                    old_img = ai.copy()
                    old_truth = truth.copy()
                elif i == 1:
                    # 执行mixup
                    out_img = cv2.addWeighted(ai, 0.5, old_img, 0.5)
                    out_bboxes = np.concatenate([old_truth, truth], axis=0)
            elif use_mixup == 3:
                # 执行mosaic
                if flip:
                    tmp = pleft
                    pleft = pright
                    pright = tmp

                left_shift = int(min(cut_x, max(0, (-int(pleft) * self.cfg.w / swidth))))
                top_shift = int(min(cut_y, max(0, (-int(ptop) * self.cfg.h / sheight))))

                right_shift = int(min((self.cfg.w - cut_x), max(0, (-int(pright) * self.cfg.w / swidth))))
                bot_shift = int(min(self.cfg.h - cut_y, max(0, (-int(pbot) * self.cfg.h / sheight))))

                # 如果使用mosaic，每个子图像保留一个bbox
                out_img, out_bbox = blend_truth_mosaic(out_img, ai, truth.copy(), self.cfg.w, self.cfg.h, cut_x,
                                                       cut_y, i, left_shift, right_shift, top_shift, bot_shift)
                out_bboxes.append(out_bbox)
                # print(img_path)
        if use_mixup == 3:
            # 针对mosaic操作，保留子图像的bbox
            out_bboxes = np.concatenate(out_bboxes, axis=0)
        # self.cfg.boxes默认为60，表示每幅图像单次最多采用60个真值框（目标类别+背景类别）参与训练
        # 针对某些图像并没有如此多的真值标签框，那么设置为0值
        out_bboxes1 = np.zeros([self.cfg.boxes, 5])
        # 截取out_bboxes前min(out_bboxes.shape[0], self.cfg.boxes)作为真值框参与训练
        out_bboxes1[:min(out_bboxes.shape[0], self.cfg.boxes)] = out_bboxes[:min(out_bboxes.shape[0], self.cfg.boxes)]
        return out_img, out_bboxes1

    def _get_val_item(self, index):
        """
        """
        img_path = self.imgs[index]
        bboxes_with_cls_id = np.array(self.truth.get(img_path), dtype=np.float)
        # 读取图像
        img = cv2.imread(os.path.join(self.cfg.dataset_dir, img_path))
        # img_height, img_width = img.shape[:2]
        # 转换到RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.cfg.w, self.cfg.h))
        # img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        # 计算真值边界框个数
        num_objs = len(bboxes_with_cls_id)
        target = {}
        # boxes to coco format
        # 转换成COCO格式保存
        boxes = bboxes_with_cls_id[..., :4]
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(bboxes_with_cls_id[..., -1].flatten(), dtype=torch.int64)
        target['image_id'] = torch.tensor([get_image_id(img_path)])
        target['area'] = (target['boxes'][:, 3]) * (target['boxes'][:, 2])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        return img, target


def get_image_id(filename: str) -> int:
    """
    Convert a string to a integer.
    Make sure that the images and the `image_id`s are in one-one correspondence.
    There are already `image_id`s in annotations of the COCO dataset,
    in which case this function is unnecessary.
    For creating one's own `get_image_id` function, one can refer to
    https://github.com/google/automl/blob/master/efficientdet/dataset/create_pascal_tfrecord.py#L86
    or refer to the following code (where the filenames are like 'level1_123.jpg')
    >>> lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    >>> lv = lv.replace("level", "")
    >>> no = f"{int(no):04d}"
    >>> return int(lv+no)
    """
    # raise NotImplementedError("Create your own 'get_image_id' function")
    # lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    # lv = lv.replace("level", "")
    # no = f"{int(no):04d}"
    # return int(lv+no)

    print("You could also create your own 'get_image_id' function.")
    # print(filename)
    parts = filename.split('/')
    id = int(parts[-1][0:-4])
    # print(id)
    return id


if __name__ == "__main__":
    from cfg import Cfg
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    # 固定随机数种子
    random.seed(2020)
    np.random.seed(2020)
    # Cfg.dataset_dir = '/mnt/e/Dataset'
    # 数据集根路径
    Cfg.dataset_dir = './'
    # 创建YOLO数据集类
    # 默认标签文件路径为/path/to/data/train.txt
    dataset = Yolo_dataset(Cfg.train_label, Cfg)
    # dataset = Yolo_dataset(Cfg.val_label, Cfg, train=False)
    # for i in tqdm(range(100)):
    for i in range(100):
        out_img, out_bboxes = dataset.__getitem__(i)
        a = draw_box(out_img.copy(), out_bboxes.astype(np.int32))
        # plt.imshow(a.astype(np.int32))
        # plt.show()
        cv2.imwrite(f"./data/img_{i}.jpg", a.astype(np.uint8))
