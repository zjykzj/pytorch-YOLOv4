# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/08 11:45
@Author        : Tianxiaomo
@File          : coco_annotatin.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import json
from collections import defaultdict
from tqdm import tqdm
import os

"""hyper parameters"""
# json_file_path = 'E:/Dataset/mscoco2017/annotations/instances_train2017.json'
# images_dir_path = 'mscoco2017/train2017/'
# output_path = '../data/val.txt'
# json_file_path = './coco/annotations/instances_train2017.json'
# images_dir_path = './coco/images/train2017/'
# output_path = './data/train.txt'
json_file_path = './coco/annotations/instances_val2017.json'
images_dir_path = './coco/images/val2017/'
output_path = './data/val.txt'

"""load json file"""
name_box_id = defaultdict(list)
id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)

"""generate labels"""
# 获取图像列表，每个图像item包含了图像名、图像id以及宽/高等信息
images = data['images']
# 获取标注列表，每个标注item包含了类别下标、标注id以及标注框坐标列表等标注信息
annotations = data['annotations']
# 遍历每个标注item
for ant in tqdm(annotations):
    # 获取图像id
    id = ant['image_id']
    # name = os.path.join(images_dir_path, images[id]['file_name'])
    # 生成对应图像路径
    name = os.path.join(images_dir_path, '{:012d}.jpg'.format(id))
    # 获取类别下标
    cat = ant['category_id']

    # 类别下标转换，这一段应该是默认的操作
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11

    # 载入字典，图像路径: [真值边界框列表，类别id]
    name_box_id[name].append([ant['bbox'], cat])

"""write to txt"""
# 将图像路径以及标注信息按行写入txt文件
with open(output_path, 'w') as f:
    for key in tqdm(name_box_id.keys()):
        # 首先写入图像路径
        f.write(key)
        # 然后获取标注信息
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
