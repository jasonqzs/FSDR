#dataset resample result
# """
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
import torch.nn.functional as F
from collections import Counter


def get_img_num_per_cls(cls_num, imb_type, imb_factor):
    img_max = 8750 / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

per_cls = get_img_num_per_cls(25, imb_type='exp', imb_factor=0.005)
print(per_cls)
total_sum = sum(per_cls)
print(total_sum)

#imb_factor=0.02
#[350, 299, 255, 218, 187, 160, 136, 117, 100, 85, 73, 62, 53, 45, 39, 33, 28, 24, 20, 17, 15, 13, 11, 9, 8, 7]
# """
# import os
# from PIL import Image
#
# # 源文件夹路径
# source_folder = '/data/ch/tmp/pycharm_project_191/fsdr/malevis/malevis_dataset/val_original'
# # 目标文件夹路径
# target_folder = '/data/ch/tmp/pycharm_project_191/fsdr/malevis/malevis_dataset/val'
#
# # 遍历每个类别的文件夹
# for category_folder in os.listdir(source_folder):
#     category_path = os.path.join(source_folder, category_folder)
#     if os.path.isdir(category_path):
#         # 遍历类别文件夹里的图像文件
#         for image_file in os.listdir(category_path):
#             image_path = os.path.join(category_path, image_file)
#             # 读取图像
#             image = Image.open(image_path)
#             # 在新的文件夹中创建对应的类别文件夹
#             target_category_path = os.path.join(target_folder, category_folder)
#             os.makedirs(target_category_path, exist_ok=True)
#             # 保存图像到新的文件夹
#             image.save(os.path.join(target_category_path, image_file))
#
# print('图像数据集已保存到新文件夹！')
