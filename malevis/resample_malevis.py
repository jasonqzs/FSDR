#resample malevis dataset的训练集
#考虑到：test集中other类的数量偏大，所以resample后的train集中，other类别的样本最多-350
import os
import sys
import random
import shutil
import numpy as np

#保证每次生成的不平衡数据集是一致的
np.random.seed(42)
random.seed(42)
#输出保存到本地
# class Logger(object):
#     def __init__(self, filename='default.log', stream=sys.stdout):
#         self.terminal = stream
#         self.log = open(filename, 'w')
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass
#
# current_directory = os.path.dirname(__file__)
# root_1 = current_directory + "/log/resample_malevis.log"
# root_2 = current_directory + "/log/resample_malevis.log_file"
#
# sys.stdout = Logger(root_1, sys.stdout)
# sys.stderr = Logger(root_2, sys.stderr)

# 数据集根目录路径
current_directory = os.path.dirname(__file__)
dataset_root = current_directory + "/malevis_dataset/train"

# 子文件夹列表，每个子文件夹表示一个类别
class_folders = sorted(os.listdir(dataset_root))

# 每个类别需要读取的数据数量列表
# data_counts = [350, 299, 255, 218, 187, 160, 136, 117, 100, 85, 73, 62, 53, 45, 39, 33, 28, 24, 20, 17, 15, 13, 11, 9, 8, 7] #imb_factor = 0.02 sum=2364
# data_counts = [350, 291, 242, 201, 167, 139, 115, 96, 80, 66, 55, 46, 38, 31, 26, 22, 18, 15, 12, 10, 8, 7, 6, 5, 4, 3] #imb_factor = 0.01
# data_counts = [350, 319, 291, 265, 242, 220, 201, 183, 167, 152, 139, 127, 115, 105, 96, 87, 80, 73, 66, 60, 55, 50, 46, 42, 38, 35] #imb_factor = 0.1 sum = 3604
# data_counts = [350, 310, 275, 244, 216, 192, 170, 151, 134, 119, 105, 93, 83, 73, 65, 58, 51, 45, 40, 35, 31, 28, 25, 22, 19, 17] #imb_factor = 0.05 sum=2951
# data_counts = [350, 283, 229, 185, 149, 121, 98, 79, 64, 51, 42, 34, 27, 22, 18, 14, 11, 9, 7, 6, 5, 4, 3, 2, 2, 1] #imb_factor=0.005 sum=1816

#class 25
# data_counts = [350, 297, 252, 214, 182, 154, 131, 111, 95, 80, 68, 58, 49, 42, 35, 30, 25, 21, 18, 15, 13, 11, 9, 8, 7] #2275 imb=50
# data_counts = [350, 308, 272, 240, 212, 187, 165, 146, 128, 113, 100, 88, 78, 69, 60, 53, 47, 41, 37, 32, 28, 25, 22, 19, 17] #2837 imb=20
# data_counts = [350, 288, 238, 196, 162, 134, 110, 91, 75, 62, 51, 42, 35, 28, 23, 19, 16, 13, 11, 9, 7, 6, 5, 4, 3] #1978
data_counts = [350, 317, 288, 262, 238, 216, 196, 178, 162, 147, 134, 121, 110, 100, 91, 82, 75, 68, 62, 56, 51, 46, 42, 38, 35] #3465 imb=10
# data_counts = [350, 280, 225, 180, 144, 116, 93, 74, 59, 47, 38, 30, 24, 19, 15, 12, 10, 8, 6, 5, 4, 3, 2, 2, 1] #1747 imb=200

# 保存数据集的文件夹路径
output_dir = current_directory + "/malevis_dataset/train_10"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个子文件夹
for i, class_folder in enumerate(class_folders):
    class_name = os.path.basename(class_folder)
    class_path = os.path.join(dataset_root, class_folder)

    # 获取该类别下的所有图像文件路径
    image_files = [os.path.join(class_path, img) for img in os.listdir(class_path)]

    # 获取当前类别需要读取的数据数量
    data_count = data_counts[i]

    # 如果数据数量大于实际可用的数据数量，则取实际可用的数据数量
    if data_count > len(image_files):
        data_count = len(image_files)

    # 从图像文件列表中随机选择相应数量的数据
    selected_files = random.sample(image_files, data_count)

    # 将选中的数据复制到输出文件夹中
    output_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for file in selected_files:
        shutil.copy(file, output_class_dir)

    # print(f"Class {class_name}: Selected {data_count} samples and saved to {output_class_dir}")