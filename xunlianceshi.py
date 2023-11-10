import os
import numpy as np
from sklearn.model_selection import train_test_split

def split_data_set(image_dir, output_dir, train_size=0.6, val_size=0.2):
    # 获取所有图像文件名
    all_images = [img for img in os.listdir(image_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 划分训练集、验证集和测试集
    train_val_images, test_images = train_test_split(all_images, train_size=train_size + val_size, random_state=42)
    train_images, val_images = train_test_split(train_val_images, train_size=train_size/(train_size + val_size), random_state=42)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将数据集写入txt文件
    with open(os.path.join(output_dir, 'train.txt'), 'w') as file:
        file.writelines(f"{image}\n" for image in train_images)

    with open(os.path.join(output_dir, 'val.txt'), 'w') as file:
        file.writelines(f"{image}\n" for image in val_images)

    with open(os.path.join(output_dir, 'test.txt'), 'w') as file:
        file.writelines(f"{image}\n" for image in test_images)

# 假设所有图像都在'image_dir'目录下
image_dir = 'path_to_your_image_directory'
output_dir = 'path_to_your_output_directory'  # 设置你希望保存txt文件的路径
split_data_set(image_dir, output_dir)
