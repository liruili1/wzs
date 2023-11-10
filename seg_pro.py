import os
from PIL import Image
import numpy as np
import shutil

# 指定图像和掩膜文件夹的路径
source_img_folder = r'C:\Users\lee\Desktop\wangzhengshuaitask\img2d'
source_mask_folder = r'C:\Users\lee\Desktop\wangzhengshuaitask\mask2d'
output_img_folder = r'C:\Users\lee\Desktop\wangzhengshuaitask\img2dseg'
output_mask_folder = r'C:\Users\lee\Desktop\wangzhengshuaitask\mask2dseg'

# 创建输出文件夹
os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# 获取掩膜文件夹中的所有文件名
mask_files = [f for f in os.listdir(source_mask_folder) if os.path.isfile(os.path.join(source_mask_folder, f))]

# 遍历掩膜文件，检查是否包含像素值
for mask_file in mask_files:
    mask_path = os.path.join(source_mask_folder, mask_file)
    img_path = os.path.join(source_img_folder, mask_file)

    # 确保对应的图像文件存在
    if os.path.exists(img_path):
        # 读取掩膜文件
        mask = Image.open(mask_path)
        # 转换为numpy数组并检查是否有非零像素
        mask_array = np.array(mask)
        if np.any(mask_array > 0):  # 假设像素值大于0的才是有效掩膜
            # 复制图像和掩膜文件到输出文件夹
            shutil.copy(img_path, os.path.join(output_img_folder, mask_file))
            shutil.copy(mask_path, os.path.join(output_mask_folder, mask_file))

print(f"图像和对应的掩膜已经被复制到 '{output_img_folder}' 和 '{output_mask_folder}'")
