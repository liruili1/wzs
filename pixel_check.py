import os
from PIL import Image
from collections import defaultdict

# 设置包含图片的文件夹路径
images_folder_path = r'C:\Users\lee\Desktop\wangzhengshuaitask\mask2d'  # 例如 '/home/user/images'

# 用于存储所有图片的像素值分布的字典
pixel_values_distribution = defaultdict(int)

# 遍历文件夹中的所有文件
for file_name in os.listdir(images_folder_path):
    # 构建完整的文件路径
    file_path = os.path.join(images_folder_path, file_name)

    # 确保是文件并且是图片
    if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 打开图片
        with Image.open(file_path) as img:
            # 将图片转换为灰度（如果需要）
            img = img.convert('L')

            # 获取图片中的所有像素值
            pixels = list(img.getdata())

            # 更新像素值分布字典
            for pixel in pixels:
                pixel_values_distribution[pixel] += 1

# 打印像素值分布情况
for pixel_value, count in pixel_values_distribution.items():
    print(f"Pixel value {pixel_value} occurs {count} times.")

print("Finished analyzing the pixel values.")
