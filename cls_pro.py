import os
from PIL import Image
import shutil

# 设置img2d和mask2d的文件夹路径
img2d_folder_path = r'C:\Users\lee\Desktop\wangzhengshuaitask\img2d'  # 替换为你的img2d文件夹路径
mask2d_folder_path = r'C:\Users\lee\Desktop\wangzhengshuaitask\mask2d'  # 替换为你的mask2d文件夹路径

# 设置新的文件夹路径来存储修改后的文件
new_img2d_folder_path = r'C:\Users\lee\Desktop\wangzhengshuaitask\cls_img2d'   # 替换为新的img2d文件夹路径
new_mask2d_folder_path = r'C:\Users\lee\Desktop\wangzhengshuaitask\cls_mask2d'    # 替换为新的mask2d文件夹路径




# 创建新的文件夹
os.makedirs(new_img2d_folder_path, exist_ok=True)
os.makedirs(new_mask2d_folder_path, exist_ok=True)

# 遍历mask2d文件夹中的所有文件
for mask_file in os.listdir(mask2d_folder_path):
    mask_file_path = os.path.join(mask2d_folder_path, mask_file)

    # 确保是文件并且是图片
    if os.path.isfile(mask_file_path) and mask_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 打开mask图片
        with Image.open(mask_file_path) as img:
            # 将图片转换为灰度（如果它不是）
            img = img.convert('L')

            # 检查图片是否包含大于0的像素值
            prefix = '1_' if any(pixel > 0 for pixel in img.getdata()) else '0_'

            # 构建新的文件名
            new_mask_file_name = prefix + mask_file
            new_mask_file_path = os.path.join(new_mask2d_folder_path, new_mask_file_name)

            # 复制并重命名mask文件到新的文件夹
            shutil.copy2(mask_file_path, new_mask_file_path)

            # 现在处理img2d文件夹中的对应文件
            img_file_path = os.path.join(img2d_folder_path, mask_file)
            new_img_file_name = prefix + mask_file
            new_img_file_path = os.path.join(new_img2d_folder_path, new_img_file_name)

            # 如果存在对应文件，则复制到新的文件夹
            if os.path.exists(img_file_path):
                shutil.copy2(img_file_path, new_img_file_path)
            else:
                print(f"Warning: {mask_file} has no corresponding image in img2d folder.")

print("Finished copying and renaming mask and image files.")
