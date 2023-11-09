import os
import shutil

# 设置主文件夹路径，这个文件夹包含了所有的lung_XXX文件夹
main_folder_path = r'C:\Users\lee\Desktop\wangzhengshuaitask\masktr'  # 例如 '/home/user/lung_data'

# 设置目标文件夹路径，所有重命名后的文件都会被移动到这个文件夹
target_folder_path = r'C:\Users\lee\Desktop\wangzhengshuaitask\mask2d'   # 例如 '/home/user/all_slices'

# 如果目标文件夹不存在，创建它
if not os.path.exists(target_folder_path):
    os.makedirs(target_folder_path)

# 遍历主文件夹中的所有子文件夹
for folder_name in os.listdir(main_folder_path):
    # 获取子文件夹的完整路径
    folder_path = os.path.join(main_folder_path, folder_name)

    # 确保它是一个文件夹
    if os.path.isdir(folder_path):
        # 遍历文件夹中的所有文件
        for file_name in os.listdir(folder_path):
            # 仅处理文件名以'slice_'开头的文件
            if file_name.startswith('slice_'):
                # 创建新的文件名，格式为：lung_XXX_slice_XXX
                new_file_name = f"{folder_name}_{file_name}"

                # 获取原文件的完整路径
                old_file_path = os.path.join(folder_path, file_name)

                # 设置新文件的完整路径
                new_file_path = os.path.join(target_folder_path, new_file_name)

                # 重命名并移动文件到目标文件夹
                shutil.move(old_file_path, new_file_path)

print("Files have been renamed and moved successfully.")
