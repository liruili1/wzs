import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

def read_split_data(txt_file: str):
    # 从txt文件中读取图像文件名
    with open(txt_file, 'r') as file:
        lines = file.read().splitlines()
    return lines
class NoduleDataset(Dataset):
    def __init__(self, image_paths,root_dir, transform=True):
        """
        Args:
            image_paths (list): List of paths to images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image or numpy ndarray to tensor.
        ]) if transform else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = 1 if os.path.basename(img_path).startswith('1_') else 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)
if __name__ == '__main__':
    root_dir = r'C:\Users\lee\Desktop\wangzhengshuaitask\cls_img2d'
    train_image_paths = read_split_data(r'C:\Users\lee\Desktop\wangzhengshuaitask\train.txt')
    val_image_paths = read_split_data(r'C:\Users\lee\Desktop\wangzhengshuaitask\val.txt')
    test_image_paths = read_split_data(r'C:\Users\lee\Desktop\wangzhengshuaitask\test.txt')

    # 创建数据集
    train_dataset = NoduleDataset(train_image_paths,root_dir=root_dir)
    val_dataset = NoduleDataset(val_image_paths,root_dir=root_dir)
    test_dataset = NoduleDataset(test_image_paths,root_dir=root_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 遍历训练数据加载器
    for images, labels in train_dataloader:
        print(images.shape, labels.shape)
