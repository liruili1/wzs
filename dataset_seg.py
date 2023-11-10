import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
def read_split_data(txt_file: str):
    # 从txt文件中读取图像文件名
    with open(txt_file, 'r') as file:
        lines = file.read().splitlines()
    return lines
class NoduleDataset(Dataset):
    def __init__(self, image_paths, root_dir, mask_dir, transform=None,scale: float = 1.0):
        """
        Args:
            txt_file (string): Txt file with the images filenames.
            root_dir (string): Root directory with all the images.
            mask_dir (string): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.scale = scale



    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            if img.ndim == 2:
                mask = np.where(img == 255, 1, 0)
                mask = np.expand_dims(mask, axis=0)

            else:
                assert 'dimension more than one'

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        # 加载图像和掩码
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        mask = Image.open(mask_path)
        img = Image.open(img_path)
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)


        if self.transform:
            image = self.transform(img)
            mask = self.transform(mask)



        return torch.tensor(img,dtype=torch.float32),torch.tensor(mask,dtype=torch.float32)


if __name__ == '__main__':
    root_dir = r'C:\Users\lee\Desktop\wangzhengshuaitask\img2dseg'
    mask_dit = r'C:\Users\lee\Desktop\wangzhengshuaitask\mask2dseg'
    train_image_paths = read_split_data(r'C:\Users\lee\Desktop\wangzhengshuaitask\train_seg.txt')
    val_image_paths = read_split_data(r'C:\Users\lee\Desktop\wangzhengshuaitask\val_seg.txt')
    test_image_paths = read_split_data(r'C:\Users\lee\Desktop\wangzhengshuaitask\test_seg.txt')
    train_dataset = NoduleDataset(train_image_paths,root_dir=root_dir,mask_dir=mask_dit)
    val_dataset = NoduleDataset(val_image_paths,root_dir=root_dir,mask_dir=mask_dit)
    test_dataset = NoduleDataset(test_image_paths,root_dir=root_dir,mask_dir=mask_dit)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 遍历训练数据加载器
    for images, labels in train_dataloader:
        print(images.shape, labels.shape)
# 遍历数据加载器
