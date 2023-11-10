from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset_seg import NoduleDataset,read_split_data

import torch
from torch.utils.data import DataLoader

from torchvision import models, transforms
from seg_net import cswinunet

num_epochs = 10
learning_rate = 1e-3
batch_size = 16
best_val_acc = 0.0  # 最佳验证准确度初始化
model_save_path = r'C:\Users\lee\Desktop\model_seg.pth'  # 模型保存路径

root_dir = r'C:\Users\lee\Desktop\wangzhengshuaitask\img2dseg'
mask_dit = r'C:\Users\lee\Desktop\wangzhengshuaitask\mask2dseg'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
train_image_paths = read_split_data(r'C:\Users\lee\Desktop\wangzhengshuaitask\train.txt')
val_image_paths = read_split_data(r'C:\Users\lee\Desktop\wangzhengshuaitask\val.txt')

train_dataset = NoduleDataset(train_image_paths, root_dir=root_dir, mask_dir=mask_dit)
val_dataset = NoduleDataset(val_image_paths, root_dir=root_dir, mask_dir=mask_dit)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        # 确保网络输出在 [0, 1] 范围内
        assert output.size() == target.size(),'channel is not equal'
        output = torch.sigmoid(output)

        # 计算交集
        intersection = (output * target).sum()

        # 计算并集
        union = output.sum() + target.sum() - intersection + self.eps

        # 计算IoU
        iou = intersection / union

        # 返回IoU的补集作为损失
        return 1 - iou

# 定义模型
model = cswinunet().to(device)
criterion = IoULoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
best_val_dice = 0.0
def dice_coefficient(preds, targets):
    smooth = 1e-6
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        # 确保网络输出在 [0, 1] 范围内
        assert output.size() == target.size(),'channel is not equal'
        output = torch.sigmoid(output)

        # 计算交集
        intersection = (output * target).sum()

        # 计算并集
        union = output.sum() + target.sum() - intersection + self.eps

        # 计算IoU
        iou = intersection / union

        # 返回IoU的补集作为损失
        return 1 - iou

for epoch in range(num_epochs):
    model.train()  # 训练模式
    train_loss = 0.0
    train_correct = 0
    train_dice = 0.0

    for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Train', unit='batch'):

        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, masks)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        preds = torch.sigmoid(outputs) > 0.5  # 假设是二分类问题
        train_dice += dice_coefficient(preds, masks).item() * imgs.size(0)

        # 计算平均损失和Dice系数
    train_loss = train_loss / len(train_loader.dataset)
    train_dice = train_dice / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Val', unit='batch'):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * imgs.size(0)
            preds = torch.sigmoid(outputs) > 0.5  # 假设是二分类问题
            val_dice += dice_coefficient(preds.float(), masks.float()).item() * imgs.size(0)

    val_loss = val_loss / len(val_loader.dataset)
    val_dice = val_dice / len(val_loader.dataset)

    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train loss: {train_loss:.4f}, Train Dice Coeff: {train_dice:.4f}')
    print(f'Val loss: {val_loss:.4f}, Val Dice Coeff: {val_dice:.4f}')

    # 如果当前验证Dice得分高于之前的最好得分，则保存模型
    if val_dice > best_val_dice:
        print(f'Validation Dice improved from {best_val_dice:.4f} to {val_dice:.4f}. Saving model...')
        best_val_dice = val_dice
        torch.save(model.state_dict(), 'best_model_dice.pth')
