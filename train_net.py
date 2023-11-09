import torch
from torch.utils.data import DataLoader
from dataset import NoduleDataset, read_split_data  # 确保 read_data_paths 在 dataset.py 中定义
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 设定参数
num_epochs = 10
learning_rate = 1e-3
batch_size = 32
best_val_acc = 0.0  # 最佳验证准确度初始化
model_save_path = r'C:\Users\lee\Desktop\model.pth'  # 模型保存路径
root_dir = r'C:\Users\lee\Desktop\wangzhengshuaitask\cls_img2d'

# 数据转换定义
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 读取数据集路径
train_image_paths = read_split_data(r'C:\Users\lee\Desktop\wangzhengshuaitask\train.txt')
val_image_paths = read_split_data(r'C:\Users\lee\Desktop\wangzhengshuaitask\val.txt')

# 用完整路径创建数据集
train_dataset = NoduleDataset(train_image_paths,root_dir=root_dir)
val_dataset = NoduleDataset(val_image_paths,root_dir=root_dir)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # 修改为两类输出
model = model.to(device)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练和验证循环
for epoch in range(num_epochs):
    model.train()  # 训练模式
    train_loss = 0.0
    train_correct = 0

    # 训练
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Train', unit='batch'):
        labels = labels.unsqueeze(1)
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)


    # 验证
    model.eval()  # 验证模式
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Val', unit='batch'):
            labels = labels.unsqueeze(1)
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            # 应用sigmoid激活函数来将logits转换为概率
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()  # 使用0.5的阈值来确定预测类别

            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    # 打印训练和验证的损失及准确率
    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 如果验证准确率高于之前的最佳准确率，则保存模型
    if val_acc > best_val_acc:
        print(f'Saving the model with val acc {val_acc:.4f}')
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)

print('Finished Training')
