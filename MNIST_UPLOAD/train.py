# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 1. 定義 CNN 模型架構
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 輸入: 1x28x28 (灰度圖)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 輸出: 16x28x28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 輸出: 16x14x14

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 輸出: 32x14x14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 輸出: 32x7x7

        # 將 32x7x7 的特徵圖展平
        self.fc1 = nn.Linear(32 * 7 * 7, 10)  # 10 代表 0-9 十個類別

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # 展平
        x = self.fc1(x)
        return x


def train_and_save_model():
    print("開始訓練...")

    # 2. 數據加載與轉換
    transform = transforms.Compose([
        transforms.ToTensor(),  # 轉換為 Tensor
        transforms.Normalize((0.5,), (0.5,))  # 正規化
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 3. 初始化模型、損失函數、優化器
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 訓練循環
    num_epochs = 3  # 為了快速演示，只訓練 3 個 epoch
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 前向傳播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向傳播與優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 5. 保存模型權重
    model_path = "mnist_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"訓練完成！模型已保存至 {model_path}")


if __name__ == "__main__":
    train_and_save_model()