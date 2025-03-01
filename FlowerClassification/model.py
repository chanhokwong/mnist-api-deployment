import torch.nn as nn
import torch.optim as optim
from torchvision import models

# 使用預訓練模型
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 凍結所有卷積層（可選）
# for param in model.parameters():
#     param.requires_grad = False

# 修改最後的全連接層
num_classes = 102
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


