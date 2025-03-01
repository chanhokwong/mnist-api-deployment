import torch
from model import *
from dataset import *

def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


# 初始化模型並加載權重
# model = models.resnet18(weights=None)  # 或使用預訓練
# model.fc = nn.Linear(model.fc.in_features, 102)
model = torch.load("flower_classifier/model.pt")
# model.load_state_dict(torch.load("flower_classifier.pth"))
test(model, test_loader)