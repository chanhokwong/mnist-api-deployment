import torch
from dataset import test_loader


def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _,predicted = torch.max(outputs.data, 1)
            correct += (predicted==labels).sum().item()
            total += labels.size(0)
        print(f"Accuracy on test set: {100 * correct / total:.2f}")

model = torch.load("model.pt")
test(model,test_loader)