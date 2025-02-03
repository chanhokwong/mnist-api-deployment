import torch
from dataset import test_loader

def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    print(f"Accuracy on test set: {100 * correct / total}")
    print(f"Accuracy on test set: {100 * correct / total:.3f}")

model = torch.load("model.pt")
test(model,test_loader)