import torch
from dataset import *
from model import *

def train(model, train_loader, criterion, optimizer, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")

    # 保存模型權重
    torch.save(model, "flower_classifier/model.pt")
    # torch.save(model,"model.pb")
    # torch.save(model.state_dict(), "flower_classifier/flower_classifier.pth")

train(model,train_loader,criterion,optimizer,epochs=16)