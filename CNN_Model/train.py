from dataset import train_loader
from model import *
import torch

def train(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
    torch.save(model,"model.pt")

train(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, epochs=5)