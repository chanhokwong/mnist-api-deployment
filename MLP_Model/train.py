from dataset import train_loader
from model import *
import torch

def train(model, train_loader, optimizer, loss_entropy, epochs=5):
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)  # output.data 為預測後的參數
            loss = loss_entropy(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch {epoch+1} loss: {running_loss/len(train_loader)}")
    torch.save(model,"model.pt")


train(model, train_loader,optimizer,loss_entropy, epochs=5)
