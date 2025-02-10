from torchvision import datasets, transforms
import torch
from model import *
from torch.utils.data.dataloader import DataLoader

batch_size = 128
transform = transforms.Compose([transforms.ToTensor()])
val_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def validate(model, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.view(-1, 784)
            recon, mu, log_var = model(data)
            val_loss += loss_function(recon, data, mu, log_var).item()
    return val_loss / len(val_loader.dataset)

model = torch.load("model.pt")
result = validate(model,val_loader)
print("loss: ",result)