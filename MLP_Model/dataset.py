from torchvision import datasets,transforms
import torch
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))])

train_dataset = datasets.MNIST(root="./data",train=True,download=True,transform=transform)
test_dataset = datasets.MNIST(root="./data",train=False,download=False,transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=64)

images, labels = next(iter(train_loader))
plt.imshow(images[0].squeeze(), cmap='gray')
plt.title(f"Label: {labels[0]}")
plt.show()