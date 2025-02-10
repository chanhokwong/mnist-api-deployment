import torch
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image

from model import *
import torch.optim as optim
from train import *

def main():
    input_dim = 784
    hidden_dim = 400
    latent_dim = 20
    epochs = 20
    batch_size = 128
    learning_rate = 1e-2

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train(epoch, model, train_loader, optimizer)

    # 監控生成樣本質量
    with torch.no_grad():
        z = torch.randn(64, latent_dim)  # 從標準正態分佈採樣
        generated = model.decoder(z)
        save_image(generated.view(64, 1, 28, 28), 'samples.png')

    # 生成64个潜在变量
    z = torch.randn(64, latent_dim)
    generated_images = model.decoder(z)

    # 可视化（需配合matplotlib或torchvision的save_image）
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(generated_images[i].view(28, 28).cpu().numpy(), cmap='gray')
    plt.show()

    torch.save(model, "model.pt")

if __name__ == "__main__":
    main()
