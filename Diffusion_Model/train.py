# import torch
#
# from Settings import *
# from U_Net import *
# from Diffusion import *
# import torch.multiprocessing as mp
#
# def main():
#     # 数据预处理
#     transform = transforms.Compose([
#         transforms.Resize(64),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     # 加载数据集
#     dataset = torchvision.datasets.CIFAR10(
#         root='./data', train=True, download=True, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
#
#     model = UNet().to(device)
#     diffusion = Diffusion(T=1000)
#     optimizer = optim.AdamW(model.parameters(), lr=2e-4)
#     epochs = 100
#
#     for epoch in range(epochs):
#         model.train()
#         pbar = tqdm(dataloader)
#         for batch, _ in pbar:
#             x0 = batch.to(device)
#
#             # 随机采样时间步
#             t = torch.randint(0, diffusion.T, (x0.size(0),), device=device)
#
#             # 正向扩散过程
#             xt, noise = diffusion.forward_process(x0, t)
#
#             # 预测噪声
#             pred_noise = model(xt, t)
#
#             # 计算损失
#             loss = nn.MSELoss()(pred_noise, noise)
#
#             # 反向传播
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
#
#     torch.save(model,"model.pt")
#
#
# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     mp.set_start_method('spawn')
#     main()


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from Diffusion import *
from U_Net import *

def train():
    # 数据加载
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # 初始化模型与扩散工具
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    diffusion = Diffusion(T=1000)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    epochs = 10
    # epochs = 100

    # 训练循环
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader)
        for batch, _ in pbar:
            x0 = batch.to(device)
            t = torch.randint(0, diffusion.T, (x0.size(0),), device=device)
            xt, noise = diffusion.forward_process(x0, t)
            pred_noise = model(xt, t)
            loss = nn.MSELoss()(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    torch.save(model,"model.pt")

if __name__ == '__main__':
    train()  # 确保主逻辑在保护块内执行
