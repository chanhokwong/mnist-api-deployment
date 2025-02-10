from Settings import *
import torch

class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, img_size=64):
        self.T = T
        self.img_size = img_size
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward_process(self, x0, t):
        # 扩展维度为 [Batch, 1, 1, 1] 确保广播正确
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bars[t]).view(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise