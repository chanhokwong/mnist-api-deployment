import torch
from Settings import *
from train import *
from Diffusion import *

@torch.no_grad()
def sample(model, diffusion, batch_size=16):
    model.eval()
    # 从纯噪声开始
    x = torch.randn(batch_size, 3, diffusion.img_size, diffusion.img_size).to(device)

    for t in reversed(range(diffusion.T)):
        # 计算时间步t的嵌入
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # 预测噪声
        pred_noise = model(x, t_tensor)

        # 反向去噪公式
        alpha_t = diffusion.alphas[t]
        alpha_bar_t = diffusion.alpha_bars[t]
        beta_t = diffusion.betas[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        ) + torch.sqrt(beta_t) * noise

    # 反归一化到[0,1]
    x = (x.clamp(-1, 1) + 1) / 2
    return x.cpu()


# 生成并保存图像
model = torch.load("model.pt")
diffusion = Diffusion(T=1000)
generated_images = sample(model, diffusion, batch_size=48)
torchvision.utils.save_image(generated_images, "generated_samples.png")


# 加速采样
# 使用DDIM（Denoising Diffusion Implicit Models）减少采样步数：
# def ddim_sample(model, diffusion, batch_size=16, ddim_steps=50):
#     model.eval()
#     x = torch.randn(batch_size, 3, diffusion.img_size, diffusion.img_size).to(device)
#     step_interval = diffusion.T // ddim_steps
#     for t in reversed(range(0, diffusion.T, step_interval)):
#         t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
#         pred_noise = model(x, t_tensor)
#         # DDIM更新公式（需调整系数）
#         # 具体公式参考论文
#     return x