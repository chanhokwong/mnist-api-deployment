from Settings import *

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 类似Transformer的位置编码
        self.proj = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.proj(emb)

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )
        self.time_mlp = nn.Linear(time_dim, out_ch)

    def forward(self, x, t_emb):
        h = self.conv(x)
        t_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)  # 调整形状匹配特征图
        return h + t_emb


class UNet(nn.Module):
    def __init__(self, in_ch=3, chs=(64, 128, 256, 512), time_dim=256):
        super().__init__()
        # 下采样路径
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        current_ch = in_ch
        for ch in chs:
            self.down.append(UNetBlock(current_ch, ch, time_dim))
            current_ch = ch
        # 中间层
        self.mid = UNetBlock(current_ch, current_ch, time_dim)
        # 上采样路径
        self.up = nn.ModuleList()
        for ch in reversed(chs):
            self.up.append(nn.ConvTranspose2d(current_ch, ch, 2, stride=2))
            self.up.append(UNetBlock(2*ch, ch, time_dim))  # 注意拼接后的通道数
            current_ch = ch
        self.final = nn.Conv2d(current_ch, in_ch, 1)  # 输出通道数必须与输入一致
        self.time_embed = TimeEmbedding(time_dim)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        skips = []
        # 下采样
        for block in self.down:
            x = block(x, t_emb)
            skips.append(x)
            x = self.pool(x)
        # 中间层
        x = self.mid(x, t_emb)
        # 上采样
        for i in range(0, len(self.up), 2):
            x = self.up[i](x)  # 转置卷积上采样
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)  # 跳跃连接
            x = self.up[i + 1](x, t_emb)
        return self.final(x)