import torch
from torch import nn

# 通道注意力机制
class channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ave_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = self.ave_pool(x).view([b, c])
        avg_out = self.fc(avg_out)
        max_out = self.max_pool(x).view([b, c])
        max_out = self.fc(max_out)
        out = avg_out + max_out
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x

# 空间注意力机制
class spacial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spacial_attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out * x

# 先进入通道注意力机制，后进入空间注意力机制，最后输出
class Cba(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(Cba, self).__init__()
        self.channel_attention = channel_attention(channel, ratio)
        self.spacial_attention = spacial_attention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spacial_attention(x)
        return x
