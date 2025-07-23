import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class EDSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=8, scale=4):
        super(EDSR, self).__init__()

        self.entry = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(num_features) for _ in range(num_blocks)
        ])

        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.entry(x)
        res = self.res_blocks(x)
        x = x + res  # Global residual connection
        out = self.upsample(x)
        return out
