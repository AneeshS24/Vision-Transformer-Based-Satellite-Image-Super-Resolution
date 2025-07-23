import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, emb_size=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSR(nn.Module):
    def __init__(self, in_channels=3, emb_size=96, patch_size=4, depth=6, heads=4, upscale=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed = PatchEmbedding(in_channels, emb_size, patch_size)
        self.encoder = nn.Sequential(*[ViTBlock(emb_size, heads) for _ in range(depth)])

        self.to_feat_map = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
        )

        self.reconstruct = nn.Sequential(
            nn.Conv2d(emb_size, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3 * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.embed(x)  # (B, N, emb)
        x = self.encoder(x)
        x = self.to_feat_map(x)
        x = x.transpose(1, 2).reshape(B, -1, H // self.patch_size, W // self.patch_size)  # (B, emb, h, w)
        x = self.reconstruct(x)  # (B, 3, H*upscale, W*upscale)
        return x
