import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class OverlapPatchEmbed(nn.Module):
    """SegFormer-style overlapping image patch embedding."""

    def __init__(self, in_channels, embed_dim, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2)
        return self.norm(tokens), height, width


class SpatialReductionAttention(nn.Module):
    """PVT/MiT attention with reduced key/value tokens for high-resolution maps."""

    def __init__(self, dim, num_heads, sr_ratio=1, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = int(sr_ratio)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, height, width):
        batch, tokens, dim = x.shape
        q = self.q(x).reshape(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_map = x.transpose(1, 2).reshape(batch, dim, height, width)
            x_reduced = self.sr(x_map).flatten(2).transpose(1, 2)
            x_reduced = self.norm(x_reduced)
        else:
            x_reduced = x

        kv = self.kv(x_reduced).reshape(batch, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(x))


class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, height, width):
        batch = x.shape[0]
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(batch, -1, height, width)
        x = self.dwconv(x).flatten(2).transpose(1, 2)
        x = self.drop(self.act(x))
        return self.drop(self.fc2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, sr_ratio=1, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, num_heads, sr_ratio, attn_drop, drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class HierarchicalViTBackbone(nn.Module):
    """Four-stage visual transformer backbone for dense prediction."""

    def __init__(
        self,
        in_channels=3,
        embed_dims=(64, 128, 320, 512),
        depths=(2, 2, 4, 2),
        num_heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
    ):
        super().__init__()
        if not (len(embed_dims) == len(depths) == len(num_heads) == len(sr_ratios) == 4):
            raise ValueError("CRT expects four backbone stages.")
        self.out_channels = tuple(embed_dims)
        self.patch_embeds = nn.ModuleList(
            [
                OverlapPatchEmbed(in_channels, embed_dims[0], 7, 4),
                OverlapPatchEmbed(embed_dims[0], embed_dims[1], 3, 2),
                OverlapPatchEmbed(embed_dims[1], embed_dims[2], 3, 2),
                OverlapPatchEmbed(embed_dims[2], embed_dims[3], 3, 2),
            ]
        )
        drop_rates = torch.linspace(0, drop_path, sum(depths)).tolist()
        cursor = 0
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        for stage_idx, depth in enumerate(depths):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    TransformerBlock(
                        embed_dims[stage_idx],
                        num_heads[stage_idx],
                        mlp_ratio=mlp_ratio,
                        sr_ratio=sr_ratios[stage_idx],
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_rates[cursor],
                    )
                )
                cursor += 1
            self.stages.append(nn.ModuleList(blocks))
            self.norms.append(nn.LayerNorm(embed_dims[stage_idx]))

    def forward(self, x):
        features = []
        for patch_embed, blocks, norm in zip(self.patch_embeds, self.stages, self.norms):
            x, height, width = patch_embed(x)
            for block in blocks:
                x = block(x, height, width)
            x = norm(x)
            batch, _, channels = x.shape
            x = x.transpose(1, 2).reshape(batch, channels, height, width)
            features.append(x)
        return features
