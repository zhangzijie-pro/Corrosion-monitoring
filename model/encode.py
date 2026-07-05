import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos import SinePositionEncoding2D


class MultiScaleTokenProjector(nn.Module):
    """Project pyramid feature maps into one transformer token sequence."""

    def __init__(self, in_channels, embed_dim, dropout=0.0, token_max_size=32):
        super().__init__()
        self.token_max_size = token_max_size
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, embed_dim, 1, bias=False),
                    nn.GroupNorm(32 if embed_dim % 32 == 0 else 1, embed_dim),
                    nn.GELU(),
                )
                for channels in in_channels
            ]
        )
        self.level_embed = nn.Parameter(torch.zeros(len(in_channels), embed_dim))
        self.pos = SinePositionEncoding2D(embed_dim // 2)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.level_embed, std=0.02)

    def forward(self, features):
        tokens = []
        spatial_shapes = []
        projected_maps = []
        for level, (feature, projection) in enumerate(zip(features, self.projections)):
            projected = projection(feature)
            height = int(projected.size(-2))
            width = int(projected.size(-1))
            if self.token_max_size and max(height, width) > self.token_max_size:
                scale = self.token_max_size / float(max(height, width))
                size = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
                projected = F.interpolate(projected, size=size, mode="bilinear", align_corners=False)
            projected_maps.append(projected)
            batch, channels, height, width = projected.shape
            pos = self.pos(height, width, projected.device, projected.dtype)
            pos = pos + self.level_embed[level].view(1, 1, -1)
            token = projected.flatten(2).transpose(1, 2) + pos
            tokens.append(token)
            spatial_shapes.append((height, width))
        return self.dropout(torch.cat(tokens, dim=1)), spatial_shapes, projected_maps


class CRTEncoder(nn.Module):
    """Global context encoder over multi-scale visual tokens."""

    def __init__(self, in_channels, embed_dim=256, depth=3, num_heads=8, mlp_ratio=4.0, dropout=0.1, token_max_size=32):
        super().__init__()
        self.projector = MultiScaleTokenProjector(in_channels, embed_dim, dropout=dropout, token_max_size=token_max_size)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features):
        tokens, spatial_shapes, projected_maps = self.projector(features)
        memory = self.norm(self.encoder(tokens))
        return {
            "memory": memory,
            "spatial_shapes": spatial_shapes,
            "projected_maps": projected_maps,
        }


def split_memory_to_maps(memory, spatial_shapes):
    maps = []
    cursor = 0
    batch, _, channels = memory.shape
    for height, width in spatial_shapes:
        length = height * width
        chunk = memory[:, cursor : cursor + length]
        maps.append(chunk.transpose(1, 2).reshape(batch, channels, height, width))
        cursor += length
    return maps


def fuse_memory_maps(memory_maps, target_size):
    resized = [
        F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
        if feature.shape[-2:] != target_size
        else feature
        for feature in memory_maps
    ]
    return torch.stack(resized, dim=0).mean(dim=0)
