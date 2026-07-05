import math

import torch
import torch.nn as nn


class SinePositionEncoding2D(nn.Module):
    """DETR-style 2D sine/cosine positional encoding."""

    def __init__(self, num_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_feats = int(num_feats)
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, height, width, device, dtype=torch.float32):
        y_embed = torch.arange(height, device=device, dtype=dtype).unsqueeze(1).repeat(1, width)
        x_embed = torch.arange(width, device=device, dtype=dtype).unsqueeze(0).repeat(height, 1)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (height - 1 + eps) * self.scale
            x_embed = x_embed / (width - 1 + eps) * self.scale

        dim_t = torch.arange(self.num_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        return torch.cat((pos_y, pos_x), dim=2).view(1, height * width, -1)


class LearnedQueryEmbedding(nn.Module):
    """DETR/RF-DETR-style learned object queries."""

    def __init__(self, num_queries, embed_dim):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, embed_dim)

    def forward(self, batch_size):
        return self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
