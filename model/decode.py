import torch
import torch.nn as nn

from .pos import LearnedQueryEmbedding


class CRTDecoder(nn.Module):
    """DETR-style query decoder with cross-attention to encoded image tokens."""

    def __init__(
        self,
        embed_dim=256,
        num_queries=8,
        depth=3,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.queries = LearnedQueryEmbedding(num_queries, embed_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, memory):
        batch_size = memory.shape[0]
        query_pos = self.queries(batch_size)
        target = torch.zeros_like(query_pos)
        decoded = self.decoder(target + query_pos, memory)
        return self.norm(decoded)
