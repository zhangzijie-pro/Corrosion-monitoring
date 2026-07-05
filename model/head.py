import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskUpscaler(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32 if hidden_dim % 32 == 0 else 1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32 if hidden_dim % 32 == 0 else 1, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CRTMaskHead(nn.Module):
    """SAM-inspired dynamic mask head plus IoU quality prediction."""

    def __init__(self, embed_dim=256, hidden_dim=128, num_queries=8, num_classes=1, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.upscaler = MaskUpscaler(embed_dim, hidden_dim)
        self.query_to_kernel = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * num_classes),
        )
        self.query_score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.iou_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, decoded_queries, dense_feature, output_size):
        dense_feature = self.upscaler(dense_feature)
        kernels = self.query_to_kernel(decoded_queries)
        batch, queries, _ = kernels.shape
        channels = dense_feature.shape[1]
        kernels = kernels.view(batch, queries * self.num_classes, channels)
        masks = torch.einsum("bqc,bchw->bqhw", kernels, dense_feature)
        masks = masks.view(batch, queries, self.num_classes, *dense_feature.shape[-2:])

        scores = self.query_score(decoded_queries).sigmoid().view(batch, queries, self.num_classes, 1, 1)
        logits = (masks * scores).sum(dim=1) / scores.sum(dim=1).clamp_min(1e-6)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        iou_pred = self.iou_head(decoded_queries).sigmoid().mean(dim=1)
        return {"out": logits, "query_masks": masks, "query_scores": scores.squeeze(-1).squeeze(-1), "iou_pred": iou_pred}
