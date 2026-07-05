import torch.nn as nn

from .backbone import HierarchicalViTBackbone
from .decode import CRTDecoder
from .encode import CRTEncoder, fuse_memory_maps, split_memory_to_maps
from .head import CRTMaskHead
class CRT(nn.Module):
    """Corrosion Recognition Transformer.

    The architecture follows the practical split used by SAM/SAM2 and DETR-like
    models: visual backbone, token encoder, query decoder, and mask/quality head.
    """

    def __init__(
        self,
        num_classes=1,
        embed_dims=(64, 128, 320, 512),
        depths=(2, 2, 4, 2),
        num_heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        encoder_dim=256,
        encoder_depth=3,
        encoder_heads=8,
        token_max_size=32,
        decoder_depth=3,
        decoder_heads=8,
        num_queries=8,
        head_dim=128,
        dropout=0.1,
        drop_path=0.1,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.backbone = HierarchicalViTBackbone(
            embed_dims=tuple(embed_dims),
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            sr_ratios=tuple(sr_ratios),
            mlp_ratio=mlp_ratio,
            drop=dropout,
            attn_drop=dropout,
            drop_path=drop_path,
        )
        self.encoder = CRTEncoder(
            self.backbone.out_channels,
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            token_max_size=token_max_size,
        )
        self.decoder = CRTDecoder(
            embed_dim=encoder_dim,
            num_queries=num_queries,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.head = CRTMaskHead(
            embed_dim=encoder_dim,
            hidden_dim=head_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, images):
        output_size = images.shape[-2:]
        pyramid = self.backbone(images)
        encoded = self.encoder(pyramid)
        decoded = self.decoder(encoded["memory"])
        memory_maps = split_memory_to_maps(encoded["memory"], encoded["spatial_shapes"])
        dense_feature = fuse_memory_maps(memory_maps, target_size=pyramid[0].shape[-2:])
        output = self.head(decoded, dense_feature, output_size)
        output["features"] = pyramid
        return output
