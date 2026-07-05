from .CRT import CRT
from .Unet import UNet


def build_model(arch="crt", num_classes=1, **kwargs):
    arch = str(arch).lower()
    if arch in {"crt", "corrosion_transformer", "corrosion_vit"}:
        kwargs.pop("in_channels", None)
        kwargs.pop("out_channels", None)
        return CRT(num_classes=num_classes, **kwargs)
    if arch in {"unet", "u-net"}:
        in_channels = kwargs.get("in_channels", 3)
        out_channels = kwargs.get("out_channels", max(2, num_classes))
        return UNet(in_channels=in_channels, out_channels=out_channels)
    raise ValueError(f"Unsupported architecture: {arch}. Use arch='crt' or arch='unet'.")


__all__ = ["CRT", "UNet", "build_model"]
