from .CRT import CRT


def build_model(arch="crt", num_classes=1, **kwargs):
    arch = str(arch).lower()
    if arch in {"crt", "corrosion_transformer", "corrosion_vit"}:
        kwargs.pop("in_channels", None)
        kwargs.pop("out_channels", None)
        return CRT(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unsupported architecture: {arch}. Use arch='crt'.")


__all__ = ["CRT", "build_model"]
