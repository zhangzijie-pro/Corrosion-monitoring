from pathlib import Path

import torch


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def infer_model_cfg_from_state_dict(cfg, state_dict):
    model_cfg = dict(cfg["model"])
    arch = str(model_cfg.pop("arch", "crt")).lower()
    num_classes = model_cfg.pop("num_classes", 1)
    return arch, num_classes, model_cfg


def save_checkpoint(path, model, optimizer, epoch, metrics, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )
