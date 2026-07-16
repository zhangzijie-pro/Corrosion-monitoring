#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from data import build_dataloaders
from engine import evaluate, train_one_epoch
from loss import MulticlassSegmentationCriterion, SegmentationCriterion
from model import build_model
from utils import get_device, seed_everything
from utils.checkpoint import save_checkpoint
from utils.torch_compat import make_grad_scaler
from utils.visualize import append_jsonl, save_architecture_diagram, save_metrics_figures


def apply_cli_overrides(cfg, args):
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.image_size is not None:
        cfg["data"]["image_size"] = [args.image_size, args.image_size]
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    if args.max_samples is not None:
        cfg["data"]["max_samples"] = args.max_samples
    return cfg


def model_arch(cfg):
    return str(cfg["model"].get("arch", "crt")).lower().replace("_", "-")


def metric_type_for_arch(arch, num_classes=2):
    if int(num_classes) > 2:
        return "multiclass"
    return "segmentation"


def build_criterion_for_arch(arch, cfg, num_classes=1):
    if int(num_classes) > 2:
        return MulticlassSegmentationCriterion(**cfg.get("loss", {}))
    return SegmentationCriterion(**cfg.get("loss", {}))


def best_metric_key_for_arch(arch, num_classes=1):
    if int(num_classes) > 2:
        return "Foreground Binary IoU"
    return "iou"


def resolve_best_metric_key(metrics, preferred_key):
    if preferred_key in metrics:
        return preferred_key
    for key in ("Mean Intersection over Union(mIoU)", "iou", "miou"):
        if key in metrics:
            return key
    raise KeyError(f"No supported best metric found. Available metrics: {sorted(metrics.keys())}")


def format_epoch_metrics(epoch, metrics, best_key):
    parts = [
        f"Epoch {epoch:03d}",
        f"train_loss={metrics['train_loss']:.4f}",
        f"val_loss={metrics['val_loss']:.4f}",
    ]
    emitted = set()
    for key in (
        best_key,
        "Mean Present Class IoU",
        "Foreground Binary IoU",
        "Foreground Binary Dice",
        "Foreground Binary Recall",
        "Mean Intersection over Union(mIoU)",
        "Pixel Accuracy",
        "Mean Pixel Accuracy",
        "Mean F1 Score",
        "miou",
        "iou",
        "dice",
        "boundary_f1",
    ):
        if key in metrics and key not in emitted:
            parts.append(f"{key}={metrics[key]:.4f}")
            emitted.add(key)
    parts.append(f"lr={metrics['lr']:.2e}")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Train a corrosion segmentation model.")
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_cli_overrides(cfg, args)
    seed = cfg.get("seed", 42)
    seed_everything(seed)
    device = get_device(cfg.get("device", "auto"))

    arch = model_arch(cfg)
    train_loader, val_loader = build_dataloaders(cfg, seed=seed, device=device)
    model_cfg = dict(cfg["model"])
    model_cfg.pop("arch", None)
    num_classes = model_cfg.pop("num_classes", 1)
    model = build_model(arch=arch, num_classes=num_classes, **model_cfg)
    model.to(device)

    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, train_cfg.get("epochs", 50)),
        eta_min=train_cfg.get("min_lr", 1e-6),
    )
    criterion = build_criterion_for_arch(arch, cfg, num_classes=num_classes).to(device)
    amp = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler = make_grad_scaler(enabled=amp)
    output_dir = Path(train_cfg.get("output_dir", f"runs/{arch}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    if arch in {"crt", "corrosion_transformer", "corrosion_vit"}:
        save_architecture_diagram(output_dir / "crt_architecture.png")

    print(f"Device: {device}")
    print(f"Model: {arch}")
    print(f"Train/val samples: {len(train_loader.dataset)}/{len(val_loader.dataset)}")
    if arch in {"crt", "corrosion_transformer", "corrosion_vit"}:
        print(f"Architecture diagram: {output_dir / 'crt_architecture.png'}")

    best_score = -math.inf
    best_metric_key = best_metric_key_for_arch(arch, num_classes=num_classes)
    metric_type = metric_type_for_arch(arch, num_classes=num_classes)
    if int(num_classes) <= 2:
        metric_type = cfg.get("eval", {}).get("metric_type", metric_type)
    for epoch in range(1, train_cfg.get("epochs", 50) + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            amp=amp,
            max_grad_norm=train_cfg.get("max_grad_norm", 0.0),
        )
        scheduler.step()
        eval_cfg = cfg.get("eval", {})
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=eval_cfg.get("threshold", 0.5),
            boundary_tolerance=eval_cfg.get("boundary_tolerance", 3),
            metric_type=metric_type,
            num_classes=num_classes,
        )
        metrics = {"epoch": epoch, **train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        append_jsonl(output_dir / "metrics.jsonl", metrics)
        save_metrics_figures(output_dir / "metrics.jsonl", output_dir / "figures", title_prefix=arch.upper())
        print(format_epoch_metrics(epoch, metrics, best_metric_key))

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, metrics, cfg)
        current_best_key = resolve_best_metric_key(metrics, best_metric_key)
        if metrics[current_best_key] > best_score:
            best_score = metrics[current_best_key]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, metrics, cfg)

    print(f"Metric figures: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
