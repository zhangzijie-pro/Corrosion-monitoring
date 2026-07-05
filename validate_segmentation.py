#!/usr/bin/env python3
import argparse

from data import build_dataloaders
from engine import evaluate
from loss import CrossEntropySegmentationCriterion, MulticlassSegmentationCriterion, SegmentationCriterion
from model import build_model
from utils import get_device, seed_everything
from utils.checkpoint import infer_model_cfg_from_state_dict, load_checkpoint


def metric_type_for_arch(arch, num_classes=2):
    if int(num_classes) > 2:
        return "multiclass"
    if str(arch).lower().replace("_", "-") in {"unet", "u-net"}:
        return "unet"
    return "segmentation"


def build_criterion_for_arch(arch, cfg, num_classes=1):
    if int(num_classes) > 2:
        return MulticlassSegmentationCriterion(**cfg.get("loss", {}))
    if str(arch).lower().replace("_", "-") in {"unet", "u-net"}:
        return CrossEntropySegmentationCriterion()
    return SegmentationCriterion(**cfg.get("loss", {}))


def format_metric_value(value):
    if isinstance(value, list):
        return "[" + ", ".join(f"{float(item):.6f}" for item in value) + "]"
    return f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser(description="Validate a corrosion segmentation checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary-tolerance", type=int, default=None)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    seed_everything(cfg.get("seed", 42))
    device = get_device(args.device)
    _, val_loader = build_dataloaders(cfg, seed=cfg.get("seed", 42), device=device)
    arch, num_classes, model_cfg = infer_model_cfg_from_state_dict(cfg, checkpoint["model"])
    model = build_model(arch=arch, num_classes=num_classes, **model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    boundary_tolerance = args.boundary_tolerance
    if boundary_tolerance is None:
        boundary_tolerance = cfg.get("eval", {}).get("boundary_tolerance", 3)
    metric_type = metric_type_for_arch(arch, num_classes=num_classes)
    if metric_type != "unet" and int(num_classes) <= 2:
        metric_type = cfg.get("eval", {}).get("metric_type", metric_type)
    metrics = evaluate(
        model,
        val_loader,
        build_criterion_for_arch(arch, cfg, num_classes=num_classes).to(device),
        device,
        threshold=args.threshold,
        boundary_tolerance=boundary_tolerance,
        metric_type=metric_type,
        num_classes=num_classes,
    )
    for key in sorted(metrics):
        print(f"{key}: {format_metric_value(metrics[key])}")


if __name__ == "__main__":
    main()
