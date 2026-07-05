#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from data import build_dataloaders
from engine import evaluate, predict_path
from loss import MulticlassSegmentationCriterion
from model import build_model
from utils import get_device, seed_everything
from utils.checkpoint import infer_model_cfg_from_state_dict, load_checkpoint


def _format_value(value):
    if isinstance(value, list):
        return "[" + ", ".join("{:.6f}".format(float(item)) for item in value) + "]"
    if isinstance(value, float):
        return "{:.6f}".format(value)
    return str(value)


def _find_sample_image():
    for root in (Path("dataset/HiRes/raw"), Path("dataset/LoRes/raw")):
        if root.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                images = sorted(root.glob(ext))
                if images:
                    return images[0]
    return None


def _load_multiclass_model(checkpoint_path, device_name):
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    arch, num_classes, model_cfg = infer_model_cfg_from_state_dict(cfg, checkpoint["model"])
    if int(num_classes) <= 2:
        raise RuntimeError(
            "Checkpoint is not a multiclass semantic segmentation model: arch={}, num_classes={}".format(
                arch, num_classes
            )
        )
    device = get_device(device_name)
    model = build_model(arch=arch, num_classes=num_classes, **model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return checkpoint, cfg, arch, num_classes, model, device


def _run_eval(model, cfg, device, seed):
    _, val_loader = build_dataloaders(cfg, seed=seed, device=device)
    criterion = MulticlassSegmentationCriterion(**cfg.get("loss", {})).to(device)
    eval_cfg = cfg.get("eval", {})
    return evaluate(
        model,
        val_loader,
        criterion,
        device,
        threshold=eval_cfg.get("threshold", 0.5),
        boundary_tolerance=eval_cfg.get("boundary_tolerance", 3),
        metric_type="multiclass",
        num_classes=cfg["model"].get("num_classes", 6),
    )


def main():
    parser = argparse.ArgumentParser(description="Test only the multiclass corrosion semantic segmentation model.")
    parser.add_argument("--checkpoint", default="runs/best.pt")
    parser.add_argument("--input", help="Image file or directory for prediction smoke test.")
    parser.add_argument("--output-dir", default="runs/multiclass_test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval", action="store_true", help="Run validation metrics with the checkpoint config data root.")
    parser.add_argument("--save-json", default="runs/multiclass_test/summary.json")
    args = parser.parse_args()

    checkpoint, cfg, arch, num_classes, model, device = _load_multiclass_model(args.checkpoint, args.device)
    seed = cfg.get("seed", 42)
    seed_everything(seed)

    print("checkpoint: {}".format(args.checkpoint))
    print("arch: {}".format(arch))
    print("num_classes: {}".format(num_classes))
    print("device: {}".format(device))
    if "epoch" in checkpoint:
        print("checkpoint_epoch: {}".format(checkpoint["epoch"]))

    summary = {
        "checkpoint": args.checkpoint,
        "arch": arch,
        "num_classes": int(num_classes),
        "device": str(device),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
    }

    if args.eval:
        metrics = _run_eval(model, cfg, device, seed)
        summary["eval_metrics"] = metrics
        for key in sorted(metrics):
            print("{}: {}".format(key, _format_value(metrics[key])))
    else:
        print("eval: skipped; pass --eval when the configured validation dataset is available.")

    input_path = Path(args.input) if args.input else _find_sample_image()
    if input_path is None:
        print("prediction: skipped; no --input was provided and no sample image was found.")
    else:
        saved = predict_path(
            model,
            input_path,
            args.output_dir,
            cfg["data"].get("image_size", [512, 512]),
            device,
            threshold=cfg.get("eval", {}).get("threshold", 0.7),
        )
        summary["predictions"] = saved
        for item in saved:
            print(
                "{} | area={:.2f}% | level={}({}) | report={}".format(
                    item["source"],
                    item["corrosion_area_percent"],
                    item["corrosion_level"],
                    item["corrosion_level_name"],
                    item["report"],
                )
            )

    save_json = Path(args.save_json)
    save_json.parent.mkdir(parents=True, exist_ok=True)
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("summary: {}".format(save_json))


if __name__ == "__main__":
    main()
