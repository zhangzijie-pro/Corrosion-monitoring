#!/usr/bin/env python3
import argparse

from engine import predict_path
from model import build_model
from utils import get_device
from utils.checkpoint import infer_model_cfg_from_state_dict, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Predict corrosion masks with a segmentation checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="runs/predictions_3")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    device = get_device(args.device)
    arch, num_classes, model_cfg = infer_model_cfg_from_state_dict(cfg, checkpoint["model"])
    model = build_model(arch=arch, num_classes=num_classes, **model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    saved = predict_path(
        model,
        args.input,
        args.output_dir,
        cfg["data"].get("image_size", [512, 512]),
        device,
        threshold=args.threshold,
    )
    for item in saved:
        print(
            f"{item['source']} | area={item['corrosion_area_percent']:.2f}% | "
            f"level={item['corrosion_level']}({item['corrosion_level_name']}) | "
            f"raw_area={item['raw_corrosion_area_percent']:.2f}%"
        )
        print(f"Saved raw mask: {item['raw_mask']}")
        print(f"Saved mask: {item['mask']}")
        print(f"Saved grade heatmap: {item['grade_heatmap']}")
        print(f"Saved report: {item['report']}")


if __name__ == "__main__":
    main()
