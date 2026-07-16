#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import build_model
from utils.checkpoint import infer_model_cfg_from_state_dict, load_checkpoint


class SegmentationONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = self.model(images)
        if isinstance(outputs, dict):
            return outputs["out"]
        return outputs


def export_onnx(
    checkpoint_path,
    output_path,
    device_name="cpu",
    opset=17,
    dynamic_batch=True,
    export_image_size=None,
    constant_folding=True,
):
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    arch, num_classes, model_cfg = infer_model_cfg_from_state_dict(cfg, checkpoint["model"])
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")

    model = build_model(arch=arch, num_classes=num_classes, **model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    wrapper = SegmentationONNXWrapper(model).to(device).eval()

    image_size = cfg.get("data", {}).get("image_size", [512, 512])
    if export_image_size is not None and int(export_image_size) > 0:
        image_size = [int(export_image_size), int(export_image_size)]
    height, width = int(image_size[0]), int(image_size[1])
    dummy = torch.randn(1, 3, height, width, device=device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch"},
            "logits": {0: "batch"},
        }

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=int(opset),
        do_constant_folding=bool(constant_folding),
    )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "onnx": str(output_path),
        "arch": arch,
        "num_classes": int(num_classes),
        "image_size": [height, width],
        "input_name": "image",
        "output_name": "logits",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "postprocess": {
            "threshold": float(cfg.get("eval", {}).get("threshold", 0.45)),
            "threshold_mode": "foreground_probability_sum",
            "background_margin": -0.05,
            "min_area_percent": 0.02,
            "min_component_area_ratio": 0.0001,
            "background_class": 0,
            "foreground_classes": [1, 2, 3, 4, 5],
        },
        "class_names": {
            "0": "Background",
            "1": "Slight corrosion",
            "2": "Light-moderate corrosion",
            "3": "Moderate corrosion",
            "4": "Severe corrosion",
            "5": "Critical corrosion",
        },
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return output_path, metadata_path


def main():
    parser = argparse.ArgumentParser(description="Export a corrosion segmentation .pt checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=15)
    parser.add_argument("--static-batch", action="store_true", help="Do not export dynamic batch axis.")
    parser.add_argument("--export-image-size", type=int, help="Override square image size for ONNX export.")
    parser.add_argument("--no-constant-folding", action="store_true")
    args = parser.parse_args()

    output_path, metadata_path = export_onnx(
        args.checkpoint,
        args.output,
        device_name=args.device,
        opset=args.opset,
        dynamic_batch=not args.static_batch,
        export_image_size=args.export_image_size,
        constant_folding=not args.no_constant_folding,
    )
    print("ONNX:", output_path)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
