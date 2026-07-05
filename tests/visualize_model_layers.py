#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.corrosion_dataset import IMAGENET_MEAN, IMAGENET_STD
from model import build_model
from model.encode import fuse_memory_maps, split_memory_to_maps
from utils import get_device
from utils.checkpoint import load_checkpoint


def load_image(path, image_size):
    original = Image.open(path).convert("RGB")
    resized = original.resize((image_size[1], image_size[0]), Image.BILINEAR)
    image = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return original, tensor.unsqueeze(0)


def normalize01(array):
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(array[finite], [1, 99])
    if hi <= lo:
        lo, hi = float(array[finite].min()), float(array[finite].max())
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0)


def save_heatmap(array, path, size=None):
    image = (normalize01(array) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    if size is not None:
        heatmap = cv2.resize(heatmap, size, interpolation=cv2.INTER_CUBIC)
    Image.fromarray(heatmap).save(path)


def save_overlay(original, mask, path, color=(255, 255, 0), alpha=0.55):
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    color = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    mask_bool = mask > 0
    out = base.copy()
    out[mask_bool] = base[mask_bool] * (1.0 - alpha) + color * alpha
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = out.astype(np.uint8)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    cv2.drawContours(out, contours, -1, (15, 23, 42), 1)
    Image.fromarray(out).save(path)


def channel_energy(tensor):
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4:
        return tensor[0].abs().mean(dim=0).numpy()
    if tensor.ndim == 3:
        return tensor[0].abs().mean(dim=-1).numpy()
    if tensor.ndim == 2:
        return tensor.abs().mean(dim=-1).numpy()
    return tensor.reshape(-1).numpy()


def save_top_channels(feature, output_dir, prefix, max_channels=16):
    feature = feature.detach().float().cpu()[0]
    channels = feature.shape[0]
    scores = feature.abs().flatten(1).mean(dim=1)
    indices = torch.topk(scores, k=min(max_channels, channels)).indices.tolist()
    thumbs = []
    for idx in indices:
        fmap = normalize01(feature[idx].numpy())
        thumb = (fmap * 255).astype(np.uint8)
        thumb = cv2.applyColorMap(thumb, cv2.COLORMAP_TURBO)
        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        thumb = cv2.resize(thumb, (128, 128), interpolation=cv2.INTER_CUBIC)
        thumbs.append((idx, thumb))

    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    canvas = Image.new("RGB", (cols * 128, rows * 150), "white")
    draw = ImageDraw.Draw(canvas)
    for n, (idx, thumb) in enumerate(thumbs):
        x = (n % cols) * 128
        y = (n // cols) * 150
        canvas.paste(Image.fromarray(thumb), (x, y))
        draw.text((x + 4, y + 130), f"ch {idx}", fill=(15, 23, 42))
    canvas.save(output_dir / f"{prefix}_top_channels.png")


def save_query_masks(query_masks, query_scores, output_dir, original_size):
    masks = query_masks.detach().float().cpu()[0, :, 0]
    scores = query_scores.detach().float().cpu()[0, :, 0]
    for idx, mask in enumerate(masks):
        save_heatmap(mask.numpy(), output_dir / f"head_query_{idx:02d}_mask.png", size=original_size)
    with open(output_dir / "head_query_scores.json", "w", encoding="utf-8") as f:
        json.dump({f"query_{idx:02d}": float(score) for idx, score in enumerate(scores)}, f, indent=2)


def run(args):
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    device = get_device(args.device)

    model_cfg = dict(cfg["model"])
    model_cfg.pop("arch", None)
    num_classes = model_cfg.pop("num_classes", 1)
    model = build_model(arch=cfg["model"].get("arch", "crt"), num_classes=num_classes, **model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    image_size = cfg["data"].get("image_size", [512, 512])
    original, image = load_image(args.input, image_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original.save(output_dir / "input_original.png")

    with torch.no_grad():
        image = image.to(device)
        output_size = image.shape[-2:]
        pyramid = model.backbone(image)
        encoded = model.encoder(pyramid)
        decoded = model.decoder(encoded["memory"])
        memory_maps = split_memory_to_maps(encoded["memory"], encoded["spatial_shapes"])
        dense_feature = fuse_memory_maps(memory_maps, target_size=pyramid[0].shape[-2:])
        outputs = model.head(decoded, dense_feature, output_size)

    summary = {
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "image_size": list(output_size),
        "threshold": args.threshold,
        "layers": {},
    }

    target_size = (original.width, original.height)
    for idx, feature in enumerate(pyramid, start=1):
        name = f"backbone_stage_{idx}"
        summary["layers"][name] = list(feature.shape)
        save_heatmap(channel_energy(feature), output_dir / f"{name}_energy.png", size=target_size)
        save_top_channels(feature, output_dir, name, max_channels=args.max_channels)

    for idx, feature in enumerate(encoded["projected_maps"], start=1):
        name = f"encoder_projected_level_{idx}"
        summary["layers"][name] = list(feature.shape)
        save_heatmap(channel_energy(feature), output_dir / f"{name}_energy.png", size=target_size)

    for idx, feature in enumerate(memory_maps, start=1):
        name = f"encoder_memory_level_{idx}"
        summary["layers"][name] = list(feature.shape)
        save_heatmap(channel_energy(feature), output_dir / f"{name}_energy.png", size=target_size)

    summary["layers"]["encoder_memory_tokens"] = list(encoded["memory"].shape)
    save_heatmap(channel_energy(encoded["memory"]), output_dir / "encoder_memory_token_energy.png")

    summary["layers"]["decoder_queries"] = list(decoded.shape)
    save_heatmap(decoded.detach().float().cpu()[0].numpy(), output_dir / "decoder_query_embedding_heatmap.png")

    summary["layers"]["dense_feature"] = list(dense_feature.shape)
    save_heatmap(channel_energy(dense_feature), output_dir / "head_dense_feature_energy.png", size=target_size)
    save_query_masks(outputs["query_masks"], outputs["query_scores"], output_dir, target_size)

    logits = outputs["out"]
    prob = torch.sigmoid(logits)
    prob = F.interpolate(prob, size=(original.height, original.width), mode="bilinear", align_corners=False)
    prob_np = prob.squeeze().detach().cpu().numpy()
    raw_mask = (prob_np >= args.threshold).astype(np.uint8) * 255
    Image.fromarray((prob_np * 255).astype(np.uint8)).save(output_dir / "final_probability.png")
    Image.fromarray(raw_mask).save(output_dir / "final_raw_mask.png")
    save_overlay(original, raw_mask, output_dir / "final_raw_overlay.png")

    summary["layers"]["head_query_masks"] = list(outputs["query_masks"].shape)
    summary["layers"]["head_query_scores"] = list(outputs["query_scores"].shape)
    summary["layers"]["head_logits"] = list(logits.shape)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved layer visualizations to: {output_dir}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize CRT intermediate layer outputs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-img", "--input", dest="input", required=True)
    parser.add_argument("--output-dir", default="tests/layer_outputs")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--max-channels", type=int, default=16)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
