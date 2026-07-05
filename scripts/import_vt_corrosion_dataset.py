#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


COLOR_TO_CLASS = {
    (0, 0, 0): 0,
    (128, 0, 0): 2,
    (0, 128, 0): 4,
    (128, 128, 0): 5,
}


def convert_mask(mask_path):
    arr = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
    code = (
        arr[..., 0].astype(np.uint32) << 16
        | arr[..., 1].astype(np.uint32) << 8
        | arr[..., 2].astype(np.uint32)
    )
    out = np.zeros(arr.shape[:2], dtype=np.uint8)
    unknown = np.ones(arr.shape[:2], dtype=bool)
    for color, cls in COLOR_TO_CLASS.items():
        color_code = (color[0] << 16) | (color[1] << 8) | color[2]
        match = code == color_code
        out[match] = cls
        unknown &= ~match
    if np.any(unknown):
        colors = np.unique(arr[unknown].reshape(-1, 3), axis=0)
        raise ValueError("Unknown VT mask colors in %s: %s" % (mask_path, colors[:20].tolist()))
    return Image.fromarray(out, mode="L")


def collect_pairs(source_root):
    root = Path(source_root)
    candidates = [
        root / "Corrosion Condition State Classification" / "512x512",
        root / "512x512",
        root,
    ]
    base = next((p for p in candidates if (p / "Train").exists()), None)
    if base is None:
        raise FileNotFoundError("Could not find VT 512x512 Train/Test directories under %s" % root)

    pairs = []
    for source_split in ("Train", "Test"):
        image_dir = base / source_split / "images_512"
        mask_dir = base / source_split / "mask_512"
        for image_path in sorted(image_dir.glob("*.jpeg"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            mask_path = mask_dir / (image_path.stem + ".png")
            if mask_path.exists():
                pairs.append((source_split, image_path, mask_path))
    if not pairs:
        raise RuntimeError("No VT image/mask pairs found under %s" % base)
    return pairs


def choose_split(source_split, rng, valid_ratio):
    if source_split == "Test":
        return "test"
    return "valid" if rng.random() < valid_ratio else "train"


def main():
    parser = argparse.ArgumentParser(description="Import Virginia Tech corrosion condition state dataset.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", default="dataset_all")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--prefix", default="vt_corrosion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    target_root = Path(args.target_root)
    pairs = collect_pairs(args.source_root)
    manifest = {
        "source_root": args.source_root,
        "target_root": str(target_root),
        "source": "Virginia Tech Corrosion Condition State Semantic Segmentation Dataset",
        "mapping": {
            "Good/background": 0,
            "Fair": 2,
            "Poor": 4,
            "Severe": 5,
        },
        "items": [],
    }
    counts = {str(i): 0 for i in range(6)}
    copied = 0
    skipped = 0

    for source_split, image_path, mask_path in pairs:
        split = choose_split(source_split, rng, args.valid_ratio)
        stem = "%s_%s_%s" % (args.prefix, source_split.lower(), image_path.stem)
        split_dir = target_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        image_out = split_dir / (stem + ".jpg")
        mask_out = split_dir / (stem + "_mask.png")
        if not args.overwrite and (image_out.exists() or mask_out.exists()):
            skipped += 1
            continue

        shutil.copy2(image_path, image_out)
        mask = convert_mask(mask_path)
        mask.save(mask_out)
        mask_arr = np.asarray(mask, dtype=np.uint8)
        unique, pixel_counts = np.unique(mask_arr, return_counts=True)
        item_counts = {str(int(k)): int(v) for k, v in zip(unique, pixel_counts)}
        for k, v in item_counts.items():
            counts[k] = counts.get(k, 0) + int(v)
        manifest["items"].append(
            {
                "split": split,
                "source_split": source_split,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "target_image": str(image_out),
                "target_mask": str(mask_out),
                "class_pixels": item_counts,
            }
        )
        copied += 1

    manifest["copied"] = copied
    manifest["skipped"] = skipped
    manifest["class_pixels"] = counts
    with open(target_root / "vt_corrosion_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("VT pairs found:", len(pairs))
    print("Copied:", copied)
    print("Skipped:", skipped)
    print("Target root:", target_root)
    print("Class pixels:", counts)


if __name__ == "__main__":
    main()
