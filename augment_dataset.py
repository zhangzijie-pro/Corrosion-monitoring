#!/usr/bin/env python3
import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")


def find_pairs(root, splits):
    pairs = []
    missing = []
    for split in splits:
        split_dir = root / split
        image_paths = []
        for ext in IMAGE_EXTS:
            image_paths.extend(split_dir.glob(ext))
        for image_path in sorted(image_paths):
            if image_path.stem.endswith("_mask"):
                continue
            mask_path = split_dir / f"{image_path.stem}_mask.png"
            if not mask_path.exists():
                missing.append(str(image_path))
                continue
            pairs.append((split, image_path, mask_path))
    if missing:
        print(f"Warning: skipped {len(missing)} images without matching *_mask.png labels.")
    if not pairs:
        raise RuntimeError(f"No dataset_all image/mask pairs found under {root}")
    return pairs


def binarize_mask(mask, black_threshold=0):
    arr = np.asarray(mask.convert("RGB"), dtype=np.uint8)
    foreground = np.max(arr, axis=2) > black_threshold
    return Image.fromarray(np.where(foreground, 255, 0).astype(np.uint8), mode="L")


def load_mask(mask_path, mask_mode, black_threshold):
    mask = Image.open(mask_path).convert("L")
    if mask_mode in {"indexed_multiclass", "multiclass_index", "index"}:
        arr = np.asarray(mask, dtype=np.uint8)
        arr = np.where(arr > 5, 5, arr).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    return binarize_mask(mask, black_threshold=black_threshold)


def save_pair(image, mask, split_dir, stem, image_quality):
    split_dir.mkdir(parents=True, exist_ok=True)
    image.save(split_dir / f"{stem}.jpg", quality=image_quality, subsampling=1)
    mask.save(split_dir / f"{stem}_mask.png")


def resize_pair(image, mask, output_size):
    if not output_size or output_size <= 0:
        return image, mask
    size = (int(output_size), int(output_size))
    return (
        image.resize(size, Image.Resampling.BILINEAR),
        mask.resize(size, Image.Resampling.NEAREST),
    )


def copy_pair(image_path, mask_path, split_dir, stem):
    split_dir.mkdir(parents=True, exist_ok=True)
    image_out = split_dir / f"{stem}{image_path.suffix.lower()}"
    mask_out = split_dir / f"{stem}_mask.png"
    shutil.copy2(image_path, image_out)
    shutil.copy2(mask_path, mask_out)


def resize_crop_or_pad(image, mask, scale, rng):
    width, height = image.size
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)

    if scale >= 1.0:
        left = rng.randint(0, max(0, new_width - width))
        top = rng.randint(0, max(0, new_height - height))
        box = (left, top, left + width, top + height)
        return image.crop(box), mask.crop(box)

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    mask_canvas = Image.new("L", (width, height), 0)
    left = rng.randint(0, width - new_width)
    top = rng.randint(0, height - new_height)
    canvas.paste(image, (left, top))
    mask_canvas.paste(mask, (left, top))
    return canvas, mask_canvas


def random_geometry(image, mask, rng, rotate_deg, translate_frac, scale_range):
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)
    if rng.random() < 0.25:
        image = ImageOps.flip(image)
        mask = ImageOps.flip(mask)

    scale = rng.uniform(scale_range[0], scale_range[1])
    image, mask = resize_crop_or_pad(image, mask, scale, rng)

    width, height = image.size
    angle = rng.uniform(-rotate_deg, rotate_deg)
    translate = (
        int(round(rng.uniform(-translate_frac, translate_frac) * width)),
        int(round(rng.uniform(-translate_frac, translate_frac) * height)),
    )
    image = image.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        translate=translate,
        fillcolor=(0, 0, 0),
    )
    mask = mask.rotate(
        angle,
        resample=Image.Resampling.NEAREST,
        expand=False,
        translate=translate,
        fillcolor=0,
    )
    return image, mask


def random_photometric(image, rng):
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.35))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.75, 1.30))
    image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.75, 1.40))

    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.2)))
    if rng.random() < 0.45:
        arr = np.asarray(image, dtype=np.float32)
        noise = rng.uniform(3.0, 10.0)
        arr += np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0.0, noise, arr.shape)
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    return image


def mask_ratio(mask):
    arr = np.asarray(mask, dtype=np.uint8)
    return float(np.mean(arr > 0))


def make_augmented_pair(image, mask, rng, args):
    original_ratio = mask_ratio(mask)
    best_image = image
    best_mask = mask
    best_delta = math.inf

    for _ in range(args.max_retries):
        aug_image, aug_mask = random_geometry(
            image,
            mask,
            rng,
            rotate_deg=args.rotate_deg,
            translate_frac=args.translate_frac,
            scale_range=(args.min_scale, args.max_scale),
        )
        if args.mask_mode not in {"indexed_multiclass", "multiclass_index", "index"}:
            aug_mask = binarize_mask(aug_mask, black_threshold=args.black_threshold)
        ratio = mask_ratio(aug_mask)
        delta = abs(ratio - original_ratio)
        if delta < best_delta:
            best_image, best_mask, best_delta = aug_image, aug_mask, delta
        if original_ratio == 0.0 or ratio >= original_ratio * args.min_mask_keep:
            best_image, best_mask = aug_image, aug_mask
            break

    best_image = random_photometric(best_image, rng)
    return best_image, best_mask


def main():
    parser = argparse.ArgumentParser(description="Offline augmentation for split dataset_all corrosion datasets.")
    parser.add_argument("--input-root", default="dataset_all")
    parser.add_argument("--output-root", default="dataset_all_augmented")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--augment-splits", nargs="+", default=["train"])
    parser.add_argument("--copies", type=int, default=8, help="Augmented variants per original image.")
    parser.add_argument("--include-originals", action="store_true", default=True)
    parser.add_argument("--no-originals", dest="include_originals", action="store_false")
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=8,
        help="Pixels with all RGB channels <= this value are background; any brighter/color pixel is corrosion.",
    )
    parser.add_argument(
        "--mask-mode",
        default="binary",
        choices=["binary", "indexed_multiclass", "multiclass_index", "index"],
        help="indexed_multiclass preserves class IDs 0..5 with nearest-neighbor geometry.",
    )
    parser.add_argument("--rotate-deg", type=float, default=18.0)
    parser.add_argument("--translate-frac", type=float, default=0.08)
    parser.add_argument("--min-scale", type=float, default=0.85)
    parser.add_argument("--max-scale", type=float, default=1.18)
    parser.add_argument("--min-mask-keep", type=float, default=0.45)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--image-quality", type=int, default=95)
    parser.add_argument("--output-image-size", type=int, default=0, help="Optional square output size. 0 preserves source size.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    pairs = find_pairs(input_root, args.splits)
    augment_splits = set(args.augment_splits)

    count = 0
    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "splits": args.splits,
        "augment_splits": args.augment_splits,
        "copies": args.copies,
        "include_originals": args.include_originals,
        "mask_rule": "foreground = max(R, G, B) > black_threshold",
        "mask_mode": args.mask_mode,
        "black_threshold": args.black_threshold,
        "output_image_size": args.output_image_size,
        "items": [],
    }

    for split, image_path, mask_path in pairs:
        image = Image.open(image_path).convert("RGB")
        mask = load_mask(mask_path, args.mask_mode, args.black_threshold)
        image, mask = resize_pair(image, mask, args.output_image_size)
        split_dir = output_root / split
        base_stem = image_path.stem

        if split not in augment_splits:
            if args.output_image_size and args.output_image_size > 0:
                save_pair(image, mask, split_dir, base_stem, args.image_quality)
            else:
                copy_pair(image_path, mask_path, split_dir, base_stem)
            manifest["items"].append(
                {
                    "split": split,
                    "source_image": str(image_path),
                    "source_mask": str(mask_path),
                    "output_stem": base_stem,
                    "type": "copied",
                    "mask_area_ratio": mask_ratio(mask),
                }
            )
            count += 1
            continue

        if args.include_originals:
            save_pair(image, mask, split_dir, base_stem, args.image_quality)
            manifest["items"].append(
                {
                    "split": split,
                    "source_image": str(image_path),
                    "source_mask": str(mask_path),
                    "output_stem": base_stem,
                    "type": "original",
                    "mask_area_ratio": mask_ratio(mask),
                }
            )
            count += 1

        for idx in range(args.copies):
            aug_image, aug_mask = make_augmented_pair(image, mask, rng, args)
            stem = f"{base_stem}_aug{idx:03d}"
            save_pair(aug_image, aug_mask, split_dir, stem, args.image_quality)
            manifest["items"].append(
                {
                    "split": split,
                    "source_image": str(image_path),
                    "source_mask": str(mask_path),
                    "output_stem": stem,
                    "type": "augmented",
                    "mask_area_ratio": mask_ratio(aug_mask),
                }
            )
            count += 1

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "augmentation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Matched source pairs: {len(pairs)}")
    print(f"Generated pairs: {count}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
