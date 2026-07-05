#!/usr/bin/env python3
import argparse
import json
import random
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_name(value, train_ratio, valid_ratio):
    if value < train_ratio:
        return "train"
    if value < train_ratio + valid_ratio:
        return "valid"
    return "test"


def iter_local_images(source_dir):
    source_dir = Path(source_dir)
    if not source_dir.exists():
        return []
    return [
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def download_file(url, path, timeout=60):
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(path, "wb") as out:
        shutil.copyfileobj(response, out)


def ensure_coco_val(cache_dir):
    zip_path = cache_dir / "val2017.zip"
    extract_dir = cache_dir / "val2017"
    if extract_dir.exists() and any(extract_dir.glob("*.jpg")):
        return extract_dir
    if not zip_path.exists():
        print(f"Downloading COCO val2017 from {COCO_VAL_URL}")
        download_file(COCO_VAL_URL, zip_path, timeout=180)
    shutil.unpack_archive(str(zip_path), str(cache_dir))
    return extract_dir


def fallback_urls(count, seed):
    rng = random.Random(seed)
    return [f"https://picsum.photos/seed/negative-{seed}-{rng.randint(1, 10_000_000)}/640/480" for _ in range(count)]


def collect_source_images(cache_dir, count, seed, local_source=None, source="coco"):
    if local_source:
        images = iter_local_images(local_source)
        if images:
            return [{"type": "local", "path": str(path), "url": None} for path in images[:count]]
    elif source == "local":
        return []

    if source == "coco":
        try:
            coco_dir = ensure_coco_val(cache_dir)
            images = iter_local_images(coco_dir)
            if images:
                return [{"type": "local", "path": str(path), "url": COCO_VAL_URL} for path in images[:count]]
        except Exception as exc:
            print(f"Warning: COCO download failed, falling back to random public image URLs: {exc}")

    downloads_dir = cache_dir / "fallback_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for idx, url in enumerate(fallback_urls(count, seed)):
        path = downloads_dir / f"fallback_{idx:04d}.jpg"
        if not path.exists():
            try:
                download_file(url, path, timeout=30)
                time.sleep(0.15)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"Warning: skipped fallback image {url}: {exc}")
                continue
        items.append({"type": "download", "path": str(path), "url": url})
        if len(items) >= count:
            break
    return items


def save_negative_pair(source_path, target_dir, stem, image_size=None):
    target_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(source_path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    image_out = target_dir / f"{stem}.jpg"
    mask_out = target_dir / f"{stem}_mask.png"
    image.save(image_out, quality=95, subsampling=1)
    Image.new("L", image.size, 0).save(mask_out)
    return image_out, mask_out, image.size


def main():
    parser = argparse.ArgumentParser(description="Add all-background negative images to dataset_all_augmented.")
    parser.add_argument("--target-root", default="dataset_all_augmented")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--cache-dir", default="external_negative_cache")
    parser.add_argument("--local-source", help="Optional local image folder to use before downloading COCO.")
    parser.add_argument("--source", choices=["coco", "random", "local"], default="coco")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--prefix", default="negative")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=0, help="Optional square resize size; 0 preserves source size.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/valid/test ratios must sum to 1.0")

    target_root = Path(args.target_root)
    cache_dir = Path(args.cache_dir)
    rng = random.Random(args.seed)
    sources = collect_source_images(cache_dir, args.count, args.seed, local_source=args.local_source, source=args.source)
    if not sources:
        raise RuntimeError("No negative sample images could be collected.")
    rng.shuffle(sources)
    sources = sources[: args.count]
    image_size = [args.image_size, args.image_size] if args.image_size and args.image_size > 0 else None

    manifest = {
        "target_root": str(target_root),
        "count_requested": int(args.count),
        "count_collected": len(sources),
        "seed": int(args.seed),
        "mask_rule": "all pixels are background class 0",
        "items": [],
    }
    copied = 0
    skipped = 0
    for idx, item in enumerate(sources):
        split = split_name(rng.random(), args.train_ratio, args.valid_ratio)
        target_dir = target_root / split
        stem = f"{args.prefix}_{idx:04d}"
        image_out = target_dir / f"{stem}.jpg"
        mask_out = target_dir / f"{stem}_mask.png"
        if not args.overwrite and (image_out.exists() or mask_out.exists()):
            skipped += 1
            manifest["items"].append({**item, "split": split, "target_image": str(image_out), "target_mask": str(mask_out), "copied": False})
            continue
        try:
            image_out, mask_out, size = save_negative_pair(item["path"], target_dir, stem, image_size=image_size)
        except Exception as exc:
            skipped += 1
            manifest["items"].append({**item, "split": split, "error": str(exc), "copied": False})
            continue
        copied += 1
        manifest["items"].append(
            {
                **item,
                "split": split,
                "target_image": str(image_out),
                "target_mask": str(mask_out),
                "width": int(size[0]),
                "height": int(size[1]),
                "copied": True,
            }
        )

    target_root.mkdir(parents=True, exist_ok=True)
    with open(target_root / "negative_samples_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Collected negative sources: {len(sources)}")
    print(f"Copied negative pairs: {copied}")
    print(f"Skipped negative pairs: {skipped}")
    print(f"Target root: {target_root}")


if __name__ == "__main__":
    main()
