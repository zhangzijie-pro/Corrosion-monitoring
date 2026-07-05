#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    cv2 = None
import numpy as np
from PIL import Image


IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")


def find_source_pairs(source_root, subsets):
    pairs = []
    missing = []
    for subset in subsets:
        raw_dir = source_root / subset / "raw"
        label_dir = source_root / subset / "labeled"
        raw_files = {}
        for ext in IMAGE_EXTS:
            raw_files.update({p.stem: p for p in raw_dir.glob(ext)})

        label_paths = []
        for ext in IMAGE_EXTS:
            label_paths.extend(label_dir.glob(ext))
        for label_path in sorted(label_paths):
            stem = label_path.stem.replace("_labels", "").replace("_label", "")
            image_path = raw_files.get(stem)
            if image_path is None:
                missing.append(str(label_path))
                continue
            pairs.append((subset, image_path, label_path))

    if missing:
        print(f"Warning: skipped {len(missing)} labels without matching raw images.")
    if not pairs:
        raise RuntimeError(f"No source image/mask pairs found under {source_root}")
    return pairs


def split_name(index, train_ratio, valid_ratio):
    if index < train_ratio:
        return "train"
    if index < train_ratio + valid_ratio:
        return "valid"
    return "test"


def foreground_from_label(mask_path, black_threshold=8):
    mask = Image.open(mask_path).convert("RGB")
    arr = np.asarray(mask, dtype=np.uint8)
    foreground = np.max(arr, axis=2) > black_threshold
    return foreground


def binarize_mask(mask_path, black_threshold=8):
    foreground = foreground_from_label(mask_path, black_threshold=black_threshold)
    return Image.fromarray(np.where(foreground, 255, 0).astype(np.uint8), mode="L")


def mask_area_ratio(mask):
    arr = np.asarray(mask, dtype=np.uint8)
    return float(np.mean(arr > 0))


def _resize_image_to_label(image, label_shape):
    target_h, target_w = label_shape
    if image.shape[:2] == (target_h, target_w):
        return image
    pil = Image.fromarray(image)
    return np.asarray(pil.resize((target_w, target_h), Image.Resampling.BILINEAR), dtype=np.uint8)


def _rgb_to_hsv_features(rgb):
    rgb_f = rgb.astype(np.float32) / 255.0
    if cv2 is not None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        hue = hsv[..., 0] / 179.0
        saturation = hsv[..., 1] / 255.0
        value = hsv[..., 2] / 255.0
        lab_a = lab[..., 1] / 255.0
        lab_b = lab[..., 2] / 255.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    else:
        hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32)
        hue = hsv[..., 0] / 255.0
        saturation = hsv[..., 1] / 255.0
        value = hsv[..., 2] / 255.0
        lab_a = np.clip((rgb_f[..., 0] - rgb_f[..., 1] + 1.0) * 0.5, 0.0, 1.0)
        lab_b = np.clip((0.5 * (rgb_f[..., 0] + rgb_f[..., 1]) - rgb_f[..., 2] + 1.0) * 0.5, 0.0, 1.0)
        gray = (0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]).astype(np.float32)
        gy, gx = np.gradient(gray)
        texture = np.sqrt(gx * gx + gy * gy)
    red = rgb_f[..., 0]
    green = rgb_f[..., 1]
    blue = rgb_f[..., 2]
    darkness = 1.0 - value
    red_brown = np.clip((red - 0.45 * green - 0.35 * blue + 0.2), 0.0, 1.0)
    texture = np.clip(texture / (np.percentile(texture, 99.0) + 1e-6), 0.0, 1.0)
    return np.stack([hue, saturation, value, darkness, red_brown, lab_a, lab_b, texture], axis=-1)


def _severity_score(features):
    # Features: hue, saturation, value, darkness, red_brown, lab_a, lab_b, texture.
    hue = features[..., 0]
    saturation = features[..., 1]
    darkness = features[..., 3]
    red_brown = features[..., 4]
    lab_a = features[..., 5]
    lab_b = features[..., 6]
    texture = features[..., 7]
    rust_hue = 1.0 - np.clip(np.abs(hue - 0.08) / 0.30, 0.0, 1.0)
    return (
        0.26 * darkness
        + 0.22 * saturation
        + 0.22 * red_brown
        + 0.12 * rust_hue
        + 0.08 * texture
        + 0.05 * lab_a
        + 0.05 * lab_b
    )


def _sample_foreground_features(pairs, black_threshold, max_pixels_per_image, seed):
    rng = np.random.default_rng(seed)
    samples = []
    per_image = []
    for subset, image_path, mask_path in pairs:
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        foreground = foreground_from_label(mask_path, black_threshold=black_threshold)
        image = _resize_image_to_label(image, foreground.shape)
        if not np.any(foreground):
            continue
        features = _rgb_to_hsv_features(image)[foreground]
        if features.shape[0] > max_pixels_per_image:
            indices = rng.choice(features.shape[0], size=max_pixels_per_image, replace=False)
            features = features[indices]
        samples.append(features.astype(np.float32))
        per_image.append(
            {
                "subset": subset,
                "image": str(image_path),
                "mask": str(mask_path),
                "sampled_pixels": int(features.shape[0]),
                "foreground_ratio": float(np.mean(foreground)),
            }
        )
    if not samples:
        raise RuntimeError("No foreground pixels found for HSI severity calibration.")
    return np.concatenate(samples, axis=0), per_image


def _numpy_kmeans(features, k, seed, iterations=50, retries=3):
    rng = np.random.default_rng(seed)
    best_labels = None
    best_centers = None
    best_compactness = np.inf
    n = features.shape[0]
    for _ in range(retries):
        centers = features[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=np.int32)
        for _iteration in range(iterations):
            distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            next_labels = np.argmin(distances, axis=1).astype(np.int32)
            if np.array_equal(next_labels, labels):
                break
            labels = next_labels
            for cluster in range(k):
                cluster_features = features[labels == cluster]
                if cluster_features.size:
                    centers[cluster] = cluster_features.mean(axis=0)
                else:
                    centers[cluster] = features[rng.integers(0, n)]
        compactness = float(np.sum((features - centers[labels]) ** 2))
        if compactness < best_compactness:
            best_compactness = compactness
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_compactness, best_labels.reshape(-1, 1), best_centers


def build_severity_calibration(pairs, black_threshold=8, num_classes=6, seed=42, max_pixels_per_image=30000):
    if int(num_classes) != 6:
        raise ValueError("hsi_severity currently expects --num-classes 6, with labels 0..5.")
    features, per_image = _sample_foreground_features(
        pairs,
        black_threshold=black_threshold,
        max_pixels_per_image=max_pixels_per_image,
        seed=seed,
    )
    k = num_classes - 1
    if features.shape[0] < k:
        raise RuntimeError(f"Not enough foreground pixels for {k} severity clusters.")
    features = features.astype(np.float32)
    if cv2 is not None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-4)
        cv2.setRNGSeed(int(seed))
        compactness, labels, centers = cv2.kmeans(
            features,
            k,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS,
        )
    else:
        compactness, labels, centers = _numpy_kmeans(features, k, seed=seed, iterations=50, retries=3)
    center_scores = _severity_score(centers)
    order = np.argsort(center_scores)
    cluster_to_class = {int(cluster): int(rank + 1) for rank, cluster in enumerate(order.tolist())}
    label_counts = np.bincount(labels.reshape(-1), minlength=k).astype(np.int64)
    class_counts = {str(cluster_to_class[i]): int(label_counts[i]) for i in range(k)}
    return {
        "method": "hsi_hsv_lab_texture_kmeans",
        "num_classes": int(num_classes),
        "black_threshold": int(black_threshold),
        "feature_names": ["hue", "saturation", "value", "darkness", "red_brown", "lab_a", "lab_b", "texture"],
        "cluster_centers": centers.astype(float).tolist(),
        "cluster_scores": center_scores.astype(float).tolist(),
        "cluster_to_class": {str(k): v for k, v in cluster_to_class.items()},
        "class_counts": class_counts,
        "sampled_images": per_image,
        "sampled_pixels": int(features.shape[0]),
        "compactness": float(compactness),
    }


def make_hsi_severity_mask(image_path, mask_path, calibration, black_threshold=8):
    foreground = foreground_from_label(mask_path, black_threshold=black_threshold)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    image = _resize_image_to_label(image, foreground.shape)
    output = np.zeros(foreground.shape, dtype=np.uint8)
    if not np.any(foreground):
        return Image.fromarray(output, mode="L")
    ys, xs = np.where(foreground)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    image_crop = image[top:bottom, left:right]
    foreground_crop = foreground[top:bottom, left:right]
    features = _rgb_to_hsv_features(image_crop)
    centers = np.asarray(calibration["cluster_centers"], dtype=np.float32)
    fg_features = features[foreground_crop].astype(np.float32)
    distances = np.sum((fg_features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    clusters = np.argmin(distances, axis=1)
    cluster_to_class = {int(k): int(v) for k, v in calibration["cluster_to_class"].items()}
    classes = np.asarray([cluster_to_class[int(cluster)] for cluster in clusters], dtype=np.uint8)
    output_crop = output[top:bottom, left:right]
    output_crop[foreground_crop] = classes
    output[top:bottom, left:right] = output_crop
    return Image.fromarray(output, mode="L")


def safe_copy_pair(image_path, mask_path, output_dir, stem, black_threshold, overwrite, mask_mode, calibration):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_out = output_dir / f"{stem}.jpg"
    mask_out = output_dir / f"{stem}_mask.png"

    if not overwrite and (image_out.exists() or mask_out.exists()):
        return False, None

    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(image_path, image_out)
    else:
        Image.open(image_path).convert("RGB").save(image_out, quality=95, subsampling=1)

    if mask_mode == "hsi_severity":
        if calibration is None:
            raise RuntimeError("Missing severity calibration for hsi_severity mask mode.")
        mask = make_hsi_severity_mask(
            image_path=image_path,
            mask_path=mask_path,
            calibration=calibration,
            black_threshold=black_threshold,
        )
    else:
        mask = binarize_mask(mask_path, black_threshold=black_threshold)
    area_ratio = mask_area_ratio(mask)
    mask.save(mask_out)
    return True, area_ratio


def main():
    parser = argparse.ArgumentParser(description="Append dataset/{HiRes,LoRes} pairs into dataset_all.")
    parser.add_argument("--source-root", default="dataset")
    parser.add_argument("--target-root", default="dataset_all")
    parser.add_argument("--subsets", nargs="+", default=["HiRes", "LoRes"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=8,
        help="Pixels with all RGB channels <= this value are background; any brighter/color pixel is corrosion.",
    )
    parser.add_argument(
        "--mask-mode",
        default="binary",
        choices=["binary", "hsi_severity"],
        help="binary writes 0/255 masks; hsi_severity writes indexed 0..5 masks.",
    )
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--max-calibration-pixels-per-image", type=int, default=1500)
    parser.add_argument("--prefix", default="mavecodd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/valid/test ratios must sum to 1.0")

    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    pairs = find_source_pairs(source_root, args.subsets)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    calibration = None
    if args.mask_mode == "hsi_severity":
        calibration = build_severity_calibration(
            pairs,
            black_threshold=args.black_threshold,
            num_classes=args.num_classes,
            seed=args.seed,
            max_pixels_per_image=args.max_calibration_pixels_per_image,
        )

    manifest = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "seed": args.seed,
        "mask_rule": "foreground = max(R, G, B) > black_threshold",
        "mask_mode": args.mask_mode,
        "num_classes": args.num_classes if args.mask_mode == "hsi_severity" else 2,
        "black_threshold": args.black_threshold,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "test_ratio": args.test_ratio,
        "items": [],
    }
    copied = 0
    skipped = 0

    for subset, image_path, mask_path in pairs:
        split = split_name(rng.random(), args.train_ratio, args.valid_ratio)
        stem = f"{args.prefix}_{subset}_{image_path.stem}"
        target_dir = target_root / split
        did_copy, mask_area_ratio_value = safe_copy_pair(
            image_path=image_path,
            mask_path=mask_path,
            output_dir=target_dir,
            stem=stem,
            black_threshold=args.black_threshold,
            overwrite=args.overwrite,
            mask_mode=args.mask_mode,
            calibration=calibration,
        )
        if did_copy:
            copied += 1
        else:
            skipped += 1
        manifest["items"].append(
            {
                "split": split,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "target_image": str(target_dir / f"{stem}.jpg"),
                "target_mask": str(target_dir / f"{stem}_mask.png"),
                "mask_area_ratio": mask_area_ratio_value,
                "copied": did_copy,
            }
        )

    target_root.mkdir(parents=True, exist_ok=True)
    if calibration is not None:
        with open(target_root / "severity_calibration.json", "w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2, ensure_ascii=False)
        manifest["severity_calibration"] = str(target_root / "severity_calibration.json")
    with open(target_root / "added_from_dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Matched source pairs: {len(pairs)}")
    print(f"Copied pairs: {copied}")
    print(f"Skipped existing pairs: {skipped}")
    print(f"Target root: {target_root}")


if __name__ == "__main__":
    main()
