import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def rgb_severity_classes(image, foreground):
    """Map foreground corrosion pixels to 5 severity classes from image RGB.

    Class 0 is background. Classes 1-5 are increasing corrosion severity.
    The score favors dark, saturated, red/brown pixels, which usually indicate
    heavier rust/corrosion than bright paint loss or light staining.
    """
    rgb = image.astype(np.float32)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)
    value = max_channel
    saturation = np.where(max_channel > 1e-6, (max_channel - min_channel) / np.maximum(max_channel, 1e-6), 0.0)
    red_dominance = np.clip((red - 0.5 * (green + blue) + 1.0) / 2.0, 0.0, 1.0)
    darkness = 1.0 - value

    severity = 0.45 * darkness + 0.35 * saturation + 0.20 * red_dominance
    bins = np.array([0.20, 0.38, 0.54, 0.70], dtype=np.float32)
    classes = np.zeros(foreground.shape, dtype=np.int64)
    classes[foreground] = np.digitize(severity[foreground], bins, right=False) + 1
    return classes


def find_dataset_all_pairs(root, splits):
    pairs = []
    missing = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(split_dir.glob(ext))
        for image_path in sorted(image_paths):
            if image_path.stem.endswith("_mask"):
                continue
            mask_path = split_dir / f"{image_path.stem}_mask.png"
            if mask_path.exists():
                pairs.append((image_path, mask_path))
            else:
                missing.append(str(image_path))
    if missing:
        print(f"Warning: skipped {len(missing)} images without matching *_mask.png labels.")
    return pairs


def find_source_pairs(root, subsets):
    pairs = []
    for subset in subsets:
        raw_dir = root / subset / "raw"
        label_dir = root / subset / "labeled"
        raw_files = {}
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            raw_files.update({p.stem: p for p in raw_dir.glob(ext)})
        label_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            label_paths.extend(label_dir.glob(ext))
        for label_path in sorted(label_paths):
            stem = label_path.stem.replace("_labels", "").replace("_label", "")
            image_path = raw_files.get(stem)
            if image_path is not None:
                pairs.append((image_path, label_path))
    return pairs


def split_pairs(pairs, val_ratio, seed):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    val_count = max(1, int(round(len(pairs) * val_ratio))) if len(pairs) > 1 else 0
    return pairs[val_count:], pairs[:val_count]


def load_data_pairs(data_cfg, seed=42):
    root = Path(data_cfg["root"])
    layout = data_cfg.get("layout", "dataset_all")
    if layout == "dataset_all":
        train_pairs = find_dataset_all_pairs(root, data_cfg.get("train_splits", ["train"]))
        val_pairs = find_dataset_all_pairs(root, data_cfg.get("val_splits", ["valid"]))
    else:
        pairs = find_source_pairs(root, data_cfg.get("subsets", ["HiRes", "LoRes"]))
        train_pairs, val_pairs = split_pairs(pairs, data_cfg.get("val_ratio", 0.2), seed)

    max_samples = data_cfg.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        train_pairs = train_pairs[: max(1, max_samples - 1)]
        val_pairs = val_pairs[: 1 if max_samples > 1 else 0]
    if not train_pairs:
        raise RuntimeError(f"No training image/mask pairs found under {root}")
    if not val_pairs:
        raise RuntimeError(f"No validation image/mask pairs found under {root}")
    return train_pairs, val_pairs


class CorrosionDataset(Dataset):
    def __init__(self, pairs, image_size, mask_threshold=127, mask_mode="binary", augment=False):
        self.pairs = list(pairs)
        self.image_size = tuple(image_size)
        self.mask_threshold = mask_threshold
        self.mask_mode = str(mask_mode).lower()
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        mask = mask.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
        image = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = np.asarray(mask, dtype=np.uint8)
        foreground = mask_arr > self.mask_threshold
        if self.mask_mode in {"indexed_multiclass", "multiclass_index", "index"}:
            mask = np.clip(mask_arr, 0, 5).astype(np.int64)
        elif self.mask_mode in {"rgb_severity", "multiclass", "multi_class"}:
            mask = rgb_severity_classes(image, foreground)
        else:
            mask = foreground.astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.2:
                image = np.ascontiguousarray(image[::-1])
                mask = np.ascontiguousarray(mask[::-1])
            if random.random() < 0.5:
                image = np.clip(image * random.uniform(0.85, 1.15) + random.uniform(-0.06, 0.06), 0.0, 1.0)

        image = torch.from_numpy(image).permute(2, 0, 1)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        if self.mask_mode in {"rgb_severity", "multiclass", "multi_class", "indexed_multiclass", "multiclass_index", "index"}:
            mask = torch.from_numpy(mask).long()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0)
        return {"image": image, "mask": mask, "path": str(image_path)}


def build_dataloaders(cfg, seed, device):
    data_cfg = cfg["data"]
    train_pairs, val_pairs = load_data_pairs(data_cfg, seed=seed)
    train_set = CorrosionDataset(
        train_pairs,
        data_cfg.get("image_size", [512, 512]),
        mask_threshold=data_cfg.get("mask_threshold", 127),
        mask_mode=data_cfg.get("mask_mode", "binary"),
        augment=True,
    )
    val_set = CorrosionDataset(
        val_pairs,
        data_cfg.get("image_size", [512, 512]),
        mask_threshold=data_cfg.get("mask_threshold", 127),
        mask_mode=data_cfg.get("mask_mode", "binary"),
        augment=False,
    )
    train_cfg = cfg["train"]
    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    return train_loader, val_loader
