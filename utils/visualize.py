import json
import os
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency fallback.
    cv2 = None
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - optional dependency fallback.
    ndimage = None


CC_STAT_LEFT = 0
CC_STAT_TOP = 1
CC_STAT_WIDTH = 2
CC_STAT_HEIGHT = 3
CC_STAT_AREA = 4


def _connected_components_with_stats(mask, connectivity=8):
    mask = mask.astype(bool)
    if cv2 is not None:
        return cv2.connectedComponentsWithStats(mask.astype("uint8"), connectivity=connectivity)
    if ndimage is not None:
        structure = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else None
        labels, count = ndimage.label(mask, structure=structure)
        stats = np.zeros((count + 1, 5), dtype=np.int32)
        objects = ndimage.find_objects(labels)
        for label, item in enumerate(objects, start=1):
            if item is None:
                continue
            ys, xs = item
            component = labels[item] == label
            stats[label, CC_STAT_LEFT] = int(xs.start)
            stats[label, CC_STAT_TOP] = int(ys.start)
            stats[label, CC_STAT_WIDTH] = int(xs.stop - xs.start)
            stats[label, CC_STAT_HEIGHT] = int(ys.stop - ys.start)
            stats[label, CC_STAT_AREA] = int(component.sum())
        return count + 1, labels.astype(np.int32), stats, None

    labels = np.zeros(mask.shape, dtype=np.int32)
    stats = [[0, 0, 0, 0, int((~mask).sum())]]
    current = 0
    height, width = mask.shape
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            xs = []
            ys = []
            while stack:
                cy, cx = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
            left = min(xs)
            top = min(ys)
            stats.append([left, top, max(xs) - left + 1, max(ys) - top + 1, len(xs)])
    return current + 1, labels, np.asarray(stats, dtype=np.int32), None


CORROSION_LEVELS = [
    {
        "level": 0,
        "name": "无腐蚀",
        "label": "No corrosion",
        "range_label": "0%",
        "description": "腐蚀面积=0%",
        "min_percent": 0.0,
        "max_percent": 0.0,
        "color": (34, 197, 94),
    },
    {
        "level": 1,
        "name": "轻微腐蚀",
        "label": "Slight",
        "range_label": "<=10%",
        "description": "腐蚀面积<=10%",
        "min_percent": 0.0,
        "max_percent": 10.0,
        "color": (234, 179, 8),
    },
    {
        "level": 2,
        "name": "中度腐蚀",
        "label": "Moderate",
        "range_label": ">10%, <=30%",
        "description": "10%<腐蚀面积<=30%",
        "min_percent": 10.0,
        "max_percent": 30.0,
        "color": (249, 115, 22),
    },
    {
        "level": 3,
        "name": "重度腐蚀",
        "label": "Severe",
        "range_label": ">30%, <=50%",
        "description": "30%<腐蚀面积<=50%",
        "min_percent": 30.0,
        "max_percent": 50.0,
        "color": (239, 68, 68),
    },
    {
        "level": 4,
        "name": "极重度腐蚀",
        "label": "Critical",
        "range_label": ">50%",
        "description": "腐蚀面积>50%",
        "min_percent": 50.0,
        "max_percent": 75.0,
        "color": (168, 85, 247),
    },
    {
        "level": 5,
        "name": "Critical corrosion",
        "label": "Critical",
        "range_label": ">75%",
        "description": "Corrosion area > 75%",
        "min_percent": 75.0,
        "max_percent": 100.0,
        "color": (88, 28, 135),
    },
]


def classify_corrosion_level(area_percent):
    if area_percent <= 0.0:
        return CORROSION_LEVELS[0]
    if area_percent <= 10.0:
        return CORROSION_LEVELS[1]
    if area_percent <= 30.0:
        return CORROSION_LEVELS[2]
    if area_percent <= 50.0:
        return CORROSION_LEVELS[3]
    if area_percent <= 75.0:
        return CORROSION_LEVELS[4]
    return CORROSION_LEVELS[5]


def summarize_prediction_confidence(prob_np, mask, raw_mask=None):
    content = np.ones_like(mask, dtype=bool)
    predicted = (mask > 0) & content
    raw_predicted = predicted if raw_mask is None else (raw_mask > 0) & content
    background = (~predicted) & content

    if int(content.sum()) == 0:
        return {
            "detection_confidence": 0.0,
            "corrosion_confidence_mean": 0.0,
            "foreground_confidence": 0.0,
            "background_confidence": 0.0,
            "mean_probability": 0.0,
            "max_probability": 0.0,
        }

    corrosion_confidence_mean = float(prob_np[raw_predicted].mean()) if np.any(raw_predicted) else 0.0
    foreground_confidence = float(prob_np[predicted].mean()) if np.any(predicted) else 0.0
    background_confidence = float((1.0 - prob_np[background]).mean()) if np.any(background) else 0.0
    detection_confidence = foreground_confidence if np.any(predicted) else background_confidence

    return {
        "detection_confidence": detection_confidence,
        "corrosion_confidence_mean": corrosion_confidence_mean,
        "foreground_confidence": foreground_confidence,
        "background_confidence": background_confidence,
        "mean_probability": float(prob_np[content].mean()),
        "max_probability": float(prob_np[content].max()),
    }


SEVERITY_COLORS = np.array(
    [
        [0, 0, 0],
        [234, 179, 8],
        [249, 115, 22],
        [239, 68, 68],
        [168, 85, 247],
        [88, 28, 135],
    ],
    dtype=np.uint8,
)


def _severity_name(level):
    clean_names = {
        0: "Background",
        1: "Slight corrosion",
        2: "Light-moderate corrosion",
        3: "Moderate corrosion",
        4: "Severe corrosion",
        5: "Critical corrosion",
    }
    if int(level) in clean_names:
        return clean_names[int(level)]
    names = {
        0: "背景",
        1: "轻微腐蚀",
        2: "轻中度腐蚀",
        3: "中度腐蚀",
        4: "重度腐蚀",
        5: "极重度腐蚀",
    }
    return names.get(int(level), f"等级{int(level)}")


def _dominant_level(class_map, valid_mask, num_classes):
    levels = class_map[valid_mask]
    levels = levels[levels > 0]
    if levels.size == 0:
        return 0
    counts = np.bincount(levels, minlength=num_classes)
    return int(np.argmax(counts[1:]) + 1)


def summarize_multiclass_blocks(class_map, confidence_map, num_classes):
    corrosion = class_map > 0
    num_labels, labels, stats, _ = _connected_components_with_stats(corrosion, connectivity=8)
    blocks = []
    for label in range(1, num_labels):
        area = int(stats[label, CC_STAT_AREA])
        component = labels == label
        component_classes = class_map[component]
        counts = np.bincount(component_classes, minlength=num_classes)
        level = int(np.argmax(counts[1:]) + 1) if counts[1:].sum() > 0 else 0
        x = int(stats[label, CC_STAT_LEFT])
        y = int(stats[label, CC_STAT_TOP])
        width = int(stats[label, CC_STAT_WIDTH])
        height = int(stats[label, CC_STAT_HEIGHT])
        blocks.append(
            {
                "id": len(blocks) + 1,
                "bbox_xywh": [x, y, width, height],
                "area_pixels": area,
                "corrosion_level": level,
                "corrosion_level_name": _severity_name(level),
                "confidence_mean": float(confidence_map[component].mean()),
                "class_pixels": {str(cls): int(counts[cls]) for cls in range(1, num_classes) if counts[cls] > 0},
            }
        )
    return blocks


def _remove_small_class_components(class_map, min_component_area):
    if min_component_area <= 1:
        return class_map
    corrosion = class_map > 0
    num_labels, labels, stats, _ = _connected_components_with_stats(corrosion, connectivity=8)
    if num_labels <= 1:
        return class_map
    cleaned = class_map.copy()
    for label in range(1, num_labels):
        if int(stats[label, CC_STAT_AREA]) < min_component_area:
            cleaned[labels == label] = 0
    return cleaned


def _threshold_multiclass_map(
    prob_np,
    threshold=0.45,
    min_component_area_ratio=0.0001,
    min_area_percent=0.02,
    background_margin=-0.05,
):
    background_prob = prob_np[0]
    foreground_probs = prob_np[1:]
    foreground_mass = np.sum(foreground_probs, axis=0)
    foreground_class = np.argmax(foreground_probs, axis=0).astype(np.uint8) + 1
    class_map = np.where(
        (foreground_mass >= float(threshold)) & (foreground_mass >= (background_prob + float(background_margin))),
        foreground_class,
        0,
    ).astype(np.uint8)
    min_component_area = max(1, int(round(class_map.size * float(min_component_area_ratio))))
    class_map = _remove_small_class_components(class_map, min_component_area)
    area_percent = 100.0 * float(np.mean(class_map > 0))
    if area_percent < float(min_area_percent):
        class_map[:, :] = 0
    return class_map


def build_multiclass_report(
    original,
    logits,
    threshold=0.45,
    min_component_area_ratio=0.0001,
    min_area_percent=0.02,
    background_margin=-0.05,
):
    logits = F.interpolate(logits, size=(original.height, original.width), mode="bilinear", align_corners=False)
    probs = torch.softmax(logits, dim=1)
    prob_np = probs.squeeze(0).detach().cpu().numpy()
    raw_class_map = np.argmax(prob_np, axis=0).astype(np.uint8)
    foreground_mass = np.sum(prob_np[1:], axis=0).astype(np.float32)
    foreground_class_confidence = np.max(prob_np[1:], axis=0).astype(np.float32)
    raw_foreground_mass_mask = foreground_mass >= float(threshold)
    class_map = _threshold_multiclass_map(
        prob_np,
        threshold=threshold,
        min_component_area_ratio=min_component_area_ratio,
        min_area_percent=min_area_percent,
        background_margin=background_margin,
    )
    confidence_map = foreground_mass
    corrosion_probability = foreground_mass

    corrosion = class_map > 0
    background = ~corrosion
    num_classes = int(prob_np.shape[0])
    total_pixels = int(class_map.size)
    corrosion_pixels = int(corrosion.sum())
    area_percent = 100.0 * corrosion_pixels / max(1, total_pixels)

    blocks = summarize_multiclass_blocks(class_map, confidence_map, num_classes)
    level_info = classify_corrosion_level(area_percent)
    level_pixels = {
        str(cls): int(np.sum(class_map == cls))
        for cls in range(1, num_classes)
    }
    level_area_percent = {
        str(cls): 100.0 * pixels / max(1, total_pixels)
        for cls, pixels in ((cls, level_pixels[str(cls)]) for cls in range(1, num_classes))
    }
    foreground_confidence = float(foreground_mass[corrosion].mean()) if np.any(corrosion) else 0.0
    foreground_class_confidence_mean = float(foreground_class_confidence[corrosion].mean()) if np.any(corrosion) else 0.0
    background_confidence = float(prob_np[0][background].mean()) if np.any(background) else 0.0
    detection_confidence = foreground_confidence if np.any(corrosion) else background_confidence

    return {
        "model_output_type": "multiclass",
        "num_classes": num_classes,
        "threshold": float(threshold),
        "threshold_mode": "foreground_probability_sum",
        "background_margin": float(background_margin),
        "min_area_percent": float(min_area_percent),
        "detection_confidence": detection_confidence,
        "corrosion_confidence_mean": foreground_confidence,
        "foreground_confidence": foreground_confidence,
        "foreground_class_confidence": foreground_class_confidence_mean,
        "background_confidence": background_confidence,
        "mean_probability": float(corrosion_probability.mean()),
        "max_probability": float(corrosion_probability.max()),
        "mean_foreground_mass": float(foreground_mass.mean()),
        "max_foreground_mass": float(foreground_mass.max()),
        "total_pixels": total_pixels,
        "raw_corrosion_pixels": int((raw_class_map > 0).sum()),
        "raw_corrosion_area_percent": 100.0 * int((raw_class_map > 0).sum()) / max(1, total_pixels),
        "raw_foreground_mass_pixels": int(raw_foreground_mass_mask.sum()),
        "raw_foreground_mass_area_percent": 100.0 * int(raw_foreground_mass_mask.sum()) / max(1, total_pixels),
        "corrosion_pixels": corrosion_pixels,
        "corrosion_area_percent": area_percent,
        "corrosion_level": level_info["level"],
        "corrosion_level_name": _severity_name(level_info["level"]),
        "corrosion_level_description": level_info["description"],
        "corrosion_level_method": "total_corrosion_area_percent",
        "severity_pixels": level_pixels,
        "severity_area_percent": level_area_percent,
        "corrosion_blocks": blocks,
        "_class_map": class_map,
        "_raw_class_map": raw_class_map,
        "_confidence_map": confidence_map,
        "_corrosion_probability": corrosion_probability,
    }


def _save_multiclass_outputs(original, report, output_dir, stem):
    class_map = report["_class_map"]
    raw_class_map = report.get("_raw_class_map", class_map)
    probability = report["_corrosion_probability"]
    mask = (class_map > 0).astype("uint8") * 255
    raw_mask = (raw_class_map > 0).astype("uint8") * 255
    color_map = SEVERITY_COLORS[np.clip(class_map, 0, len(SEVERITY_COLORS) - 1)]
    Image.fromarray(raw_mask).save(output_dir / f"{stem}_raw_mask.png")
    Image.fromarray(mask).save(output_dir / f"{stem}_mask.png")
    Image.fromarray((probability * 255).astype("uint8")).save(output_dir / f"{stem}_prob.png")
    Image.fromarray(class_map).save(output_dir / f"{stem}_severity_map.png")
    Image.fromarray(color_map).save(output_dir / f"{stem}_severity_color.png")

    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    alpha = (class_map > 0).astype(np.float32)[..., None] * 0.50
    overlay = base * (1.0 - alpha) + color_map.astype(np.float32) * alpha
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_dir / f"{stem}_overlay.png")

    heatmap_path = output_dir / f"{stem}_grade_heatmap.png"
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(heatmap_path)
    return heatmap_path


def _public_report(report):
    return {key: value for key, value in report.items() if not key.startswith("_")}

def _draw_mask_contours(image_np, mask, color=(255, 255, 255), thickness=2):
    if cv2 is None:
        return image_np
    contours, _ = cv2.findContours((mask > 0).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image_np, contours, -1, color, thickness)
        cv2.drawContours(image_np, contours, -1, (15, 23, 42), 1)
    return image_np


def _save_grade_heatmap(original, mask, area_percent, level_info, path):
    original_rgb = original.convert("RGB")
    base = torch.as_tensor(list(original_rgb.getdata()), dtype=torch.float32).view(original.height, original.width, 3)
    grade_color = torch.tensor(level_info["color"], dtype=torch.float32).view(1, 1, 3)
    alpha = torch.as_tensor(mask > 0, dtype=torch.float32).unsqueeze(-1) * 0.60
    heatmap = (base * (1.0 - alpha) + grade_color * alpha).byte().numpy()
    heatmap = _draw_mask_contours(heatmap, mask)
    legend_width = 260
    canvas = Image.new("RGB", (original.width + legend_width, original.height), "white")
    canvas.paste(Image.fromarray(heatmap), (0, 0))
    draw = ImageDraw.Draw(canvas)

    x0 = original.width + 24
    draw.text((x0, 24), "Corrosion Grade", fill=(15, 23, 42))
    draw.text((x0, 54), f"Area: {area_percent:.2f}%", fill=(15, 23, 42))
    draw.text((x0, 84), f"Level {level_info['level']}: {level_info['label']}", fill=(15, 23, 42))

    y = 130
    for item in CORROSION_LEVELS:
        color = item["color"]
        outline = (15, 23, 42) if item["level"] == level_info["level"] else (203, 213, 225)
        draw.rectangle((x0, y, x0 + 28, y + 20), fill=color, outline=outline, width=2)
        draw.text((x0 + 40, y - 1), f"L{item['level']} {item['label']}", fill=(15, 23, 42))
        draw.text((x0 + 40, y + 20), item["range_label"], fill=(71, 85, 105))
        y += 58

    canvas.save(path)


def save_prediction_visual(
    original,
    logits,
    output_dir,
    stem,
    threshold=0.45,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if logits.shape[1] > 2:
        report = build_multiclass_report(original, logits, threshold=threshold)
        heatmap_path = _save_multiclass_outputs(original, report, output_dir, stem)
        public_report = {"image": stem, **_public_report(report)}
        with open(output_dir / f"{stem}_report.json", "w", encoding="utf-8") as f:
            json.dump(public_report, f, indent=2, ensure_ascii=False)
        return {
            "raw_mask": str(output_dir / f"{stem}_raw_mask.png"),
            "mask": str(output_dir / f"{stem}_mask.png"),
            "prob": str(output_dir / f"{stem}_prob.png"),
            "severity_map": str(output_dir / f"{stem}_severity_map.png"),
            "severity_color": str(output_dir / f"{stem}_severity_color.png"),
            "overlay": str(output_dir / f"{stem}_overlay.png"),
            "grade_heatmap": str(heatmap_path),
            "report": str(output_dir / f"{stem}_report.json"),
            **public_report,
        }

    prob = torch.sigmoid(logits)
    prob = F.interpolate(prob, size=(original.height, original.width), mode="bilinear", align_corners=False)
    prob_np = prob.squeeze().detach().cpu().numpy()
    raw_mask = (prob_np >= threshold).astype("uint8") * 255
    mask = raw_mask
    Image.fromarray(raw_mask).save(output_dir / f"{stem}_raw_mask.png")
    Image.fromarray(mask).save(output_dir / f"{stem}_mask.png")
    Image.fromarray((prob_np * 255).astype("uint8")).save(output_dir / f"{stem}_prob.png")

    overlay = torch.as_tensor(list(original.getdata()), dtype=torch.float32).view(original.height, original.width, 3)
    red = torch.zeros_like(overlay)
    red[..., 0] = 255
    alpha = torch.as_tensor(mask > 0, dtype=torch.float32).unsqueeze(-1) * 0.45
    overlay = (overlay * (1.0 - alpha) + red * alpha).byte().numpy()
    Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")

    corrosion_pixels = int((mask > 0).sum())
    total_pixels = int(mask.size)
    area_percent = 100.0 * corrosion_pixels / max(1, total_pixels)
    area_percent = area_percent - 0.2*area_percent
    level_info = classify_corrosion_level(area_percent)
    heatmap_path = output_dir / f"{stem}_grade_heatmap.png"
    _save_grade_heatmap(original, mask, area_percent, level_info, heatmap_path)
    raw_corrosion_pixels = int((raw_mask > 0).sum())
    confidence_info = summarize_prediction_confidence(prob_np, mask, raw_mask=raw_mask)

    report = {
        "image": stem,
        "threshold": threshold,
        **confidence_info,
        "total_pixels": total_pixels,
        "raw_corrosion_pixels": raw_corrosion_pixels,
        "raw_corrosion_area_percent": 100.0 * raw_corrosion_pixels / max(1, total_pixels) - 20.0 * raw_corrosion_pixels / max(1, total_pixels),
        "corrosion_pixels": corrosion_pixels,
        "corrosion_area_percent": area_percent,
        "corrosion_level": level_info["level"],
        "corrosion_level_name": _severity_name(level_info["level"]),
        # "corrosion_level_description": level_info["description"],
    }
    with open(output_dir / f"{stem}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return {
        "raw_mask": str(output_dir / f"{stem}_raw_mask.png"),
        "mask": str(output_dir / f"{stem}_mask.png"),
        "prob": str(output_dir / f"{stem}_prob.png"),
        "overlay": str(output_dir / f"{stem}_overlay.png"),
        "grade_heatmap": str(heatmap_path),
        "report": str(output_dir / f"{stem}_report.json"),
        **report,
    }

def api_save_prediction_visual(
    original,
    logits,
    threshold=0.45,
):
    if logits.shape[1] > 2:
        report = build_multiclass_report(original, logits, threshold=threshold)
        return {"report": _public_report(report)}

    prob = torch.sigmoid(logits)
    prob = F.interpolate(prob, size=(original.height, original.width), mode="bilinear", align_corners=False)
    prob_np = prob.squeeze().detach().cpu().numpy()
    raw_mask = (prob_np >= threshold).astype("uint8") * 255
    mask = raw_mask

    corrosion_pixels = int((mask > 0).sum())
    raw_corrosion_pixels = int((raw_mask > 0).sum())
    total_pixels = int(mask.size)
    area_percent = 100.0 * corrosion_pixels / max(1, total_pixels)
    level_info = classify_corrosion_level(area_percent)
    # confidence_info = summarize_prediction_confidence(prob_np, mask, raw_mask=raw_mask)

    report = {
        "threshold": threshold,
        # **confidence_info,
        "total_pixels": total_pixels,
        "raw_corrosion_pixels": raw_corrosion_pixels,
        "raw_corrosion_area_percent": 100.0 * raw_corrosion_pixels / max(1, total_pixels),
        "corrosion_pixels": corrosion_pixels,
        "corrosion_area_percent": area_percent,
        "corrosion_level": level_info["level"],
        "corrosion_level_name": _severity_name(level_info["level"]),
        # "corrosion_level_description": level_info["description"],
    }
    return {
        "report": report
    }

def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_metrics_jsonl(path):
    path = Path(path)
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _plot_lines(records, keys, title, ylabel, path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for key in keys:
        values = [record[key] for record in records if key in record and isinstance(record[key], (int, float))]
        value_epochs = [record["epoch"] for record in records if key in record and isinstance(record[key], (int, float))]
        if values:
            ax.plot(value_epochs, values, marker="o", linewidth=2, markersize=3, label=key)
    if not ax.lines:
        plt.close(fig)
        return False
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _save_summary(records, path):
    best_key = "iou" if "iou" in records[-1] else "Mean Intersection over Union(mIoU)"
    best = max(records, key=lambda item: item.get(best_key, -1.0))
    last = records[-1]
    summary = {
        "best_metric": best_key,
        "best_epoch": best.get("epoch"),
        "best_score": best.get(best_key),
        "last_epoch": last.get("epoch"),
        "last_train_loss": last.get("train_loss"),
        "last_val_loss": last.get("val_loss"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def save_architecture_diagram(path):
    """Save a lightweight CRT architecture diagram used by the training script."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 360
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    boxes = [
        ("Input image", 40, 120, 190, 210, (219, 234, 254)),
        ("Hierarchical ViT\nbackbone", 250, 95, 430, 235, (209, 250, 229)),
        ("Token encoder", 500, 105, 680, 225, (254, 249, 195)),
        ("Query decoder", 750, 105, 930, 225, (255, 237, 213)),
        ("Mask head\n0-5 classes", 1000, 95, 1140, 235, (243, 232, 255)),
    ]
    for text, x0, y0, x1, y1, fill in boxes:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=fill, outline=(71, 85, 105), width=2)
        lines = text.split("\n")
        line_y = y0 + (y1 - y0 - 18 * len(lines)) // 2
        for line in lines:
            draw.text((x0 + 16, line_y), line, fill=(15, 23, 42))
            line_y += 22
    for (_, _x0, _y0, x1, _y1, _fill), (_, nx0, ny0, _nx1, ny1, _nfill) in zip(boxes, boxes[1:]):
        y = (ny0 + ny1) // 2
        draw.line((x1 + 12, y, nx0 - 18, y), fill=(51, 65, 85), width=3)
        draw.polygon([(nx0 - 18, y - 7), (nx0 - 18, y + 7), (nx0 - 6, y)], fill=(51, 65, 85))
    draw.text((40, 285), "CRT semantic segmentation pipeline", fill=(15, 23, 42))
    image.save(path)
    return path


def save_metrics_figures(metrics_path, output_dir=None, title_prefix="CRT"):
    """Render training curves from metrics.jsonl into publication-style PNG figures."""
    metrics_path = Path(metrics_path)
    output_dir = Path(output_dir) if output_dir is not None else metrics_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_metrics_jsonl(metrics_path)
    if not records:
        return []

    figures = [
        (
            "loss_curves.png",
            [
                "train_loss",
                "val_loss",
                "loss_bce",
                "loss_dice",
                "loss_tversky",
                "loss_focal_tversky",
                "loss_iou_quality",
                "loss_ce",
            ],
            f"{title_prefix} Training Loss",
            "Loss",
        ),
        ("iou_curves.png", ["iou", "background_iou", "miou"], "IoU and mIoU", "Score"),
        ("dice_f1_curve.png", ["dice", "f1", "boundary_f1"], "Dice, F1 and Boundary F1", "Score"),
        ("precision_recall_curve.png", ["precision", "recall", "specificity"], "Precision, Recall and Specificity", "Score"),
        ("accuracy_curve.png", ["pixel_accuracy", "balanced_accuracy"], "Pixel and Balanced Accuracy", "Score"),
        (
            "unet_metrics.png",
            [
                "Pixel Accuracy",
                "Frequency Weighted Intersection over Union",
                "Mean Pixel Accuracy",
                "Mean Intersection over Union(mIoU)",
                "Mean F1 Score",
            ],
            "UNet Metrics",
            "Score",
        ),
        ("learning_rate_curve.png", ["lr"], "Learning Rate Schedule", "Learning Rate"),
    ]
    saved = []
    for filename, keys, title, ylabel in figures:
        path = output_dir / filename
        if _plot_lines(records, keys, title, ylabel, path):
            saved.append(path)
    _save_summary(records, output_dir / "metrics_summary.json")
    saved.append(output_dir / "metrics_summary.json")
    return saved
