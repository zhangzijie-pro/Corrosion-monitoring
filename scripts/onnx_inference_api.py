#!/usr/bin/env python3
from __future__ import print_function

import argparse
import base64
import io
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

try:
    from flask import Flask, jsonify, request
except ImportError:  # pragma: no cover
    Flask = None


DEFAULT_CLASS_NAMES = {
    "0": "Background",
    "1": "Slight corrosion",
    "2": "Light-moderate corrosion",
    "3": "Moderate corrosion",
    "4": "Severe corrosion",
    "5": "Critical corrosion",
}

SEVERITY_COLORS = np.asarray(
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


def load_metadata(path):
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "image_size": [512, 512],
        "input_name": "image",
        "output_name": "logits",
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "postprocess": {
            "threshold": 0.45,
            "threshold_mode": "foreground_probability_sum",
            "background_margin": -0.05,
            "min_area_percent": 0.02,
            "min_component_area_ratio": 0.0001,
        },
        "class_names": DEFAULT_CLASS_NAMES,
    }


def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def preprocess_image(image, metadata):
    image_size = metadata.get("image_size", [512, 512])
    height, width = int(image_size[0]), int(image_size[1])
    resized = image.convert("RGB").resize((width, height), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(metadata.get("normalization", {}).get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(metadata.get("normalization", {}).get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    arr = (arr - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def component_boxes(mask):
    mask = mask.astype(bool)
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    boxes = []
    current = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            q = deque([(y, x)])
            labels[y, x] = current
            xs = []
            ys = []
            while q:
                cy, cx = q.popleft()
                xs.append(cx)
                ys.append(cy)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            q.append((ny, nx))
            left = min(xs)
            top = min(ys)
            boxes.append((current, left, top, max(xs) - left + 1, max(ys) - top + 1, len(xs)))
    return labels, boxes


def remove_small_components(class_map, min_component_area):
    if min_component_area <= 1:
        return class_map
    labels, boxes = component_boxes(class_map > 0)
    cleaned = class_map.copy()
    for label, _left, _top, _width, _height, area in boxes:
        if area < min_component_area:
            cleaned[labels == label] = 0
    return cleaned


def threshold_class_map(prob_np, metadata, postprocess_override=None):
    post = dict(metadata.get("postprocess", {}))
    if postprocess_override:
        post.update(postprocess_override)
    threshold = float(post.get("threshold", 0.45))
    min_area_percent = float(post.get("min_area_percent", 0.02))
    min_component_area_ratio = float(post.get("min_component_area_ratio", 0.0001))
    background_margin = float(post.get("background_margin", -0.05))

    bg = prob_np[0]
    fg_probs = prob_np[1:]
    fg_mass = np.sum(fg_probs, axis=0)
    fg_cls = np.argmax(fg_probs, axis=0).astype(np.uint8) + 1
    class_map = np.where((fg_mass >= threshold) & (fg_mass >= (bg + background_margin)), fg_cls, 0).astype(np.uint8)
    min_area = max(1, int(round(class_map.size * min_component_area_ratio)))
    class_map = remove_small_components(class_map, min_area)
    area_percent = 100.0 * float(np.mean(class_map > 0))
    if area_percent < min_area_percent:
        class_map[:, :] = 0
    return class_map


def classify_area(area_percent):
    if area_percent <= 0.0:
        return 0
    if area_percent <= 10.0:
        return 1
    if area_percent <= 30.0:
        return 2
    if area_percent <= 50.0:
        return 3
    if area_percent <= 75.0:
        return 4
    return 5


def summarize_blocks(class_map, confidence_map, class_names):
    labels, boxes = component_boxes(class_map > 0)
    blocks = []
    for label, left, top, width, height, area in boxes:
        component = labels == label
        values = class_map[component]
        counts = np.bincount(values, minlength=6)
        level = int(np.argmax(counts[1:]) + 1) if counts[1:].sum() > 0 else 0
        blocks.append(
            {
                "id": len(blocks) + 1,
                "bbox_xywh": [int(left), int(top), int(width), int(height)],
                "area_pixels": int(area),
                "corrosion_level": level,
                "corrosion_level_name": class_names.get(str(level), "Level %d" % level),
                "confidence_mean": float(confidence_map[component].mean()) if np.any(component) else 0.0,
                "class_pixels": dict((str(i), int(counts[i])) for i in range(1, min(len(counts), 6)) if counts[i] > 0),
            }
        )
    return blocks


def make_overlay(original, class_map):
    resized = original.convert("RGB").resize((class_map.shape[1], class_map.shape[0]), Image.BILINEAR)
    base = np.asarray(resized, dtype=np.float32)
    colors = SEVERITY_COLORS[np.clip(class_map, 0, len(SEVERITY_COLORS) - 1)]
    alpha = (class_map > 0).astype(np.float32)[..., None] * 0.50
    overlay = base * (1.0 - alpha) + colors.astype(np.float32) * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def image_to_data_url(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class ONNXCorrosionPredictor(object):
    def __init__(self, onnx_path, metadata_path=None, providers=None):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed.")
        self.onnx_path = str(onnx_path)
        self.metadata = load_metadata(metadata_path)
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.input_name = self.metadata.get("input_name") or self.session.get_inputs()[0].name
        self.output_name = self.metadata.get("output_name") or self.session.get_outputs()[0].name

    def predict(self, image, include_overlay=False, postprocess_override=None):
        tensor = preprocess_image(image, self.metadata)
        logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
        probs = softmax(logits, axis=1)[0]
        post = dict(self.metadata.get("postprocess", {}))
        if postprocess_override:
            post.update(postprocess_override)
        metadata = dict(self.metadata)
        metadata["postprocess"] = post
        class_map = threshold_class_map(probs, metadata)
        corrosion_probability = (1.0 - probs[0]).astype(np.float32)
        confidence_map = corrosion_probability
        corrosion = class_map > 0
        background = ~corrosion
        total_pixels = int(class_map.size)
        corrosion_pixels = int(corrosion.sum())
        area_percent = 100.0 * corrosion_pixels / max(1, total_pixels)
        class_names = self.metadata.get("class_names", DEFAULT_CLASS_NAMES)
        level = classify_area(area_percent)
        level_pixels = dict((str(i), int(np.sum(class_map == i))) for i in range(1, probs.shape[0]))
        report = {
            "model_output_type": "onnx_multiclass",
            "num_classes": int(probs.shape[0]),
            "image_width": int(class_map.shape[1]),
            "image_height": int(class_map.shape[0]),
            "threshold": float(post.get("threshold", 0.45)),
            "threshold_mode": post.get("threshold_mode", "foreground_probability_sum"),
            "background_margin": float(post.get("background_margin", -0.05)),
            "total_pixels": total_pixels,
            "raw_corrosion_pixels": int((np.argmax(probs, axis=0) > 0).sum()),
            "raw_corrosion_area_percent": 100.0 * int((np.argmax(probs, axis=0) > 0).sum()) / max(1, total_pixels),
            "raw_foreground_mass_pixels": int((corrosion_probability >= float(post.get("threshold", 0.45))).sum()),
            "raw_foreground_mass_area_percent": 100.0 * int((corrosion_probability >= float(post.get("threshold", 0.45))).sum()) / max(1, total_pixels),
            "corrosion_pixels": corrosion_pixels,
            "corrosion_area_percent": area_percent,
            "corrosion_level": int(level),
            "corrosion_level_name": class_names.get(str(level), "Level %d" % level),
            "corrosion_level_method": "total_corrosion_area_percent",
            "severity_pixels": level_pixels,
            "severity_area_percent": dict((k, 100.0 * v / max(1, total_pixels)) for k, v in level_pixels.items()),
            "foreground_confidence": float(confidence_map[corrosion].mean()) if np.any(corrosion) else 0.0,
            "background_confidence": float(probs[0][background].mean()) if np.any(background) else 0.0,
            "mean_probability": float(corrosion_probability.mean()),
            "max_probability": float(corrosion_probability.max()),
            "mean_foreground_mass": float(corrosion_probability.mean()),
            "max_foreground_mass": float(corrosion_probability.max()),
            "corrosion_blocks": summarize_blocks(class_map, confidence_map, class_names),
        }
        report["detection_confidence"] = report["foreground_confidence"] if corrosion_pixels else report["background_confidence"]
        if include_overlay:
            report["overlay_image"] = image_to_data_url(make_overlay(image, class_map))
        return report


def run_cli(args):
    predictor = ONNXCorrosionPredictor(args.model, args.metadata)
    image = Image.open(args.input).convert("RGB")
    override = {}
    if args.threshold is not None:
        override["threshold"] = args.threshold
    if args.min_area_percent is not None:
        override["min_area_percent"] = args.min_area_percent
    if args.background_margin is not None:
        override["background_margin"] = args.background_margin
    report = predictor.predict(image, include_overlay=args.include_overlay, postprocess_override=override)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def create_app(model_path, metadata_path=None):
    if Flask is None:
        raise RuntimeError("Flask is not installed.")
    predictor = ONNXCorrosionPredictor(model_path, metadata_path)
    app = Flask(__name__)

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"ok": True, "model": str(model_path), "runtime": "onnxruntime"})

    @app.route("/api/onnx/predict", methods=["POST"])
    def predict():
        file_obj = request.files.get("image")
        if file_obj is None:
            return jsonify({"error": "Missing image file field: image"}), 400
        try:
            image = Image.open(file_obj.stream).convert("RGB")
            include_overlay = request.form.get("include_overlay", "1").lower() not in ("0", "false", "no")
            override = {}
            for key in ("threshold", "min_area_percent", "min_component_area_ratio", "background_margin"):
                value = request.form.get(key)
                if value not in (None, ""):
                    override[key] = float(value)
            return jsonify(predictor.predict(image, include_overlay=include_overlay, postprocess_override=override))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


def main():
    parser = argparse.ArgumentParser(description="Run ONNX corrosion segmentation inference or API.")
    parser.add_argument("--model", required=True, help="Path to .onnx model.")
    parser.add_argument("--metadata", help="Path to .metadata.json from export_pt_to_onnx.py.")
    parser.add_argument("--input", help="Image path for CLI prediction.")
    parser.add_argument("--output", help="Optional JSON output path for CLI prediction.")
    parser.add_argument("--include-overlay", action="store_true")
    parser.add_argument("--threshold", type=float, help="Foreground probability-sum threshold.")
    parser.add_argument("--min-area-percent", type=float, help="Suppress detections below this image area percent.")
    parser.add_argument("--background-margin", type=float, help="Foreground mass margin relative to background; lower values are more sensitive.")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()

    if args.serve:
        app = create_app(args.model, args.metadata)
        app.run(host=args.host, port=args.port, debug=False)
        return
    if not args.input:
        raise SystemExit("--input is required unless --serve is used.")
    run_cli(args)


if __name__ == "__main__":
    main()
