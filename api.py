#!/usr/bin/env python3
import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from data.corrosion_dataset import IMAGENET_MEAN, IMAGENET_STD
from model import build_model
from utils import get_device
from utils.checkpoint import infer_model_cfg_from_state_dict, load_checkpoint
from utils.torch_compat import autocast_for_device, inference_context
from utils.visualize import SEVERITY_COLORS, build_multiclass_report


DEFAULT_CHECKPOINT = "runs/pt/best.pt"
CHECKPOINT_PATH = Path(os.environ.get("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT))
MULTICLASS_CHECKPOINT_PATH = Path(
    os.environ.get("MULTICLASS_MODEL_CHECKPOINT", os.environ.get("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT))
)
DEVICE_NAME = os.environ.get("MODEL_DEVICE", "auto")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "20"))
DEFAULT_POSTPROCESS_THRESHOLD = float(os.environ.get("POSTPROCESS_THRESHOLD", "0.45"))
DEFAULT_POSTPROCESS_MIN_AREA_PERCENT = float(os.environ.get("POSTPROCESS_MIN_AREA_PERCENT", "0.02"))
DEFAULT_POSTPROCESS_MIN_COMPONENT_AREA_RATIO = float(os.environ.get("POSTPROCESS_MIN_COMPONENT_AREA_RATIO", "0.0001"))
DEFAULT_POSTPROCESS_BACKGROUND_MARGIN = float(os.environ.get("POSTPROCESS_BACKGROUND_MARGIN", "-0.05"))

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_MODEL = None
_CFG = None
_DEVICE = None
_MODEL_CACHE = {}


MULTICLASS_PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>多语义腐蚀分割模型测试</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; color: #1f2937; background: #f3f5f8; }
    main { max-width: 1220px; margin: 0 auto; padding: 28px; }
    h1 { font-size: 24px; margin: 0 0 18px; }
    .layout { display: grid; grid-template-columns: 340px 1fr; gap: 18px; align-items: start; }
    form, .panel { background: #fff; border: 1px solid #d8dee8; border-radius: 8px; padding: 18px; }
    label { display: block; font-size: 13px; margin: 12px 0 6px; color: #4b5563; }
    input[type="file"] { box-sizing: border-box; width: 100%; }
    button { margin-top: 16px; width: 100%; height: 38px; border: 0; border-radius: 6px; background: #155eef; color: #fff; font-weight: 700; cursor: pointer; }
    button:disabled { background: #94a3b8; cursor: wait; }
    .meta { font-size: 13px; color: #64748b; margin-bottom: 14px; }
    .summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }
    .metric { border: 1px solid #d8dee8; border-radius: 8px; padding: 10px; background: #f8fafc; }
    .metric .label { font-size: 12px; color: #64748b; margin-bottom: 4px; }
    .metric .value { font-size: 16px; font-weight: 700; color: #111827; word-break: break-word; }
    .viewer { position: relative; display: inline-block; max-width: 100%; border: 1px solid #d8dee8; border-radius: 8px; overflow: hidden; background: #0f172a; }
    .viewer img { display: block; max-width: 100%; height: auto; }
    .hotspot { position: absolute; box-sizing: border-box; border: 2px solid rgba(255,255,255,0.0); background: rgba(255,255,255,0.0); cursor: crosshair; }
    .hotspot:hover { border-color: rgba(255,255,255,0.95); background: rgba(255,255,255,0.10); }
    .tooltip { position: fixed; display: none; pointer-events: none; z-index: 20; min-width: 220px; border: 1px solid #cbd5e1; border-radius: 8px; background: #fff; box-shadow: 0 10px 30px rgba(15,23,42,0.18); padding: 10px; font-size: 13px; }
    .tooltip .line { display: flex; justify-content: space-between; gap: 14px; margin: 4px 0; }
    .tooltip .key { color: #64748b; }
    .tooltip .value { color: #111827; font-weight: 700; }
    pre { white-space: pre-wrap; word-break: break-word; background: #101828; color: #d7e2f1; padding: 12px; border-radius: 8px; overflow: auto; max-height: 360px; }
    .error { color: #b42318; font-weight: 700; }
    @media (max-width: 920px) { .layout { grid-template-columns: 1fr; } .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  </style>
</head>
<body>
<main>
  <h1>多语义腐蚀分割模型测试</h1>
  <div class="layout">
    <form id="predictForm">
      <div class="meta">Checkpoint: {{ checkpoint }}</div>
      <label>上传图片</label>
      <input name="image" type="file" accept="image/*" required>
      <button id="submitBtn" type="submit">预测</button>
    </form>
    <section class="panel">
      <div id="status" class="meta">等待上传图片。</div>
      <div class="summary">
        <div class="metric"><div class="label">等级</div><div id="level" class="value">-</div></div>
        <div class="metric"><div class="label">等级名称</div><div id="name" class="value">-</div></div>
        <div class="metric"><div class="label">面积占比</div><div id="area" class="value">-</div></div>
        <div class="metric"><div class="label">置信度</div><div id="confidence" class="value">-</div></div>
      </div>
      <div id="viewer" class="viewer" hidden>
        <img id="overlay" alt="腐蚀分割叠加图">
        <div id="hotspots"></div>
      </div>
      <pre id="json"></pre>
    </section>
  </div>
  <div id="tooltip" class="tooltip"></div>
</main>
<script>
const form = document.getElementById('predictForm');
const btn = document.getElementById('submitBtn');
const statusEl = document.getElementById('status');
const jsonEl = document.getElementById('json');
const viewer = document.getElementById('viewer');
const overlay = document.getElementById('overlay');
const hotspots = document.getElementById('hotspots');
const tooltip = document.getElementById('tooltip');
const setText = (id, value) => { document.getElementById(id).textContent = value ?? '-'; };

function tooltipHtml(block) {
  const confidence = Number.isFinite(block.confidence_mean) ? block.confidence_mean.toFixed(6) : block.confidence_mean;
  return `
    <div class="line"><span class="key">corrosion_level</span><span class="value">${block.corrosion_level}</span></div>
    <div class="line"><span class="key">corrosion_level_name</span><span class="value">${block.corrosion_level_name}</span></div>
    <div class="line"><span class="key">confidence_mean</span><span class="value">${confidence}</span></div>
  `;
}

function renderHotspots(blocks, width, height) {
  hotspots.innerHTML = '';
  blocks.forEach((block) => {
    const [x, y, w, h] = block.bbox_xywh;
    const item = document.createElement('div');
    item.className = 'hotspot';
    item.style.left = `${(x / width) * 100}%`;
    item.style.top = `${(y / height) * 100}%`;
    item.style.width = `${(w / width) * 100}%`;
    item.style.height = `${(h / height) * 100}%`;
    item.addEventListener('mouseenter', () => { tooltip.innerHTML = tooltipHtml(block); tooltip.style.display = 'block'; });
    item.addEventListener('mousemove', (event) => {
      tooltip.style.left = `${event.clientX + 14}px`;
      tooltip.style.top = `${event.clientY + 14}px`;
    });
    item.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
    hotspots.appendChild(item);
  });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  btn.disabled = true;
  statusEl.textContent = '预测中...';
  statusEl.className = 'meta';
  jsonEl.textContent = '';
  viewer.hidden = true;
  hotspots.innerHTML = '';
  try {
    const response = await fetch('/api/multiclass/predict', { method: 'POST', body: new FormData(form) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '预测失败');
    setText('level', data.corrosion_level);
    setText('name', data.corrosion_level_name);
    setText('area', Number.isFinite(data.corrosion_area_percent) ? `${data.corrosion_area_percent.toFixed(2)}%` : data.corrosion_area_percent);
    setText('confidence', Number.isFinite(data.detection_confidence) ? data.detection_confidence.toFixed(6) : data.detection_confidence);
    overlay.onload = () => {
      renderHotspots(data.corrosion_blocks || [], data.image_width, data.image_height);
      viewer.hidden = false;
    };
    overlay.src = data.overlay_image;
    jsonEl.textContent = JSON.stringify(data.report, null, 2);
    statusEl.textContent = '预测完成。鼠标移动到叠加图中的腐蚀区域可查看该区域信息。';
  } catch (error) {
    statusEl.textContent = error.message;
    statusEl.className = 'meta error';
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
"""


def _allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _load_model_for_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    cache_key = str(checkpoint_path.resolve()) if checkpoint_path.exists() else str(checkpoint_path)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Set MODEL_CHECKPOINT or MULTICLASS_MODEL_CHECKPOINT."
        )
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    arch, num_classes, model_cfg = infer_model_cfg_from_state_dict(cfg, checkpoint["model"])
    device = get_device(DEVICE_NAME)
    model = build_model(arch=arch, num_classes=num_classes, **model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    loaded = (model, cfg, device)
    _MODEL_CACHE[cache_key] = loaded
    return loaded


def _load_model():
    global _MODEL, _CFG, _DEVICE
    if _MODEL is not None:
        return _MODEL, _CFG, _DEVICE
    model, cfg, device = _load_model_for_path(CHECKPOINT_PATH)
    _MODEL = model
    _CFG = cfg
    _DEVICE = device
    return _MODEL, _CFG, _DEVICE


def _preprocess_image(original, image_size):
    resized = original.resize((image_size[1], image_size[0]), Image.BILINEAR)
    image = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0)


def _read_request_image():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return None, (jsonify({"error": "Missing image file field: image"}), 400)
    if not _allowed_file(file.filename):
        return None, (jsonify({"error": f"Unsupported file type: {Path(file.filename).suffix}"}), 400)
    try:
        return Image.open(file.stream).convert("RGB"), None
    except Exception as exc:
        return None, (jsonify({"error": f"Invalid image file: {exc}"}), 400)


def _run_model_prediction(checkpoint_path, original):
    model, cfg, device = _load_model_for_path(checkpoint_path)
    image_size = cfg["data"].get("image_size", [512, 512])
    image = _preprocess_image(original, image_size)
    amp = device.type == "cuda" and str(os.environ.get("PREDICT_AMP", "1")).lower() not in {"0", "false", "no"}

    with inference_context(), autocast_for_device(device, enabled=amp):
        outputs = model(image.to(device, non_blocking=True))
        logits = outputs["out"] if isinstance(outputs, dict) else outputs
    return logits.cpu()


def _image_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _public_report(report):
    return {key: value for key, value in report.items() if not key.startswith("_")}


def _multiclass_overlay_image(original, class_map):
    color_map = SEVERITY_COLORS[np.clip(class_map, 0, len(SEVERITY_COLORS) - 1)]
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    alpha = (class_map > 0).astype(np.float32)[..., None] * 0.50
    overlay = base * (1.0 - alpha) + color_map.astype(np.float32) * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def _float_form_value(name, default):
    value = request.form.get(name)
    if value is None or value == "":
        return default
    return float(value)


@app.get("/")
def index():
    return multiclass_index()


@app.get("/multiclass")
def multiclass_index():
    return render_template_string(MULTICLASS_PAGE, checkpoint=str(MULTICLASS_CHECKPOINT_PATH))


@app.get("/api/health")
def health():
    loaded = _MODEL is not None
    return jsonify(
        {
            "ok": True,
            "checkpoint": str(CHECKPOINT_PATH),
            "multiclass_checkpoint": str(MULTICLASS_CHECKPOINT_PATH),
            "checkpoint_exists": CHECKPOINT_PATH.exists(),
            "multiclass_checkpoint_exists": MULTICLASS_CHECKPOINT_PATH.exists(),
            "device": str(_DEVICE) if _DEVICE is not None else DEVICE_NAME,
            "model_loaded": loaded,
            "cached_models": len(_MODEL_CACHE),
        }
    )


@app.post("/api/multiclass/predict")
def predict_multiclass():
    original, error = _read_request_image()
    if error is not None:
        return error

    try:
        logits = _run_model_prediction(MULTICLASS_CHECKPOINT_PATH, original)
        if logits.shape[1] <= 2:
            return jsonify({"error": "The configured multiclass checkpoint did not produce multiclass logits."}), 400

        _, cfg, _ = _load_model_for_path(MULTICLASS_CHECKPOINT_PATH)
        threshold = _float_form_value("threshold", DEFAULT_POSTPROCESS_THRESHOLD)
        min_area_percent = _float_form_value("min_area_percent", DEFAULT_POSTPROCESS_MIN_AREA_PERCENT)
        min_component_area_ratio = _float_form_value(
            "min_component_area_ratio",
            DEFAULT_POSTPROCESS_MIN_COMPONENT_AREA_RATIO,
        )
        background_margin = _float_form_value("background_margin", DEFAULT_POSTPROCESS_BACKGROUND_MARGIN)
        report = build_multiclass_report(
            original,
            logits,
            threshold=threshold,
            min_area_percent=min_area_percent,
            min_component_area_ratio=min_component_area_ratio,
            background_margin=background_margin,
        )
        public_report = _public_report(report)
        overlay = _multiclass_overlay_image(original, report["_class_map"])
        blocks = [
            {
                "id": block.get("id"),
                "bbox_xywh": block.get("bbox_xywh"),
                "corrosion_level": block.get("corrosion_level"),
                "corrosion_level_name": block.get("corrosion_level_name"),
                "confidence_mean": block.get("confidence_mean"),
            }
            for block in public_report.get("corrosion_blocks", [])
        ]
        payload = {
            "image_width": original.width,
            "image_height": original.height,
            "overlay_image": _image_data_url(overlay),
            "corrosion_blocks": blocks,
            "corrosion_level": public_report.get("corrosion_level"),
            "corrosion_level_name": public_report.get("corrosion_level_name"),
            "corrosion_area_percent": public_report.get("corrosion_area_percent"),
            "detection_confidence": public_report.get("detection_confidence"),
            "report": public_report,
        }
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/predict")
def predict():
    return predict_multiclass()

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "9000"))
    app.run(host=host, port=port, debug=False)
