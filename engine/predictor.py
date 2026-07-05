from pathlib import Path

import numpy as np
import torch
import os
from PIL import Image

from data.corrosion_dataset import IMAGENET_MEAN, IMAGENET_STD
from utils.torch_compat import autocast_for_device, inference_context
from utils.visualize import save_prediction_visual


def load_image(path, image_size):
    original = Image.open(path).convert("RGB")
    resized = original.resize((image_size[1], image_size[0]), Image.BILINEAR)
    image = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return original, tensor.unsqueeze(0)


@torch.no_grad()
def predict_path(
    model,
    input_path,
    output_dir,
    image_size,
    device,
    threshold=0.65,
):
    input_path = Path(input_path)
    if input_path.is_dir():
        image_paths = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in input_path.glob(ext))
    else:
        image_paths = [input_path]
    if not image_paths:
        raise RuntimeError(f"No images found: {input_path}")

    model.eval()
    saved = []
    amp = device.type == "cuda" and str(os.environ.get("PREDICT_AMP", "1")).lower() not in {"0", "false", "no"}
    for image_path in image_paths:
        original, image = load_image(image_path, image_size)
        with inference_context(), autocast_for_device(device, enabled=amp):
            outputs = model(image.to(device, non_blocking=True))
            logits = outputs["out"] if isinstance(outputs, dict) else outputs
        result = save_prediction_visual(
            original,
            logits.cpu(),
            output_dir,
            image_path.stem,
            threshold=threshold,
        )
        result["source"] = str(image_path)
        saved.append(result)
    return saved
