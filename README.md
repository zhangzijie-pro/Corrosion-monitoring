# Corrosion Monitoring and Multi-class Segmentation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Segmentation-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-green)](https://onnxruntime.ai/)
[![Task](https://img.shields.io/badge/Task-Corrosion%20Monitoring-orange)](https://github.com/zhangzijie-pro/Corrosion-monitoring)

> Corrosion monitoring for ship and steel surface inspection, with 0-5 semantic corrosion segmentation, external corrosion dataset import, ONNX export, and Python 3.6 edge deployment support.

---

## Features

- Multi-class corrosion segmentation: class `0` is background; classes `1-5` represent increasing corrosion severity.
- HSI / HSV severity labeling: converts binary corrosion annotations into fine-grained masks using color, brightness, red-brown tendency, and texture features.
- External corrosion data import: maps public condition-state masks into the project 0-5 severity label space.
- CRT backbone: transformer-style corrosion recognition model with encoder / decoder segmentation head.
- Robust post-processing: foreground probability thresholding, background-priority filtering, tiny-component removal, and minimum-area suppression.
- ONNX deployment: exports `.pt` checkpoints to ONNX opset 15 for Python 3.6 ONNXRuntime.
- ARM64 deployment: Docker service package is prepared separately for Linux ARM64 / aarch64 targets.

---

## Project Structure

```text
Corrosion-monitoring/
|-- add_dataset.py                  # Convert dataset/{HiRes,LoRes} into train/valid/test masks
|-- augment_dataset.py              # Offline image/mask augmentation
|-- train_segmentation.py           # Main training script
|-- validate_segmentation.py        # Validate a checkpoint
|-- test_multiclass_model.py        # Multiclass evaluation and prediction smoke test
|-- api.py                          # PyTorch checkpoint Flask API
|-- predict_corrosion.py            # Single image / directory prediction
|-- config/
|   `-- train.yaml                  # Training config
|-- data/
|   `-- corrosion_dataset.py        # Dataset loader and mask modes
|-- engine/
|   |-- trainer.py                  # Train / evaluate loops
|   `-- predictor.py                # Prediction entry
|-- loss/
|   `-- segmentation_loss.py        # Binary and multiclass losses
|-- metrics/
|   `-- segmentation.py             # Binary and multiclass segmentation metrics
|-- model/
|   |-- CRT.py                      # Corrosion Recognition Transformer
|   |-- backbone.py
|   |-- encode.py
|   |-- decode.py
|   `-- head.py
|-- scripts/
|   |-- export_pt_to_onnx.py        # Export .pt checkpoint to ONNX
|   `-- onnx_inference_api.py       # Lightweight ONNXRuntime API / CLI
|-- utils/
|   |-- checkpoint.py
|   |-- postprocess.py
|   |-- torch_compat.py
|   `-- visualize.py
`-- docs/
    `-- onnx_api_frontend_integration.md
```

Generated datasets, checkpoints, ONNX files, Docker packages, and release bundles are intentionally excluded from Git.

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/zhangzijie-pro/Corrosion-monitoring.git
cd Corrosion-monitoring
pip install -r requirements-py36.txt
```

For training, install a PyTorch build that matches your CUDA / CPU environment.

### 2. Prepare Source Dataset

Expected raw layout:

```text
dataset/
|-- HiRes/
|   |-- raw/        # original images, e.g. hull001.jpg
|   `-- labeled/    # masks, e.g. hull001_label.jpg
`-- LoRes/
    |-- raw/
    `-- labeled/
```

The label images can be binary or colored foreground masks. Near-black JPEG compression noise is suppressed by `--black-threshold`.

### 3. Build Train / Valid / Test Dataset

```bash
python add_dataset.py \
  --source-root dataset \
  --target-root dataset_all \
  --subsets HiRes LoRes \
  --black-threshold 8 \
  --mask-mode hsi_severity \
  --num-classes 6 \
  --overwrite
```

Outputs:

```text
dataset_all/
|-- train/
|-- valid/
|-- test/
|-- added_from_dataset_manifest.json
`-- severity_calibration.json
```

Mask values:

| Class | Meaning |
| --- | --- |
| 0 | Background / no corrosion |
| 1 | Slight corrosion |
| 2 | Light-moderate corrosion |
| 3 | Moderate corrosion |
| 4 | Severe corrosion |
| 5 | Critical corrosion |

### 4. Offline Augmentation

```bash
python augment_dataset.py \
  --input-root dataset_all \
  --output-root dataset_all_augmented \
  --mask-mode indexed_multiclass \
  --copies 8 \
  --augment-splits train \
  --include-originals \
  --output-image-size 512
```

`indexed_multiclass` preserves class IDs `0-5` with nearest-neighbor mask transforms.

### 5. Import External Corrosion Data

The recommended external source is the Virginia Tech corrosion condition-state semantic segmentation dataset. Use the 512x512 `Train/Test` split from that package:

```bash
python scripts/import_vt_corrosion_dataset.py \
  --source-root external_corrosion_cache/vt_corrosion_condition_state \
  --target-root dataset_all \
  --valid-ratio 0.1 \
  --seed 42 \
  --overwrite
```

The importer maps condition states into this project label space: background/good=`0`, fair=`2`, poor=`4`, severe=`5`. The original ship corrosion masks still provide HSI/KMeans-generated classes `1-5`.

---

## Training

Default config:

```text
config/train.yaml
```

Smoke test:

```bash
python train_segmentation.py \
  --config config/train.yaml \
  --data-root dataset_all_augmented \
  --epochs 2 \
  --output-dir runs/crt_corrosion_multisource_smoke
```

Full training:

```bash
python train_segmentation.py \
  --config config/train.yaml \
  --data-root dataset_all_augmented \
  --output-dir runs/crt_corrosion_multisource
```

Training outputs:

```text
runs/<experiment>/
|-- best.pt
|-- last.pt
|-- metrics.jsonl
|-- resolved_config.json
`-- figures/
```

---

## Evaluation

Run validation metrics:

```bash
python validate_segmentation.py \
  --checkpoint runs/crt_corrosion_multisource/best.pt \
  --device auto
```

Run multiclass smoke test:

```bash
python test_multiclass_model.py \
  --checkpoint runs/crt_dataset_all_multiclass_neg/best.pt \
  --eval \
  --input dataset_all_augmented/test \
  --output-dir runs/multiclass_test
```

Important metrics:

- `Foreground Binary IoU`
- `Foreground Binary Dice`
- `Mean Foreground IoU`
- `Mean Present Class IoU`
- `Mean Intersection over Union(mIoU)`
- `Pixel Accuracy`
- per-class IoU / F1 from confusion matrix

---

## Inference API

PyTorch checkpoint API:

```bash
MODEL_CHECKPOINT=runs/crt_corrosion_multisource/best.pt \
MULTICLASS_MODEL_CHECKPOINT=runs/crt_corrosion_multisource/best.pt \
MODEL_DEVICE=cpu \
PORT=9000 \
python api.py
```

Endpoints:

- `GET /health`: service status
- `POST /predict`: image upload, returns corrosion area, level, confidence, and optional mask overlay

ONNX API:

```bash
python scripts/onnx_inference_api.py \
  --model runs/onnx_15/model.onnx \
  --host 0.0.0.0 \
  --port 9000
```

See [docs/onnx_api_frontend_integration.md](docs/onnx_api_frontend_integration.md) for response parsing and front-end integration details.

---

## ONNX Export

```bash
python scripts/export_pt_to_onnx.py \
  --checkpoint runs/crt_dataset_all_multiclass_neg/best.pt \
  --output runs/onnx_15/model.onnx \
  --opset 15 \
  --input-size 512 512 \
  --num-classes 6
```

Recommended release assets:

```text
model.onnx
metadata.json
class_mapping.json
```

The production model is expected to be published through GitHub Releases instead of being committed to Git.

---

## ARM64 Docker Deployment

The ARM64 Python 3.6 ONNXRuntime deployment package is intentionally excluded from this source repository. Keep deployment bundles outside Git and attach the final model / package to a GitHub Release.

The service should expose:

- `GET /health`
- `POST /predict`

The final ONNX model can be mounted into the container at runtime, or baked into a private deployment artifact depending on the target device policy.

---

## Dataset Notes

The 0-5 severity masks are generated from annotated corrosion foreground pixels. The conversion uses HSI / HSV color cues, red-brown corrosion tendency, brightness, saturation, texture strength, and KMeans clustering. Clusters are sorted from slight to critical corrosion based on darkness, saturation, red-brown tendency, and local texture.

Generalization is improved by mixing the original ship corrosion images with an external corrosion condition-state dataset and applying synchronized image/mask augmentation. The current training flow intentionally avoids all-black random negative samples because they can make the model over-conservative on out-of-dataset corrosion images.

---

## Config

Recommended multiclass settings are in `config/train.yaml`:

- `num_classes: 6`
- `mask_mode: indexed_multiclass`
- multiclass loss with class weighting
- foreground-thresholded post-processing

Adjust batch size, workers, and learning rate according to GPU memory.

---

## Future Improvements

- Add manually reviewed external corrosion domains.
- Add manual severity review tooling for classes `1-5`.
- Add calibration plots for foreground probability threshold selection.
- Add lightweight quantized ONNX / MNN / TensorRT variants for embedded deployment.

---

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Notes

- Do not commit raw datasets, generated datasets, runs, ONNX files, or Docker bundles.
- Publish trained models and deployment archives through GitHub Releases.
- For production, validate false-positive rate on ordinary images before deploying to an inspection workflow.
