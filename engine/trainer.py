import torch

from metrics import SegmentationMeter, UNetSegmentationMeter
from utils.torch_compat import autocast_for_device


def unwrap_logits(outputs):
    return outputs["out"] if isinstance(outputs, dict) else outputs


def _move_batch(batch, device):
    return batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, amp=False, max_grad_norm=0.0):
    model.train()
    total = 0.0
    loss_parts = {}
    for batch in loader:
        images, masks = _move_batch(batch, device)
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()
        with autocast_for_device(device, enabled=amp):
            outputs = model(images)
            loss, parts = criterion(outputs, masks)
        scaler.scale(loss).backward()
        if max_grad_norm and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * images.size(0)
        for key, value in parts.items():
            loss_parts[key] = loss_parts.get(key, 0.0) + float(value) * images.size(0)
    size = max(1, len(loader.dataset))
    metrics = {key: value / size for key, value in loss_parts.items()}
    metrics["train_loss"] = total / size
    return metrics


def build_metric_meter(metric_type="segmentation", threshold=0.5, boundary_tolerance=3, num_classes=2):
    if str(metric_type).lower() in {"unet", "unet_metrics"}:
        return UNetSegmentationMeter(num_classes=num_classes, threshold=threshold)
    if str(metric_type).lower() in {"multiclass", "multi_class", "semantic"}:
        return UNetSegmentationMeter(num_classes=num_classes, threshold=threshold)
    return SegmentationMeter(threshold=threshold, boundary_tolerance=boundary_tolerance)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    threshold=0.5,
    boundary_tolerance=3,
    metric_type="segmentation",
    num_classes=2,
):
    model.eval()
    meter = build_metric_meter(
        metric_type,
        threshold=threshold,
        boundary_tolerance=boundary_tolerance,
        num_classes=num_classes,
    )
    total = 0.0
    loss_parts = {}
    for batch in loader:
        images, masks = _move_batch(batch, device)
        outputs = model(images)
        loss, parts = criterion(outputs, masks)
        logits = unwrap_logits(outputs)
        meter.update(logits, masks)
        total += loss.item() * images.size(0)
        for key, value in parts.items():
            loss_parts[key] = loss_parts.get(key, 0.0) + float(value) * images.size(0)
    size = max(1, len(loader.dataset))
    metrics = {key: value / size for key, value in loss_parts.items()}
    metrics["val_loss"] = total / size
    metrics.update(meter.compute())
    return metrics
