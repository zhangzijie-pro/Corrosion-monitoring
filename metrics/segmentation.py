import torch
import torch.nn.functional as F


def _boundary(mask):
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return (mask - eroded).clamp_min(0.0)


class SegmentationMeter:
    """Accumulates paper-style binary segmentation metrics."""

    def __init__(self, threshold=0.5, boundary_tolerance=3):
        self.threshold = threshold
        self.boundary_tolerance = boundary_tolerance
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0
        self.boundary_tp = self.boundary_fp = self.boundary_fn = 0.0

    @torch.no_grad()
    def update(self, logits, targets):
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        preds = (torch.sigmoid(logits) >= self.threshold).float()
        targets = (targets > 0.5).float()
        inv_preds = 1.0 - preds
        inv_targets = 1.0 - targets
        self.tp += torch.sum(preds * targets).item()
        self.fp += torch.sum(preds * inv_targets).item()
        self.fn += torch.sum(inv_preds * targets).item()
        self.tn += torch.sum(inv_preds * inv_targets).item()

        pred_b = _boundary(preds)
        target_b = _boundary(targets)
        if self.boundary_tolerance > 0:
            kernel_size = 2 * self.boundary_tolerance + 1
            pred_match = F.max_pool2d(pred_b, kernel_size=kernel_size, stride=1, padding=self.boundary_tolerance)
            target_match = F.max_pool2d(target_b, kernel_size=kernel_size, stride=1, padding=self.boundary_tolerance)
        else:
            pred_match = pred_b
            target_match = target_b
        self.boundary_tp += torch.sum(pred_b * target_match).item()
        self.boundary_fp += torch.sum(pred_b * (1.0 - target_match)).item()
        self.boundary_fn += torch.sum(target_b * (1.0 - pred_match)).item()

    def compute(self):
        eps = 1e-7
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        bg_iou = self.tn / (self.tn + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        specificity = self.tn / (self.tn + self.fp + eps)
        dice = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + eps)
        pixel_acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        boundary_precision = self.boundary_tp / (self.boundary_tp + self.boundary_fp + eps)
        boundary_recall = self.boundary_tp / (self.boundary_tp + self.boundary_fn + eps)
        boundary_f1 = 2.0 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall + eps)
        return {
            "iou": iou,
            "background_iou": bg_iou,
            "miou": 0.5 * (iou + bg_iou),
            "dice": dice,
            "f1": dice,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "pixel_accuracy": pixel_acc,
            "balanced_accuracy": 0.5 * (recall + specificity),
            "boundary_f1": boundary_f1,
        }
