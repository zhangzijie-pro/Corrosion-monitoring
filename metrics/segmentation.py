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


class MulticlassSegmentationMeter:
    """Confusion-matrix metrics for indexed 0..N semantic segmentation masks."""

    def __init__(self, num_classes=6, threshold=0.5):
        self.num_classes = int(num_classes)
        self.threshold = float(threshold)
        self.reset()

    def reset(self):
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, targets):
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)
        if logits.shape[1] > 2:
            background = probs[:, 0]
            foreground = probs[:, 1:].sum(dim=1)
            foreground_class = torch.argmax(probs[:, 1:], dim=1) + 1
            preds = torch.where((foreground >= self.threshold) & (foreground >= background), foreground_class, 0)
        else:
            preds = torch.argmax(probs, dim=1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        elif targets.ndim == 4:
            targets = torch.argmax(targets, dim=1)
        targets = targets.long().clamp(0, self.num_classes - 1)
        preds = preds.long().clamp(0, self.num_classes - 1)
        encoded = targets.reshape(-1) * self.num_classes + preds.reshape(-1)
        counts = torch.bincount(encoded, minlength=self.num_classes * self.num_classes)
        self.confusion += counts.reshape(self.num_classes, self.num_classes).cpu().to(torch.float64)

    def compute(self):
        eps = 1e-7
        cm = self.confusion
        tp = torch.diag(cm)
        support = cm.sum(dim=1)
        predicted = cm.sum(dim=0)
        total = cm.sum().clamp_min(eps)
        union = support + predicted - tp
        iou = tp / union.clamp_min(eps)
        precision = tp / predicted.clamp_min(eps)
        recall = tp / support.clamp_min(eps)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(eps)
        present = support > 0
        foreground_present = present.clone()
        if foreground_present.numel() > 0:
            foreground_present[0] = False
        foreground_iou = iou[foreground_present]
        present_iou = iou[present]
        binary_tp = cm[1:, 1:].sum()
        binary_fp = cm[0, 1:].sum()
        binary_fn = cm[1:, 0].sum()
        binary_precision = binary_tp / (binary_tp + binary_fp).clamp_min(eps)
        binary_recall = binary_tp / (binary_tp + binary_fn).clamp_min(eps)
        binary_iou = binary_tp / (binary_tp + binary_fp + binary_fn).clamp_min(eps)
        binary_dice = 2.0 * binary_tp / (2.0 * binary_tp + binary_fp + binary_fn).clamp_min(eps)
        return {
            "Pixel Accuracy": float(tp.sum() / total),
            "Foreground Binary IoU": float(binary_iou),
            "Foreground Binary Dice": float(binary_dice),
            "Foreground Binary Precision": float(binary_precision),
            "Foreground Binary Recall": float(binary_recall),
            "Mean Pixel Accuracy": float(recall[present].mean()) if present.any() else 0.0,
            "Mean Intersection over Union(mIoU)": float(present_iou.mean()) if present_iou.numel() else 0.0,
            "Mean Present Class IoU": float(present_iou.mean()) if present_iou.numel() else 0.0,
            "Mean Foreground IoU": float(foreground_iou.mean()) if foreground_iou.numel() else 0.0,
            "Mean F1 Score": float(f1[present].mean()) if present.any() else 0.0,
            "Frequency Weighted Intersection over Union": float((support[present] * iou[present]).sum() / total) if present.any() else 0.0,
            "Class IoU": [float(v) for v in iou],
            "Class Precision": [float(v) for v in precision],
            "Class Recall": [float(v) for v in recall],
            "Class F1": [float(v) for v in f1],
        }
