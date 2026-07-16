import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationCriterion(nn.Module):
    """Configurable segmentation loss for recall-oriented corrosion masks."""

    def __init__(
        self,
        bce_weight=0.30,
        dice_weight=0.30,
        tversky_weight=0.30,
        focal_tversky_weight=0.10,
        iou_weight=0.05,
        smooth=1.0,
        tversky_alpha=0.30,
        tversky_beta=0.70,
        focal_tversky_gamma=0.75,
        pos_weight=1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.focal_tversky_weight = focal_tversky_weight
        self.iou_weight = iou_weight
        self.smooth = smooth
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_tversky_gamma = focal_tversky_gamma
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, outputs, targets):
        logits = outputs["out"] if isinstance(outputs, dict) else outputs
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = torch.sum(probs * targets, dims)
        fp = torch.sum(probs * (1.0 - targets), dims)
        fn = torch.sum((1.0 - probs) * targets, dims)

        dice_denom = torch.sum(probs + targets, dims)
        dice_score = (2.0 * intersection + self.smooth) / (dice_denom + self.smooth)
        dice_loss = 1.0 - torch.mean(dice_score)

        tversky_score = (intersection + self.smooth) / (
            intersection + self.tversky_alpha * fp + self.tversky_beta * fn + self.smooth
        )
        tversky_loss = 1.0 - torch.mean(tversky_score)
        focal_tversky_loss = torch.mean(torch.pow(1.0 - tversky_score, self.focal_tversky_gamma))

        loss = (
            self.bce_weight * bce
            + self.dice_weight * dice_loss
            + self.tversky_weight * tversky_loss
            + self.focal_tversky_weight * focal_tversky_loss
        )
        parts = {
            "loss_bce": bce.detach(),
            "loss_dice": dice_loss.detach(),
            "loss_tversky": tversky_loss.detach(),
            "loss_focal_tversky": focal_tversky_loss.detach(),
        }
        if isinstance(outputs, dict) and "iou_pred" in outputs and self.iou_weight > 0:
            hard = (probs > 0.5).float()
            inter = torch.sum(hard * targets, dims)
            pred_union = torch.sum((hard + targets) > 0, dims).float()
            target_iou = (inter / pred_union.clamp_min(1.0)).unsqueeze(1).detach()
            iou_loss = F.l1_loss(outputs["iou_pred"], target_iou)
            loss = loss + self.iou_weight * iou_loss
            parts["loss_iou_quality"] = iou_loss.detach()
        parts["loss"] = loss.detach()
        return loss, parts


class MulticlassSegmentationCriterion(nn.Module):
    """Combined multiclass loss for imbalanced semantic corrosion masks."""

    def __init__(
        self,
        ce_weight=0.45,
        dice_weight=0.40,
        focal_weight=0.15,
        focal_gamma=2.0,
        background_dice_weight=0.25,
        class_weights=None,
        ignore_index=None,
        smooth=1.0,
        **unused_kwargs,
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.focal_gamma = float(focal_gamma)
        self.background_dice_weight = float(background_dice_weight)
        self.ignore_index = ignore_index
        self.smooth = float(smooth)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    @staticmethod
    def _targets_to_indices(targets):
        if targets.ndim == 4 and targets.shape[1] == 1:
            return (targets[:, 0] > 0.5).long()
        if targets.ndim == 4:
            return torch.argmax(targets, dim=1).long()
        return targets.long()

    def _ce_kwargs(self, logits):
        kwargs = {}
        if self.class_weights is not None:
            kwargs["weight"] = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        if self.ignore_index is not None:
            kwargs["ignore_index"] = int(self.ignore_index)
        return kwargs

    def forward(self, outputs, targets):
        logits = outputs["out"] if isinstance(outputs, dict) else outputs
        targets = self._targets_to_indices(targets)
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        ce_per_pixel = F.cross_entropy(logits, targets, reduction="none", **self._ce_kwargs(logits))
        valid = torch.ones_like(targets, dtype=torch.bool)
        if self.ignore_index is not None:
            valid = targets != int(self.ignore_index)
        ce = ce_per_pixel[valid].mean() if valid.any() else ce_per_pixel.mean()

        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        safe_targets = targets.clamp(min=0, max=num_classes - 1)
        one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).to(dtype=probs.dtype)
        valid_f = valid.unsqueeze(1).to(dtype=probs.dtype)
        probs = probs * valid_f
        one_hot = one_hot * valid_f

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        cardinality = torch.sum(probs + one_hot, dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        dice_weights = torch.ones(num_classes, device=logits.device, dtype=logits.dtype)
        if num_classes > 1:
            dice_weights[0] = self.background_dice_weight
        present = torch.sum(one_hot, dims) > 0
        dice_weights = dice_weights * present.to(dtype=logits.dtype)
        dice_loss = 1.0 - (dice_per_class * dice_weights).sum() / dice_weights.sum().clamp_min(1e-6)

        with torch.no_grad():
            pt = torch.exp(-ce_per_pixel.clamp_min(0.0))
        focal_per_pixel = torch.pow(1.0 - pt, self.focal_gamma) * ce_per_pixel
        focal = focal_per_pixel[valid].mean() if valid.any() else focal_per_pixel.mean()

        loss = self.ce_weight * ce + self.dice_weight * dice_loss + self.focal_weight * focal
        return loss, {
            "loss_ce": ce.detach(),
            "loss_dice": dice_loss.detach(),
            "loss_focal": focal.detach(),
            "loss": loss.detach(),
        }
