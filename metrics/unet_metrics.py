import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["SegmentationMetric", "UNetSegmentationMeter"]


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
        return self.confusionMatrix

    def pixelAccuracy(self):
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        denominator = self.confusionMatrix.sum(axis=1)
        denominator = np.where(denominator == 0, 1e-12, denominator)
        classAcc = np.diag(self.confusionMatrix) / denominator
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        union = np.where(union == 0, 1e-12, union)
        IoU = intersection / union
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())
        return mIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        denominator1 = np.sum(self.confusionMatrix)
        denominator1 = np.where(denominator1 == 0, 1e-12, denominator1)
        freq = np.sum(self.confusionMatrix, axis=1) / denominator1
        denominator2 = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        denominator2 = np.where(denominator2 == 0, 1e-12, denominator2)
        iu = np.diag(self.confusionMatrix) / denominator2
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def classF1Score(self):
        tp = np.diag(self.confusionMatrix)
        fp = self.confusionMatrix.sum(axis=0) - tp
        fn = self.confusionMatrix.sum(axis=1) - tp

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)

        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return f1

    def meanF1Score(self):
        f1 = self.classF1Score()
        mean_f1 = np.nanmean(f1)
        return mean_f1

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def get_scores(self):
        scores = {
            'Pixel Accuracy': self.pixelAccuracy(),
            'Class Pixel Accuracy': self.classPixelAccuracy(),
            'Intersection over Union': self.IntersectionOverUnion(),
            'Class F1 Score': self.classF1Score(),
            'Frequency Weighted Intersection over Union': self.Frequency_Weighted_Intersection_over_Union(),
            'Mean Pixel Accuracy': self.meanPixelAccuracy(),
            'Mean Intersection over Union(mIoU)': self.meanIntersectionOverUnion(),
            'Mean F1 Score': self.meanF1Score()
        }
        return scores


class UNetSegmentationMeter:
    """Torch-friendly adapter around the numpy UNet segmentation metrics."""

    def __init__(self, num_classes=2, threshold=0.5):
        self.metric = SegmentationMetric(num_classes)

    @torch.no_grad()
    def update(self, logits, targets):
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        preds = torch.argmax(logits, dim=1).long()

        if targets.ndim == 4 and targets.shape[1] == 1:
            labels = (targets[:, 0] > 0.5).long()
        elif targets.ndim == 4:
            labels = torch.argmax(targets, dim=1).long()
        else:
            labels = targets.long()

        self.metric.addBatch(preds.cpu().numpy(), labels.cpu().numpy())

    def compute(self):
        iou = self.metric.IntersectionOverUnion()
        class_acc = self.metric.classPixelAccuracy()
        class_f1 = self.metric.classF1Score()
        support = self.metric.confusionMatrix.sum(axis=1)
        present = support > 0
        foreground_present = present.copy()
        if foreground_present.size > 0:
            foreground_present[0] = False
        present_iou = float(np.nanmean(iou[present])) if np.any(present) else 0.0
        foreground_iou = float(np.nanmean(iou[foreground_present])) if np.any(foreground_present) else 0.0
        foreground_f1 = float(np.nanmean(class_f1[foreground_present])) if np.any(foreground_present) else 0.0
        return {
            "Pixel Accuracy": float(self.metric.pixelAccuracy()),
            "Class Pixel Accuracy": class_acc.tolist(),
            "Intersection over Union": iou.tolist(),
            "Class F1 Score": class_f1.tolist(),
            "Frequency Weighted Intersection over Union": float(self.metric.Frequency_Weighted_Intersection_over_Union()),
            "Mean Pixel Accuracy": float(self.metric.meanPixelAccuracy()),
            "Mean Intersection over Union(mIoU)": float(self.metric.meanIntersectionOverUnion()),
            "Mean Present Class IoU": present_iou,
            "Mean Foreground IoU": foreground_iou,
            "Mean Foreground F1 Score": foreground_f1,
            "Class Pixel Support": support.astype(np.int64).tolist(),
            "Mean F1 Score": float(self.metric.meanF1Score()),
        }
