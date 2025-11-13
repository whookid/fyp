"""Segmentation metrics for depth-based experiments."""

from __future__ import annotations

import torch


def compute_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    with torch.no_grad():
        preds = preds.view(-1).long()
        targets = targets.view(-1).long()
        mask = targets >= 0
        preds = preds[mask]
        targets = targets[mask]
        cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=preds.device)
        indices = targets * num_classes + preds
        cm += torch.bincount(indices, minlength=num_classes**2).view(num_classes, num_classes)
        return cm


def intersection_over_union(confusion_matrix: torch.Tensor) -> torch.Tensor:
    true_positive = confusion_matrix.diag().float()
    false_positive = confusion_matrix.sum(dim=0).float() - true_positive
    false_negative = confusion_matrix.sum(dim=1).float() - true_positive
    denom = true_positive + false_positive + false_negative
    iou = true_positive / torch.clamp(denom, min=1e-6)
    return iou


def pixel_accuracy(confusion_matrix: torch.Tensor) -> float:
    correct = confusion_matrix.diag().sum().float()
    total = confusion_matrix.sum().float()
    return float((correct / torch.clamp(total, min=1e-6)).item())


def mean_iou(confusion_matrix: torch.Tensor) -> float:
    return float(intersection_over_union(confusion_matrix).mean().item())
