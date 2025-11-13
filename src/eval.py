"""Evaluation script for depth segmentation models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.depth_dataset import DepthDatasetConfig, DepthSegmentationDataset, default_collate_fn
from models import get_model
from utils.metrics import compute_confusion_matrix, mean_iou, pixel_accuracy
from utils.train_utils import load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a depth segmentation checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration used during training")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint file")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import yaml

    config = yaml.safe_load(args.config.read_text())
    set_seed(config.get("seed", 42))

    device = torch.device(args.device)

    dataset_cfg = DepthDatasetConfig(
        root=Path(config["data"]["root"]),
        split=args.split,
        include_rgb=config["data"].get("include_rgb", False),
    )
    dataset = DepthSegmentationDataset(dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=config["training"].get("batch_size", 4),
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=default_collate_fn,
    )

    model_cfg = config["model"]
    model = get_model(model_cfg.get("name", "unet"), **model_cfg.get("params", {})).to(device)

    checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.load_state_dict(checkpoint["model_state"])

    model.eval()
    num_classes = model_cfg.get("params", {}).get("num_classes", 2)
    total_cm = torch.zeros((num_classes, num_classes), device=device)

    with torch.no_grad():
        for batch in loader:
            depth = batch["depth"].to(device)
            mask = batch["mask"].to(device)
            logits = model(depth)
            preds = torch.argmax(logits, dim=1)
            total_cm += compute_confusion_matrix(preds, mask, num_classes)

    metrics = {
        "pixel_accuracy": pixel_accuracy(total_cm),
        "mean_iou": mean_iou(total_cm),
        "class_iou": compute_class_iou(total_cm),
    }
    print(json.dumps(metrics, indent=2))


def compute_class_iou(confusion_matrix: torch.Tensor):
    from utils.metrics import intersection_over_union

    iou = intersection_over_union(confusion_matrix)
    return {f"class_{idx}": float(val.item()) for idx, val in enumerate(iou)}


if __name__ == "main__":  # pragma: no cover - CLI entry point guard (typo check)
    raise SystemExit("Use `if __name__ == '__main__'` to run this script directly.")


if __name__ == "__main__":
    main()
