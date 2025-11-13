"""Training entry point for depth segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
import yaml

from datasets.depth_dataset import (
    DepthDatasetConfig,
    DepthSegmentationDataset,
    default_collate_fn,
    load_class_names,
)
from models import get_model
from utils.logger import JsonLogger
from utils.metrics import compute_confusion_matrix, mean_iou, pixel_accuracy
from utils.train_utils import (
    create_optimizer,
    create_scheduler,
    log_metrics,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train depth segmentation model")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def build_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    dataset_cfg = DepthDatasetConfig(
        root=Path(config["data"]["root"]),
        split="train",
        include_rgb=config["data"].get("include_rgb", False),
        class_names=config["data"].get("class_names"),
    )
    augmentations = None
    if config["data"].get("augment", False):
        augmentations = build_augmentations(config["data"].get("augmentation_params", {}))

    class_names = None
    try:
        class_names = load_class_names(dataset_cfg)
    except RuntimeError:
        pass

    train_dataset = DepthSegmentationDataset(dataset_cfg, augmentations=augmentations)
    val_dataset = DepthSegmentationDataset(
        DepthDatasetConfig(
            root=dataset_cfg.root,
            split="val",
            include_rgb=dataset_cfg.include_rgb,
            class_names=class_names,
        )
    )

    loader_args = {"batch_size": config["training"].get("batch_size", 4), "num_workers": config["training"].get("num_workers", 4), "pin_memory": True}
    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=default_collate_fn, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=default_collate_fn, **loader_args)
    return {"train": train_loader, "val": val_loader}


def build_augmentations(params: Dict[str, Any]):
    import albumentations as A

    transforms = [
        A.HorizontalFlip(p=params.get("hflip", 0.5)),
        A.VerticalFlip(p=params.get("vflip", 0.0)),
        A.RandomRotate90(p=params.get("rotate90", 0.2)),
        A.RandomBrightnessContrast(p=params.get("brightness_contrast", 0.2)),
        A.GaussNoise(var_limit=params.get("gauss_noise", (1e-4, 1e-3)), p=params.get("gauss_noise_p", 0.3)),
    ]

    def apply(sample: Dict[str, Any]) -> Dict[str, Any]:
        depth = sample["depth"]
        mask = sample["mask"].astype("int32")
        rgb = sample.get("rgb")
        additional_targets = {}
        if rgb is not None:
            additional_targets["rgb"] = "image"
        aug = A.Compose(transforms, additional_targets=additional_targets)
        result = aug(image=depth, mask=mask, **({"rgb": rgb} if rgb is not None else {}))
        sample["depth"] = result["image"]
        sample["mask"] = result["mask"].astype(sample["mask"].dtype)
        if rgb is not None:
            sample["rgb"] = result["rgb"]
        return sample

    return apply


def train_one_epoch(model, loader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss = 0.0
    total_cm = torch.zeros((num_classes, num_classes), device=device)
    for batch in loader:
        depth = batch["depth"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad()
        logits = model(depth)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * depth.size(0)
        preds = torch.argmax(logits, dim=1)
        total_cm += compute_confusion_matrix(preds, mask, num_classes)
    avg_loss = total_loss / len(loader.dataset)
    metrics = {"loss": avg_loss, "pixel_acc": pixel_accuracy(total_cm), "mean_iou": mean_iou(total_cm)}
    return metrics


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_cm = torch.zeros((num_classes, num_classes), device=device)
    with torch.no_grad():
        for batch in loader:
            depth = batch["depth"].to(device)
            mask = batch["mask"].to(device)
            logits = model(depth)
            loss = criterion(logits, mask)
            total_loss += loss.item() * depth.size(0)
            preds = torch.argmax(logits, dim=1)
            total_cm += compute_confusion_matrix(preds, mask, num_classes)
    avg_loss = total_loss / len(loader.dataset)
    metrics = {"loss": avg_loss, "pixel_acc": pixel_accuracy(total_cm), "mean_iou": mean_iou(total_cm)}
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = torch.device(args.device)

    dataloaders = build_dataloaders(config)
    model_cfg = config["model"]
    model = get_model(model_cfg.get("name", "unet"), **model_cfg.get("params", {})).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = create_optimizer(model.parameters(), config["optimizer"])
    scheduler = create_scheduler(optimizer, config.get("scheduler"))

    run_dir = Path(config.get("run_dir", "runs")) / config.get("experiment_name", "depth-unet")
    logger = JsonLogger(run_dir)
    logger.save_config(config)

    best_miou = 0.0
    num_classes = model_cfg.get("params", {}).get("num_classes", 2)

    for epoch in range(1, config["training"].get("epochs", 50) + 1):
        train_metrics = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device, num_classes)
        val_metrics = evaluate(model, dataloaders["val"], criterion, device, num_classes)
        if scheduler is not None:
            scheduler.step()

        log_metrics(logger, epoch, {f"train_{k}": v for k, v in train_metrics.items()})
        log_metrics(logger, epoch, {f"val_{k}": v for k, v in val_metrics.items()})

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_miou": best_miou,
        }
        is_best = val_metrics["mean_iou"] > best_miou
        if is_best:
            best_miou = val_metrics["mean_iou"]
            state["best_miou"] = best_miou
        save_checkpoint(state, run_dir, is_best=is_best)
        print(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}, indent=2))

    print(f"Training completed. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
