"""Helper utilities for training loops."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .logger import JsonLogger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_optimizer(params, optimizer_cfg: Dict[str, Any]) -> Optimizer:
    name = optimizer_cfg.get("name", "adam").lower()
    lr = optimizer_cfg.get("lr", 1e-3)
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = optimizer_cfg.get("momentum", 0.9)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def create_scheduler(optimizer: Optimizer, scheduler_cfg: Dict[str, Any] | None) -> _LRScheduler | None:
    if not scheduler_cfg:
        return None
    name = scheduler_cfg.get("name", "cosine").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_cfg.get("t_max", 50), eta_min=scheduler_cfg.get("eta_min", 1e-6)
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_cfg.get("step_size", 25), gamma=scheduler_cfg.get("gamma", 0.1)
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def save_checkpoint(state: Dict[str, Any], checkpoint_dir: Path, is_best: bool = False) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "last.ckpt"
    torch.save(state, path)
    if is_best:
        best_path = checkpoint_dir / "best.ckpt"
        torch.save(state, best_path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def log_metrics(logger: JsonLogger, step: int, metrics: Dict[str, Any]) -> None:
    entry = {"step": step, **metrics}
    logger.log(entry)
