"""Model registry for depth segmentation."""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from .unet import DepthUNet, UNetConfig


def get_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "unet":
        config = UNetConfig(**kwargs)
        return DepthUNet(config)
    raise ValueError(f"Unknown model '{name}'")


def available_models() -> Dict[str, Callable[..., nn.Module]]:
    return {"unet": lambda **kwargs: DepthUNet(UNetConfig(**kwargs))}
