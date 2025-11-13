"""UNet implementation tailored for depth-based segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNetConfig:
    in_channels: int = 1
    num_classes: int = 2
    base_channels: int = 32
    depth: int = 4
    use_batch_norm: bool = True


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        if use_batch_norm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        if use_batch_norm:
            layers.insert(-1, nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple forwarding
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, use_batch_norm)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, use_batch_norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class DepthUNet(nn.Module):
    """Flexible UNet backbone."""

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config

        channels = [config.base_channels * (2**i) for i in range(config.depth)]
        self.input_conv = ConvBlock(config.in_channels, channels[0], config.use_batch_norm)

        self.down_blocks = nn.ModuleList()
        for idx in range(config.depth - 1):
            self.down_blocks.append(DownBlock(channels[idx], channels[idx + 1], config.use_batch_norm))

        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, config.use_batch_norm)

        reversed_channels = list(reversed(channels))
        self.up_blocks = nn.ModuleList()
        in_ch = channels[-1] * 2
        for out_ch in reversed_channels:
            self.up_blocks.append(UpBlock(in_ch, out_ch, config.use_batch_norm))
            in_ch = out_ch

        self.classifier = nn.Conv2d(channels[0], config.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        skips = []
        current = x
        for down in self.down_blocks:
            skip, current = down(current)
            skips.append(skip)
        current = self.bottleneck(current)
        for skip, up in zip(reversed(skips), self.up_blocks):
            current = up(current, skip)
        logits = self.classifier(current)
        return logits


def build_model(model_name: str, **kwargs) -> nn.Module:
    if model_name.lower() == "unet":
        config = UNetConfig(**kwargs)
        return DepthUNet(config)
    raise ValueError(f"Unsupported model: {model_name}")


def get_model_registry() -> Dict[str, callable]:
    return {"unet": lambda **kwargs: DepthUNet(UNetConfig(**kwargs))}
