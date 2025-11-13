"""Depth camera segmentation dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class DepthDatasetConfig:
    """Configuration describing the dataset layout."""

    root: Path
    split: str
    include_rgb: bool = False
    class_names: Optional[List[str]] = None


class DepthSegmentationDataset(Dataset):
    """Loads depth frames and segmentation masks from disk.

    Depth frames are expected to be stored as 16-bit PNGs representing normalized
    depth values in the range [0, 65535]. Masks should be single-channel PNGs
    containing class indices.
    """

    def __init__(
        self,
        config: DepthDatasetConfig,
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        augmentations: Optional[Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]] = None,
    ) -> None:
        self.config = config
        self.transform = transform
        self.augmentations = augmentations
        self.depth_dir = config.root / config.split / "images"
        self.mask_dir = config.root / config.split / "masks"
        self.rgb_dir = config.root / config.split / "rgb" if config.include_rgb else None
        self.frame_ids = sorted([p.stem for p in self.depth_dir.glob("*.png")])
        if not self.frame_ids:
            raise RuntimeError(f"No depth frames found in {self.depth_dir}")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.frame_ids)

    def _load_depth(self, frame_id: str) -> np.ndarray:
        depth_path = self.depth_dir / f"{frame_id}.png"
        depth = np.array(Image.open(depth_path), dtype=np.float32)
        depth /= 65535.0
        depth = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
        return depth

    def _load_mask(self, frame_id: str) -> np.ndarray:
        mask_path = self.mask_dir / f"{frame_id}.png"
        mask = np.array(Image.open(mask_path), dtype=np.int64)
        return mask

    def _load_rgb(self, frame_id: str) -> Optional[np.ndarray]:
        if self.rgb_dir is None:
            return None
        rgb_path = self.rgb_dir / f"{frame_id}.png"
        if not rgb_path.exists():
            return None
        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        return rgb

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_id = self.frame_ids[idx]
        depth = self._load_depth(frame_id)
        mask = self._load_mask(frame_id)
        rgb = self._load_rgb(frame_id)

        sample: Dict[str, np.ndarray] = {"depth": depth, "mask": mask}
        if rgb is not None:
            sample["rgb"] = rgb

        if self.augmentations is not None:
            sample = self.augmentations(sample)

        tensor_sample = self._to_tensor(sample)

        if self.transform is not None:
            tensor_sample = self.transform(tensor_sample)

        return tensor_sample

    @staticmethod
    def _to_tensor(sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        depth = sample["depth"][None, ...]  # (1, H, W)
        depth_tensor = torch.from_numpy(depth.astype(np.float32))
        mask_tensor = torch.from_numpy(sample["mask"].astype(np.int64))
        output: Dict[str, torch.Tensor] = {"depth": depth_tensor, "mask": mask_tensor}
        if "rgb" in sample:
            rgb = sample["rgb"].transpose(2, 0, 1)
            output["rgb"] = torch.from_numpy(rgb.astype(np.float32))
        return output


def default_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    depths = torch.stack([item["depth"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    output = {"depth": depths, "mask": masks}
    if "rgb" in batch[0]:
        rgbs = torch.stack([item["rgb"] for item in batch], dim=0)
        output["rgb"] = rgbs
    return output


def load_class_names(config: DepthDatasetConfig) -> List[str]:
    if config.class_names:
        return config.class_names
    metadata_path = config.root / "metadata.json"
    if metadata_path.exists():
        import json

        metadata = json.loads(metadata_path.read_text())
        if "class_names" in metadata:
            return list(metadata["class_names"])
    raise RuntimeError("Class names must be provided either in config or metadata.json")
