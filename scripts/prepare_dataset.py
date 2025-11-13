#!/usr/bin/env python3
"""Utility script to organize raw depth camera captures into train/val/test splits.

The script expects a directory structure like:

raw_dataset/
  depth/
    frame_0001.png
    ... (16-bit PNG or .npy arrays)
  masks/
    frame_0001.png
    ... (uint8 segmentation masks)
  rgb/ (optional)
    frame_0001.png

The output directory will contain:
processed_dataset/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
  metadata.json

This script is intentionally lightweight so that students can adapt it to the
specific format provided by their 3D depth camera SDK.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

try:
    import cv2
except ImportError as exc:  # pragma: no cover - OpenCV is optional during tests
    raise SystemExit(
        "OpenCV is required for dataset preparation. Install it via `pip install opencv-python`."
    ) from exc


RNG = random.Random()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split and normalize depth dataset")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to raw dataset")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for processed dataset")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument(
        "--depth-format",
        choices=["png", "npy"],
        default="png",
        help="Depth file format. PNGs are assumed to be 16-bit single-channel.",
    )
    parser.add_argument(
        "--clip",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Optional depth clipping range in meters (after converting from millimeters if needed).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting")
    return parser.parse_args()


def list_frames(depth_dir: Path, format_suffix: str) -> List[str]:
    suffix = f".{format_suffix}"
    return sorted([p.stem for p in depth_dir.glob(f"*{suffix}")])


def load_depth(depth_path: Path, format_suffix: str) -> np.ndarray:
    if format_suffix == "png":
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Failed to read depth map: {depth_path}")
        if depth.dtype != np.uint16:
            raise ValueError(
                f"Expected 16-bit PNG depth map, got dtype {depth.dtype} at {depth_path}"
            )
        depth = depth.astype(np.float32)
    else:
        depth = np.load(depth_path).astype(np.float32)
    return depth


def normalize_depth(depth: np.ndarray, clip: Tuple[float, float] | None) -> np.ndarray:
    if clip:
        depth = np.clip(depth, clip[0], clip[1])
    finite_mask = np.isfinite(depth)
    if not finite_mask.any():
        return np.zeros_like(depth, dtype=np.float32)
    min_val = float(depth[finite_mask].min())
    max_val = float(depth[finite_mask].max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    normalized = (depth - min_val) / (max_val - min_val)
    return normalized.astype(np.float32)


def save_depth(depth: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_to_save = np.clip(depth * 65535.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), depth_to_save)


def copy_mask(mask_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    if mask.ndim != 2:
        raise ValueError(f"Expected single-channel mask for {mask_path}")
    cv2.imwrite(str(output_path), mask)


def copy_rgb(rgb_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Failed to read rgb frame: {rgb_path}")
    cv2.imwrite(str(output_path), rgb)


def split_indices(frame_ids: List[str], val_ratio: float, test_ratio: float) -> Tuple[List[str], List[str], List[str]]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")
    RNG.shuffle(frame_ids)
    total = len(frame_ids)
    val_size = int(total * val_ratio)
    test_size = int(total * test_ratio)
    val_ids = frame_ids[:val_size]
    test_ids = frame_ids[val_size : val_size + test_size]
    train_ids = frame_ids[val_size + test_size :]
    return train_ids, val_ids, test_ids


def process_split(
    split_name: str,
    frame_ids: List[str],
    depth_dir: Path,
    mask_dir: Path,
    output_root: Path,
    format_suffix: str,
    clip: Tuple[float, float] | None,
    rgb_dir: Path | None,
) -> None:
    for frame_id in tqdm(frame_ids, desc=f"Processing {split_name}"):
        depth_path = depth_dir / f"{frame_id}.{format_suffix}"
        depth = load_depth(depth_path, format_suffix)
        depth = normalize_depth(depth, clip)
        save_depth(depth, output_root / split_name / "images" / f"{frame_id}.png")
        mask_path = mask_dir / f"{frame_id}.png"
        copy_mask(mask_path, output_root / split_name / "masks" / f"{frame_id}.png")
        if rgb_dir is not None:
            rgb_path = rgb_dir / f"{frame_id}.png"
            if rgb_path.exists():
                copy_rgb(rgb_path, output_root / split_name / "rgb" / f"{frame_id}.png")


def main() -> None:
    args = parse_args()
    RNG.seed(args.seed)

    depth_dir = args.data_root / "depth"
    mask_dir = args.data_root / "masks"
    rgb_dir = args.data_root / "rgb"
    rgb_dir = rgb_dir if rgb_dir.exists() else None

    frame_ids = list_frames(depth_dir, args.depth_format)
    if not frame_ids:
        raise SystemExit(f"No depth frames found in {depth_dir}")

    train_ids, val_ids, test_ids = split_indices(frame_ids, args.val_ratio, args.test_ratio)

    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "num_frames": len(frame_ids),
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
        "clip": args.clip,
        "depth_format": args.depth_format,
    }
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2))

    for split_name, split_ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
        process_split(
            split_name,
            split_ids,
            depth_dir,
            mask_dir,
            args.output_root,
            args.depth_format,
            args.clip,
            rgb_dir,
        )


if __name__ == "__main__":
    main()
