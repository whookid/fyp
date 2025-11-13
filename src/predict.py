"""Run inference on depth frames using a trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models import get_model
from utils.train_utils import load_checkpoint


COLOR_MAP = np.array(
    [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run depth segmentation inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Input file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Directory to save predictions")
    parser.add_argument("--model-name", type=str, default="unet")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_depth(path: Path) -> torch.Tensor:
    depth = np.array(Image.open(path), dtype=np.float32) / 65535.0
    tensor = torch.from_numpy(depth[None, None, ...])
    return tensor


def colorize_mask(mask: np.ndarray, num_classes: int) -> Image.Image:
    palette = COLOR_MAP.copy()
    if num_classes > len(palette):
        extra = np.random.randint(0, 255, size=(num_classes - len(palette), 3), dtype=np.uint8)
        palette = np.concatenate([palette, extra], axis=0)
    colored = palette[mask]
    return Image.fromarray(colored)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    model = get_model(args.model_name, in_channels=1, num_classes=args.num_classes).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if args.input.is_file():
        input_paths = [args.input]
    else:
        input_paths = sorted(args.input.glob("*.png"))
    if not input_paths:
        raise SystemExit("No input files found")

    args.output.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for input_path in input_paths:
            depth = load_depth(input_path).to(device)
            logits = model(depth)
            preds = torch.argmax(logits, dim=1).cpu().numpy()[0]
            mask_img = Image.fromarray(preds.astype(np.uint8))
            overlay = colorize_mask(preds, args.num_classes)
            mask_img.save(args.output / f"{input_path.stem}_mask.png")
            overlay.save(args.output / f"{input_path.stem}_overlay.png")


if __name__ == "__main__":
    main()
