# 3D Depth Camera Semantic Segmentation

This repository contains a complete starter kit for a final-year project focused
on semantic segmentation using data captured from a 3D depth camera. It
includes:

- A detailed architectural design document outlining the research workflow.
- Data preparation utilities for converting raw depth captures into curated
  training, validation, and test splits.
- Modular PyTorch implementations of depth-aware segmentation models (UNet by
  default) with extensible registries for future architectures.
- Training, evaluation, and inference scripts with reproducible experiment
  logging.

## Repository Structure

```
.
├── configs/               # YAML experiment configurations
├── docs/                  # Project design and future reporting assets
├── scripts/               # Dataset preparation utilities
├── src/
│   ├── datasets/          # Dataset definitions and transforms
│   ├── models/            # Model implementations and registry
│   ├── utils/             # Training/evaluation helpers
│   ├── eval.py            # Evaluation entry point
│   ├── predict.py         # Inference script
│   └── train.py           # Training entry point
└── README.md
```

## Getting Started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare the dataset**

   Organize raw captures as described in `docs/design.md`, then run:

   ```bash
   python scripts/prepare_dataset.py \
       --data-root path/to/raw_dataset \
       --output-root data/processed_dataset \
       --depth-format png
   ```

3. **Train the model**

   ```bash
   export PYTHONPATH=src:$PYTHONPATH
   python src/train.py --config configs/depth_unet.yaml
   ```

4. **Evaluate a checkpoint**

   ```bash
   export PYTHONPATH=src:$PYTHONPATH
   python src/eval.py --config configs/depth_unet.yaml --checkpoint runs/depth-unet-baseline/best.ckpt
   ```

5. **Run inference on new frames**

   ```bash
   export PYTHONPATH=src:$PYTHONPATH
   python src/predict.py \
       --checkpoint runs/depth-unet-baseline/best.ckpt \
       --input demo/frame.png \
       --output outputs/ \
       --num-classes 4
   ```

## Requirements

The provided `requirements.txt` lists the baseline dependencies (adjust
versions according to your hardware):

```
torch
torchvision
pyyaml
albumentations
opencv-python
numpy
Pillow
tqdm
```

## Next Steps

- Extend `src/models/` with additional architectures (e.g., DeepLabV3).
- Integrate TensorBoard or Weights & Biases for richer experiment tracking.
- Build a simple Streamlit or FastAPI dashboard to visualize segmentation
  overlays in real-time.
- Document experiments and key findings in `docs/reporting.md` as you progress.
