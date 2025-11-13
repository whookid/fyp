# 3D Depth Camera Semantic Segmentation Project Design

## Project Overview
This project provides a full-stack research and engineering template for a final-year
project (FYP) that explores semantic segmentation using data collected from a
3D depth camera. The focus is on building a reproducible experimentation
framework that supports:

- Efficient ingestion of raw depth maps (single-channel or RGB-D) exported from
  commodity sensors such as Intel RealSense, Azure Kinect, or ZED cameras.
- Modern neural architectures tailored to depth data with minimal modifications
  so that pre-trained RGB models can be leveraged.
- End-to-end training, validation, and inference pipelines that can be executed
  on a single GPU workstation.
- Experiment tracking, hyper-parameter management, and reporting utilities to
  help communicate results in the final dissertation.

The repository is structured so that the theoretical design, data processing,
model development, experimentation, and evaluation are cleanly separated. Each
module can be iterated upon independently during the research lifecycle.

## System Architecture
```
┌───────────────────────┐
│   Data Acquisition     │
│  (Depth camera & GT)   │
└──────────┬────────────┘
           │ Raw frames (depth + mask)
┌──────────▼────────────┐
│   Data Management      │
│ (Pre-processing, split │
│  scripts, metadata)    │
└──────────┬────────────┘
           │ PyTorch Dataset
┌──────────▼────────────┐
│  Training Pipeline     │
│ (UNet/DeepLab backbones│
│  w/ depth augmentations)│
└──────────┬────────────┘
           │ Checkpoints, metrics
┌──────────▼────────────┐
│   Evaluation & Demo    │
│ (Quantitative + visual │
│  analytics, inference) │
└────────────────────────┘
```

### Data Layer
- **scripts/prepare_dataset.py** converts raw depth sequences and ground-truth
  masks into train/val/test splits, optionally generating metadata such as mean
  depth statistics and class distribution histograms.
- **src/datasets/depth_dataset.py** defines a PyTorch `Dataset` that reads depth
  frames (16-bit PNG, TIFF, or NumPy arrays) and segmentation masks. It supports
  paired RGB frames when available and provides hooks for augmentation.

### Model Layer
- **src/models/unet.py** implements a light-weight UNet that accepts configurable
  input channels and produces multi-class logits. The module structure keeps the
  encoder/decoder blocks modular so that future depth-aware components (e.g.,
  squeeze-and-excitation layers) can be inserted with minimal refactoring.
- Additional architectures can be added under `src/models/` and exposed via the
  model registry in `src/models/__init__.py`.

### Training & Evaluation
- **src/train.py** orchestrates the end-to-end training loop with configuration
  files in YAML format (`configs/*.yaml`). The trainer handles experiment
  reproducibility (seed setting), logging, and checkpointing.
- **src/eval.py** performs validation on saved checkpoints and computes
  pixel-accuracy, mean IoU, and class-wise IoUs using utilities in
  `src/utils/metrics.py`.
- **src/predict.py** provides a command-line tool to run inference on a single
  frame or directory of frames, producing colorized segmentation overlays.

### Experiment Tracking & Reporting
- Metrics and configuration metadata are serialized to JSON for downstream
  analysis in Jupyter notebooks. Optional integration hooks for Weights & Biases
  or TensorBoard are exposed in `src/utils/logger.py`.
- `docs/reporting.md` (to be authored during experimentation) will summarize key
  findings, ablations, and visualizations for direct inclusion in the final FYP
  thesis.

## Data Flow
1. **Acquisition**: Use the depth camera SDK to export frames (e.g., 16-bit PNG)
   and annotate segmentation masks using tools like LabelMe or Supervisely.
2. **Preparation**: Run `python scripts/prepare_dataset.py --data-root ...` to
   normalize depth values, remove outliers, and split into train/val/test.
3. **Training**: Launch `python src/train.py --config configs/depth_unet.yaml` to
   train the model. Checkpoints and logs are stored under `runs/<experiment_name>`.
4. **Evaluation**: Run `python src/eval.py --checkpoint runs/.../best.ckpt` to
   compute metrics on the validation/test sets.
5. **Inference/Demo**: Use `python src/predict.py --input demo/frame.png` to
   generate visual overlays suitable for presentation.

## Key Algorithms
- **Depth Normalization**: Per-scan min/max normalization with optional clipping
  to sensor-specific valid ranges (e.g., 0.2–10 m). Implemented in
  `scripts/prepare_dataset.normalize_depth`.
- **Augmentation**: Albumentations-based pipeline including random scaling,
  rotation, elastic deformation, and depth-specific noise (Gaussian, speckle).
- **Model Architecture**: Encoder-decoder UNet with residual depth encoder and
  attention gates to capture fine geometry cues.
- **Optimization**: Combination of Dice Loss and Cross-Entropy to mitigate class
  imbalance.

## Milestones & Deliverables
1. **Week 1–2**: Acquire dataset, validate file formats, implement dataset
   loader and visualization utilities.
2. **Week 3–4**: Implement baseline UNet, run sanity check training on a subset
   of data.
3. **Week 5–6**: Integrate augmentations and experiment with hyper-parameters.
4. **Week 7–8**: Add advanced backbones (DeepLabV3, transformer-based models).
5. **Week 9**: Conduct ablation studies, evaluate on test set.
6. **Week 10**: Prepare thesis figures, final report, and deployment demo.

## Future Extensions
- Fuse RGB and depth modalities using early or late fusion strategies.
- Integrate temporal information from depth video streams using ConvLSTMs.
- Deploy trained models to embedded hardware (Jetson Xavier) for real-time demos.
- Explore self-supervised pre-training using contrastive objectives on unlabeled
  depth scans.

