"""Dataset registry for depth segmentation."""

from .depth_dataset import DepthDatasetConfig, DepthSegmentationDataset, default_collate_fn, load_class_names

__all__ = [
    "DepthDatasetConfig",
    "DepthSegmentationDataset",
    "default_collate_fn",
    "load_class_names",
]
