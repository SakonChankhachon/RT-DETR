import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')

# Get dataset without transforms
dataset = cfg.train_dataloader.dataset
print(f"Dataset type: {type(dataset)}")

# Temporarily disable transforms
original_transforms = dataset.transforms
dataset.transforms = None

# Get raw data
idx = 0
img, target = dataset[idx]

print(f"\nRaw data (no transforms):")
print(f"  Image type: {type(img)}")
if hasattr(img, 'size'):
    print(f"  Image size: {img.size}")
print(f"  Boxes shape: {target['boxes'].shape}")
print(f"  Boxes values: {target['boxes']}")
print(f"  Boxes range: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}]")

# Restore transforms
dataset.transforms = original_transforms