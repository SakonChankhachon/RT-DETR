# tools/check_dataset_format.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
val_dataset = cfg.val_dataloader.dataset

# Check without transforms
print("Checking raw dataset output (no transforms)...")
val_dataset.transforms = None

# Get one sample
img, target = val_dataset[0]

print(f"\nRaw target (before transforms):")
for key, value in target.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        if key == 'boxes' and value.numel() > 0:
            print(f"    First box: {value[0]}")
            print(f"    Box range: [{value.min():.1f}, {value.max():.1f}]")
        elif key == 'labels':
            print(f"    Values: {value}")