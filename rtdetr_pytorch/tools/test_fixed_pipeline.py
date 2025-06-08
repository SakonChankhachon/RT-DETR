import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from src.core import YAMLConfig
import numpy as np


cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')

# Get dataset and check output
val_dataset = cfg.val_dataloader.dataset

print("Testing fixed pipeline:")
for i in range(3):
    img, target = val_dataset[i]
    
    print(f"\nSample {i}:")
    print(f"  Image shape: {img.shape if torch.is_tensor(img) else img.size}")
    print(f"  Boxes: {target['boxes']}")
    print(f"  Box range: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}]")
    
    # Sanity check
    if len(target['boxes']) > 0:
        box = target['boxes'][0]
        print(f"  First box (cxcywh): cx={box[0]:.3f}, cy={box[1]:.3f}, w={box[2]:.3f}, h={box[3]:.3f}")
        
        # Check if values look reasonable
        if box[2] < 0.01 or box[3] < 0.01:
            print("  ⚠️  WARNING: Box too small!")
        elif box[2] > 0.9 or box[3] > 0.9:
            print("  ⚠️  WARNING: Box too large!")
        else:
            print("  ✅ Box size looks reasonable")