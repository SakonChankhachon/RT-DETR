# tools/check_val_data.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
val_loader = cfg.val_dataloader

print(f"Validation dataset size: {len(val_loader.dataset)}")
print(f"Batch size: {val_loader.batch_size}")

for i, (images, targets) in enumerate(val_loader):
    print(f"\nBatch {i}:")
    print(f"  Images shape: {images.shape}")
    
    for j, target in enumerate(targets):
        print(f"  Target {j}:")
        for k, v in target.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                if k == 'labels':
                    print(f"      unique values: {v.unique()}")
                    print(f"      all zeros? {(v == 0).all()}")
                elif k == 'boxes':
                    print(f"      range: [{v.min():.3f}, {v.max():.3f}]")
                    if v.shape[0] > 0:
                        print(f"      first box: {v[0]}")
                elif k == 'landmarks':
                    print(f"      range: [{v.min():.3f}, {v.max():.3f}]")
    
    if i >= 2:
        break