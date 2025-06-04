import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
train_loader = cfg.train_dataloader

# Test one batch
for images, targets in train_loader:
    print("Batch info:")
    print(f"  Images: {images.shape}")
    print(f"  Targets: {len(targets)}")
    
    # Check first target
    t = targets[0]
    print(f"\nFirst target:")
    print(f"  Boxes shape: {t['boxes'].shape}")
    print(f"  Boxes range: [{t['boxes'].min():.3f}, {t['boxes'].max():.3f}]")
    print(f"  First box (cxcywh): {t['boxes'][0]}")
    
    # Validate boxes
    if t['boxes'].numel() > 0:
        cx, cy, w, h = t['boxes'][0]
        print(f"\n  Box breakdown:")
        print(f"    Center: ({cx:.3f}, {cy:.3f})")
        print(f"    Size: ({w:.3f}, {h:.3f})")
        
        # Convert back to xyxy to check
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        print(f"    As xyxy: ({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})")
        
        # Check validity
        if x2 > x1 and y2 > y1:
            print("    ✅ Valid box")
        else:
            print("    ❌ Invalid box!")
    
    break