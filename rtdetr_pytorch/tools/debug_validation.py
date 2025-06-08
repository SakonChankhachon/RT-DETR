import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
model = cfg.model
model.eval()

# Test with validation data
val_loader = cfg.val_dataloader

with torch.no_grad():
    for i, (images, targets) in enumerate(val_loader):
        if i > 0:
            break
            
        # Check targets format
        print("Target boxes format:")
        print(f"  Shape: {targets[0]['boxes'].shape}")
        print(f"  Range: [{targets[0]['boxes'].min():.3f}, {targets[0]['boxes'].max():.3f}]")
        print(f"  Labels: {targets[0]['labels'].unique()}")
        
        # Run model
        outputs = model(images[:1])
        
        # Check outputs
        print("\nModel outputs:")
        print(f"  Logits shape: {outputs['pred_logits'].shape}")
        print(f"  Max score: {torch.sigmoid(outputs['pred_logits']).max():.3f}")
        print(f"  Boxes range: [{outputs['pred_boxes'].min():.3f}, {outputs['pred_boxes'].max():.3f}]")
        
        # Check postprocessor
        orig_sizes = torch.stack([t["orig_size"] for t in targets[:1]], dim=0)
        results = cfg.postprocessor(outputs, orig_sizes)
        
        print(f"\nPostprocessor results:")
        print(f"  Num detections: {len(results[0]['boxes'])}")
        if len(results[0]['boxes']) > 0:
            print(f"  Max score: {results[0]['scores'].max():.3f}")