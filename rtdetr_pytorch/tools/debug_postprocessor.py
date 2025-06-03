# tools/debug_postprocessor.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
model = cfg.model
postprocessor = cfg.postprocessor

# Test with dummy data
model.eval()
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 640, 640)
    outputs = model(dummy_input)
    
    # Check outputs
    print("Model outputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, range=[{v.min():.3f}, {v.max():.3f}]")
    
    # Test postprocessor
    orig_sizes = torch.tensor([[640, 640]])
    results = postprocessor(outputs, orig_sizes)
    
    print("\nPostprocessor results:")
    if isinstance(results, list):
        for i, res in enumerate(results):
            print(f"  Image {i}:")
            for k, v in res.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: shape={v.shape}")
                    if len(v) > 0:
                        if k == 'scores':
                            print(f"      range: [{v.min():.3f}, {v.max():.3f}]")
                        elif k == 'labels':
                            print(f"      unique: {v.unique()}")