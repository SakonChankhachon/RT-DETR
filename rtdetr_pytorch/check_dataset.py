# check_dataset.py
import torch
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
train_loader = cfg.train_dataloader

print("Checking dataset...")
for i, (images, targets) in enumerate(train_loader):
    print(f"\nBatch {i}:")
    print(f"- Images shape: {images.shape}")
    
    for j, target in enumerate(targets):
        print(f"\n  Target {j}:")
        for key, value in target.items():
            if isinstance(value, torch.Tensor):
                print(f"    - {key}: shape {value.shape}")
        
        if 'landmarks' in target:
            print(f"    ✅ Has landmarks!")
            print(f"    First face landmarks: {target['landmarks'][0] if len(target['landmarks']) > 0 else 'No faces'}")
        else:
            print(f"    ❌ No landmarks in target!")
    
    if i >= 2:  # Check first 3 batches
        break