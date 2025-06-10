# tools/test_landmark_normalization.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from src.core import YAMLConfig

def test_normalization():
    print("Testing landmark normalization...")
    
    cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
    
    # Get one sample
    dataset = cfg.train_dataloader.dataset
    img, target = dataset[0]
    
    print(f"\nAfter full transform pipeline:")
    print(f"  Image type: {type(img)}")
    if torch.is_tensor(img):
        print(f"  Image shape: {img.shape}")
        print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
    
    print(f"\n  Boxes: {target['boxes']}")
    print(f"  Boxes range: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}]")
    
    if 'landmarks' in target:
        print(f"  Landmarks: {target['landmarks']}")
        print(f"  Landmarks range: [{target['landmarks'].min():.3f}, {target['landmarks'].max():.3f}]")
        
        # Check if normalized
        if target['landmarks'].max() <= 1.0:
            print("  ✅ Landmarks are properly normalized!")
        else:
            print("  ❌ Landmarks are NOT normalized!")
            print("  This will cause huge landmark loss!")
    else:
        print("  ❌ No landmarks found!")

if __name__ == '__main__':
    test_normalization()