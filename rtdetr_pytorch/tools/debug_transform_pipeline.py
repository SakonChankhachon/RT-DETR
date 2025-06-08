import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from PIL import Image
from src.core import YAMLConfig, create

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')

# Create dummy data to test transform pipeline
dummy_img = Image.new('RGB', (640, 640), color='white')
dummy_target = {
    'boxes': torch.tensor([[320, 320, 480, 480]], dtype=torch.float32),  # xyxy pixel
    'labels': torch.tensor([0]),
    'orig_size': torch.tensor([640, 640]),
    'size': torch.tensor([640, 640]),
}

print("Original target:")
print(f"  Boxes (pixel xyxy): {dummy_target['boxes']}")

# Get transforms
transform_cfg = cfg.yaml_cfg['val_dataloader']['dataset']['transforms']
transforms = create(transform_cfg['type'], **transform_cfg)

# Apply transforms step by step
if hasattr(transforms, 'transforms'):
    for i, t in enumerate(transforms.transforms):
        print(f"\nAfter transform {i} ({type(t).__name__}):")
        dummy_img, dummy_target = t(dummy_img, dummy_target)
        if 'boxes' in dummy_target:
            print(f"  Boxes: {dummy_target['boxes']}")
            print(f"  Box shape: {dummy_target['boxes'].shape}")
else:
    dummy_img, dummy_target = transforms(dummy_img, dummy_target)
    print(f"\nAfter all transforms:")
    print(f"  Boxes: {dummy_target['boxes']}")