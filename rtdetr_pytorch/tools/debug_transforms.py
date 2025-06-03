import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
val_loader = cfg.val_dataloader

print("Checking validation transforms...")
dataset = val_loader.dataset
print(f"Dataset type: {type(dataset)}")
print(f"Transforms: {dataset.transforms}")

# Get raw data before transforms
if hasattr(dataset, 'prepare'):
    # Get one sample
    idx = 0
    img, target = dataset.coco.loadImgs(dataset.ids[idx])[0], dataset.coco.loadAnns(dataset.coco.getAnnIds(dataset.ids[idx]))
    image_id = dataset.ids[idx]
    target = {'image_id': image_id, 'annotations': target}
    img, target = dataset.prepare(img, target)
    
    print("\nBefore transforms:")
    print(f"  Image size: {img.size if hasattr(img, 'size') else 'N/A'}")
    print(f"  Boxes shape: {target['boxes'].shape}")
    print(f"  Boxes range: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}]")
    print(f"  First box: {target['boxes'][0] if len(target['boxes']) > 0 else 'No boxes'}")

# Get transformed data
for i, (images, targets) in enumerate(val_loader):
    print("\nAfter transforms:")
    target = targets[0]
    print(f"  Image shape: {images[0].shape}")
    print(f"  Boxes shape: {target['boxes'].shape}")
    print(f"  Boxes range: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}]")
    print(f"  First box: {target['boxes'][0]}")
    
    # Check if normalized
    if target['boxes'].max() > 1.0:
        print("  ❌ ERROR: Boxes are NOT normalized!")
    else:
        print("  ✅ Boxes are normalized")
    break