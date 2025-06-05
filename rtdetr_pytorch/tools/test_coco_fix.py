# tools/test_coco_fix.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from src.core import YAMLConfig
from src.data import get_coco_api_from_dataset

print("Testing COCO format fix...")

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
val_loader = cfg.val_dataloader

# Test the dataset output
print("\n1. Checking dataset output with transforms:")
for i, (images, targets) in enumerate(val_loader):
    target = targets[0]
    print(f"\nBatch {i}:")
    print(f"  Labels: {target['labels']}")
    print(f"  Boxes shape: {target['boxes'].shape}")
    if len(target['boxes']) > 0:
        print(f"  First box: {target['boxes'][0]}")
    break

# Check COCO conversion
print("\n2. Checking COCO API conversion:")
coco_gt = get_coco_api_from_dataset(val_loader.dataset)

print(f"  Number of images: {len(coco_gt.imgs)}")
print(f"  Number of annotations: {len(coco_gt.anns)}")
print(f"  Categories: {coco_gt.cats}")

# Check a few annotations
print("\n3. Sample COCO annotations:")
for i, (ann_id, ann) in enumerate(list(coco_gt.anns.items())[:3]):
    print(f"\nAnnotation {ann_id}:")
    print(f"  Category ID: {ann.get('category_id')}")
    print(f"  BBox: {ann.get('bbox')}")
    print(f"  Area: {ann.get('area')}")
    
    # Check if bbox is valid
    bbox = ann.get('bbox', [])
    if len(bbox) == 4:
        x, y, w, h = bbox
        print(f"  Decoded bbox: x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}")
        if w < 0 or h < 0:
            print("  ⚠️  WARNING: Negative width or height!")
        else:
            print("  ✅ Valid bbox format")

print("\n4. Testing with dummy predictions...")
from src.data import CocoEvaluator

# Create evaluator
evaluator = CocoEvaluator(coco_gt, ['bbox'])

# Create a dummy detection
dummy_results = {}
img_ids = list(coco_gt.imgs.keys())[:1]  # First image
for img_id in img_ids:
    # Get image size
    img_info = coco_gt.imgs[img_id]
    w, h = img_info['width'], img_info['height']
    
    dummy_results[img_id] = {
        'boxes': torch.tensor([[100, 100, 200, 200]]),  # xyxy format
        'scores': torch.tensor([0.9]),
        'labels': torch.tensor([1])  # Category 1
    }
    print(f"\nCreated dummy detection for image {img_id}")

try:
    evaluator.update(dummy_results)
    print("✅ Evaluator update successful!")
except Exception as e:
    print(f"❌ Evaluator error: {e}")