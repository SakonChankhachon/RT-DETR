# tools/test_coco_fix_v2.py
"""
Test script to verify COCO conversion fix
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch

# Replace the original coco_utils with the fixed version
import src.data.coco.coco_utils as original_coco_utils
import importlib

# Load the fixed utilities
print("Loading fixed COCO utilities...")
import src.data.coco.coco_utils_fixed as fixed_utils

# Monkey patch the original module with fixed functions
original_coco_utils.convert_to_coco_api = fixed_utils.convert_normalized_cxcywh_to_coco_api
original_coco_utils.get_coco_api_from_dataset = fixed_utils.get_coco_api_from_dataset

print("Testing COCO format fix...")

from src.core import YAMLConfig
from src.data import get_coco_api_from_dataset

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
val_loader = cfg.val_dataloader

# Test the dataset output
print("\n1. Checking dataset output:")
for i, (images, targets) in enumerate(val_loader):
    target = targets[0]
    print(f"\nBatch {i}:")
    print(f"  Labels: {target['labels']}")
    print(f"  Boxes shape: {target['boxes'].shape}")
    if len(target['boxes']) > 0:
        print(f"  First box (cxcywh normalized): {target['boxes'][0]}")
        print(f"  Box format: normalized={target['boxes'].max() <= 1.0}")
    break

# Check COCO conversion
print("\n2. Checking COCO API conversion:")
coco_gt = get_coco_api_from_dataset(val_loader.dataset)

print(f"  Number of images: {len(coco_gt.imgs)}")
print(f"  Number of annotations: {len(coco_gt.anns)}")
print(f"  Categories: {coco_gt.cats}")

# Check a few annotations
print("\n3. Sample COCO annotations (should be in pixel xywh format):")
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
        if w > 0 and h > 0:
            print("  ✅ Valid bbox format (positive width and height)")
        else:
            print("  ❌ WARNING: Invalid bbox!")

print("\n4. Testing with CocoEvaluator...")
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
        'labels': torch.tensor([0])  # Category 0 (face)
    }
    print(f"\nCreated dummy detection for image {img_id}")

try:
    evaluator.update(dummy_results)
    print("✅ Evaluator update successful!")
except Exception as e:
    print(f"❌ Evaluator error: {e}")
    import traceback
    traceback.print_exc()

print("\n5. Running actual model inference test...")
# Test with actual model output
model = cfg.model
model.eval()

with torch.no_grad():
    for images, targets in val_loader:
        outputs = model(images[:1])  # Just first image
        
        # Test postprocessor
        orig_sizes = torch.stack([t["orig_size"] for t in targets[:1]], dim=0)
        results = cfg.postprocessor(outputs, orig_sizes)
        
        print(f"\nModel output processed successfully!")
        if results:
            print(f"  Number of detections: {len(results[0]['boxes'])}")
        
        # Try updating evaluator with real results
        img_id = targets[0]['image_id'].item()
        try:
            evaluator.update({img_id: results[0]})
            print("  ✅ Real detection update successful!")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        break

print("\nTest complete!")