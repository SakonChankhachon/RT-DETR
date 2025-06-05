# tools/debug_validation_data.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from src.core import YAMLConfig
import numpy as np

print("Debugging validation data...")

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
val_loader = cfg.val_dataloader

print(f"\nValidation dataset info:")
print(f"  Dataset size: {len(val_loader.dataset)}")
print(f"  Batch size: {val_loader.batch_size}")

# Check a few validation samples
for i, (images, targets) in enumerate(val_loader):
    print(f"\nBatch {i}:")
    print(f"  Images shape: {images.shape}")
    print(f"  Number of targets: {len(targets)}")
    
    # Check first target in detail
    if len(targets) > 0:
        target = targets[0]
        print(f"\n  First target keys: {target.keys()}")
        
        if 'boxes' in target:
            boxes = target['boxes']
            print(f"    Boxes shape: {boxes.shape}")
            print(f"    Boxes dtype: {boxes.dtype}")
            print(f"    Boxes range: [{boxes.min():.3f}, {boxes.max():.3f}]")
            if len(boxes) > 0:
                print(f"    First box: {boxes[0]}")
                print(f"    Box format check (should be cxcywh normalized):")
                cx, cy, w, h = boxes[0]
                print(f"      Center: ({cx:.3f}, {cy:.3f})")
                print(f"      Size: ({w:.3f}, {h:.3f})")
        
        if 'labels' in target:
            labels = target['labels']
            print(f"    Labels shape: {labels.shape}")
            print(f"    Unique labels: {labels.unique()}")
            print(f"    All labels are 0 (face)? {(labels == 0).all()}")
        
        if 'image_id' in target:
            print(f"    Image ID: {target['image_id']}")
    
    # Check more samples
    total_boxes = sum(len(t['boxes']) for t in targets if 'boxes' in t)
    print(f"\n  Total boxes in batch: {total_boxes}")
    print(f"  Average boxes per image: {total_boxes / len(targets):.2f}")
    
    if i >= 2:  # Check first 3 batches
        break

# Check the CocoEvaluator expectations
print("\n\nChecking COCO API compatibility...")
from src.data import get_coco_api_from_dataset

try:
    coco_gt = get_coco_api_from_dataset(val_loader.dataset)
    print(f"COCO GT created successfully")
    print(f"Number of images: {len(coco_gt.imgs)}")
    print(f"Number of annotations: {len(coco_gt.anns)}")
    
    # Check a few annotations
    ann_ids = list(coco_gt.anns.keys())[:5]
    for ann_id in ann_ids:
        ann = coco_gt.anns[ann_id]
        print(f"\nAnnotation {ann_id}:")
        print(f"  Category ID: {ann.get('category_id', 'N/A')}")
        print(f"  BBox: {ann.get('bbox', 'N/A')}")
        print(f"  Area: {ann.get('area', 'N/A')}")
        
except Exception as e:
    print(f"Error creating COCO GT: {e}")
    import traceback
    traceback.print_exc()