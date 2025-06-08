import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import json
import os
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
val_dataset = cfg.val_dataloader.dataset

# Load raw annotations
ann_file = val_dataset.ann_file
with open(ann_file, 'r') as f:
    annotations = json.load(f)

print(f"Total images: {len(annotations)}")

# Check first few annotations
for i, (img_id, ann) in enumerate(list(annotations.items())[:3]):
    print(f"\nImage {img_id}:")
    print(f"  Filename: {ann['filename']}")
    print(f"  Boxes: {ann['boxes']}")
    print(f"  Landmarks: {ann.get('landmarks', 'No landmarks')[:50]}...")  # First 50 chars
    
    # Analyze box format
    if ann['boxes']:
        box = ann['boxes'][0]
        print(f"  First box: {box}")
        if len(box) == 4:
            if box[2] < 1 and box[3] < 1:
                print("  -> Looks like normalized cxcywh format")
            elif box[2] > box[0] and box[3] > box[1]:
                print("  -> Looks like xyxy format")
            else:
                print("  -> Format unclear")