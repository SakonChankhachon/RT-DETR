import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# tools/debug_dataset_detailed_v2.py
import torch
from src.core import YAMLConfig
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
val_dataset = cfg.val_dataloader.dataset

# Get raw data without transforms
print("1. Checking raw dataset (no transforms):")
val_dataset.transforms = None
img, target = val_dataset[0]

print(f"Image size: {img.size}")
print(f"Target keys: {target.keys()}")
print(f"Boxes shape: {target['boxes'].shape}")
print(f"Raw boxes: {target['boxes']}")
print(f"Labels: {target['labels']}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Original image with boxes
ax1.imshow(img)
ax1.set_title('Raw Dataset Output')

# Draw boxes on original image
w, h = img.size
for box in target['boxes']:
    # Box is normalized xyxy
    x1, y1, x2, y2 = box
    x1_px = x1 * w
    y1_px = y1 * h
    x2_px = x2 * w
    y2_px = y2 * h
    
    rect = patches.Rectangle((x1_px, y1_px), x2_px-x1_px, y2_px-y1_px, 
                           linewidth=2, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    
    # Add text showing normalized coordinates
    ax1.text(x1_px, y1_px-5, f'({x1:.3f}, {y1:.3f})', color='red', fontsize=8)

# Now manually apply transforms to check
print("\n2. Manual transform check:")

# Import individual transforms
from torchvision.transforms import Resize, ToTensor
from src.data.transforms import ConvertDtype, ConvertBox

# Apply Resize
resize = Resize((640, 640))
img_resized = resize(img)
print(f"After Resize: image size = {img_resized.size}")

# Apply ToTensor
to_tensor = ToTensor()
img_tensor = to_tensor(img_resized)
print(f"After ToTensor: shape = {img_tensor.shape}")

# Check ConvertBox behavior
print("\n3. Testing ConvertBox transform:")
print(f"Original boxes (xyxy normalized): {target['boxes']}")

# Create ConvertBox transform
convert_box = ConvertBox(out_fmt='cxcywh', normalize=True)

# Mock image for ConvertBox
class MockImage:
    def __init__(self, w, h):
        self.size = (w, h)

# Test with resized dimensions
mock_img = MockImage(640, 640)
_, target_converted = convert_box(mock_img, target.copy())
print(f"After ConvertBox: {target_converted['boxes']}")

# Calculate expected cxcywh
box_xyxy = target['boxes'][0]
expected_cx = (box_xyxy[0] + box_xyxy[2]) / 2
expected_cy = (box_xyxy[1] + box_xyxy[3]) / 2
expected_w = box_xyxy[2] - box_xyxy[0]
expected_h = box_xyxy[3] - box_xyxy[1]
print(f"Expected cxcywh: [{expected_cx:.3f}, {expected_cy:.3f}, {expected_w:.3f}, {expected_h:.3f}]")

# Visualize transformed
ax2.imshow(img_resized)
ax2.set_title('After Resize (640x640)')

# Draw converted box
if len(target_converted['boxes']) > 0:
    cx, cy, w, h = target_converted['boxes'][0]
    x1 = (cx - w/2) * 640
    y1 = (cy - h/2) * 640
    w_px = w * 640
    h_px = h * 640
    
    rect = patches.Rectangle((x1, y1), w_px, h_px, 
                           linewidth=2, edgecolor='g', facecolor='none')
    ax2.add_patch(rect)
    ax2.text(x1, y1-5, f'cx={cx:.3f}, cy={cy:.3f}', color='green', fontsize=8)

plt.tight_layout()
plt.savefig('debug_transforms.png')
print("\nSaved visualization to debug_transforms.png")

# Check actual dataloader output
print("\n4. Checking actual dataloader output:")
# Restore transforms
val_dataset.transforms = cfg.val_dataloader.dataset.transforms
val_loader = cfg.val_dataloader

for i, (images, targets) in enumerate(val_loader):
    print(f"\nBatch {i}:")
    print(f"  Image tensor shape: {images.shape}")
    
    for j, target in enumerate(targets[:2]):
        print(f"\n  Target {j}:")
        print(f"    Boxes: {target['boxes']}")
        print(f"    Labels: {target['labels']}")
        print(f"    orig_size: {target['orig_size']}")
        
        # Check if boxes look correct
        if len(target['boxes']) > 0:
            box = target['boxes'][0]
            print(f"    First box analysis:")
            print(f"      Values: cx={box[0]:.3f}, cy={box[1]:.3f}, w={box[2]:.3f}, h={box[3]:.3f}")
            
            # Sanity check
            if box[2] < 0.01 or box[3] < 0.01:
                print("      WARNING: Box width/height too small!")
            if box[0] < 0 or box[0] > 1 or box[1] < 0 or box[1] > 1:
                print("      WARNING: Box center outside [0,1] range!")
    
    if i >= 1:
        break