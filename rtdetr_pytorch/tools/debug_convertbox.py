import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from src.data.transforms import ConvertBox

# Test ConvertBox directly
print("Testing ConvertBox implementation:")

# Test case 1: Normalized xyxy -> cxcywh
test_boxes = torch.tensor([[0.5576, 0.1523, 0.6826, 0.2738]])
target = {'boxes': test_boxes}

# Create mock image
class MockImage:
    size = (640, 640)

convert_box = ConvertBox(out_fmt='cxcywh', normalize=True)
_, result = convert_box(MockImage(), target)

print(f"Input (xyxy normalized): {test_boxes}")
print(f"Output: {result['boxes']}")

# Manual calculation
x1, y1, x2, y2 = test_boxes[0]
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
w = x2 - x1
h = y2 - y1
print(f"Expected (cxcywh): [{cx:.4f}, {cy:.4f}, {w:.4f}, {h:.4f}]")

# Test case 2: Pixel coordinates
test_boxes_pixel = torch.tensor([[357, 97, 437, 175]])  # 640x640 scale
target2 = {'boxes': test_boxes_pixel}
_, result2 = convert_box(MockImage(), target2)
print(f"\nInput (xyxy pixel): {test_boxes_pixel}")
print(f"Output: {result2['boxes']}")