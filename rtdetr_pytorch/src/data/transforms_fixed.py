# src/data/transforms_fixed.py
"""
แก้ไข transforms เพื่อให้ normalization ทำงานได้ถูกต้อง
"""

import torch
import torch.nn as nn
from src.core import register

@register
class FixedConvertBox(nn.Module):
    """แก้ไข ConvertBox ให้ normalize อย่างถูกต้อง"""
    
    def __init__(self, out_fmt='', normalize=False):
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

    def forward(self, img, target):
        if target is None:
            return img, target
            
        # Get image dimensions
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            w, h = img.size
        
        # Process boxes
        if 'boxes' in target and target['boxes'].numel() > 0:
            boxes = target['boxes'].float()
            
            # Convert to cxcywh if requested
            if self.out_fmt == 'cxcywh':
                # Assume input is xyxy
                x1, y1, x2, y2 = boxes.unbind(-1)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                boxes = torch.stack([cx, cy, width, height], dim=-1)
            
            # Normalize if requested
            if self.normalize:
                # Check if already normalized
                if boxes.max() > 1.5:  # Clearly not normalized
                    boxes[:, [0, 2]] = boxes[:, [0, 2]] / w  # x, width
                    boxes[:, [1, 3]] = boxes[:, [1, 3]] / h  # y, height
                
                # Clamp to valid range
                boxes = torch.clamp(boxes, 0.0, 1.0)
            
            target['boxes'] = boxes
        
        # Process landmarks
        if 'landmarks' in target and target['landmarks'].numel() > 0:
            landmarks = target['landmarks'].float()
            
            # Normalize if requested
            if self.normalize:
                # Check if already normalized
                if landmarks.max() > 1.5:  # Clearly not normalized
                    landmarks[:, 0::2] = landmarks[:, 0::2] / w  # x coordinates
                    landmarks[:, 1::2] = landmarks[:, 1::2] / h  # y coordinates
                
                # Clamp to valid range
                landmarks = torch.clamp(landmarks, 0.0, 1.0)
            
            target['landmarks'] = landmarks
        
        return img, target


@register 
class StrictSanitizeLandmarks(nn.Module):
    """Sanitize landmarks อย่างเข้มงวด"""
    
    def __init__(self, min_val=0.0, max_val=1.0, remove_invalid=True):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.remove_invalid = remove_invalid
    
    def forward(self, img, target=None):
        if target is None or 'landmarks' not in target:
            return img, target
        
        landmarks = target['landmarks']
        if landmarks.numel() == 0:
            return img, target
        
        # Check for invalid landmarks
        if self.remove_invalid:
            x_coords = landmarks[:, 0::2]  # x coordinates
            y_coords = landmarks[:, 1::2]  # y coordinates
            
            # Find valid landmarks (mostly within bounds)
            x_valid = ((x_coords >= -0.1) & (x_coords <= 1.1)).float().mean(dim=1)
            y_valid = ((y_coords >= -0.1) & (y_coords <= 1.1)).float().mean(dim=1)
            
            # Keep faces where at least 80% of landmarks are valid
            keep = (x_valid >= 0.8) & (y_valid >= 0.8)
            
            if keep.any():
                # Filter all annotations
                for key in ['landmarks', 'boxes', 'labels', 'area', 'iscrowd']:
                    if key in target and target[key].numel() > 0:
                        target[key] = target[key][keep]
            else:
                # Create empty tensors
                num_landmarks = landmarks.shape[1]
                target['landmarks'] = torch.zeros(0, num_landmarks)
                target['boxes'] = torch.zeros(0, 4)
                target['labels'] = torch.zeros(0, dtype=torch.int64)
                target['area'] = torch.zeros(0)
                target['iscrowd'] = torch.zeros(0, dtype=torch.int64)
        
        # Clamp remaining landmarks
        if target['landmarks'].numel() > 0:
            target['landmarks'] = torch.clamp(
                target['landmarks'], self.min_val, self.max_val
            )
        
        return img, target


@register
class StrictSanitizeBoundingBox(nn.Module):
    """Sanitize bounding boxes อย่างเข้มงวด"""
    
    def __init__(self, min_size=0.01, max_size=1.0, remove_invalid=True):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.remove_invalid = remove_invalid
    
    def forward(self, img, target=None):
        if target is None or 'boxes' not in target:
            return img, target
        
        boxes = target['boxes']
        if boxes.numel() == 0:
            return img, target
        
        if self.remove_invalid:
            # Check box validity
            if boxes.shape[-1] == 4:  # cxcywh or xyxy
                if len(boxes) > 0:
                    # Assume cxcywh format after normalization
                    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    
                    # Valid boxes have positive width/height and reasonable coordinates
                    valid = (w >= self.min_size) & (h >= self.min_size) & \
                           (w <= self.max_size) & (h <= self.max_size) & \
                           (cx >= 0) & (cx <= 1) & (cy >= 0) & (cy <= 1)
                    
                    if valid.any():
                        # Filter all annotations
                        for key in ['boxes', 'landmarks', 'labels', 'area', 'iscrowd']:
                            if key in target and target[key].numel() > 0:
                                target[key] = target[key][valid]
                    else:
                        # Create empty tensors
                        target['boxes'] = torch.zeros(0, 4)
                        target['landmarks'] = torch.zeros(0, target.get('landmarks', torch.zeros(0, 10)).shape[1])
                        target['labels'] = torch.zeros(0, dtype=torch.int64)
                        target['area'] = torch.zeros(0)
                        target['iscrowd'] = torch.zeros(0, dtype=torch.int64)
        
        return img, target