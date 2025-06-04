import torch
import numpy as np
from PIL import Image
from src.core import register
import json
import os

@register
class FaceLandmarkDataset(torch.utils.data.Dataset):
    """Dataset for face detection and landmark localization"""
    
    __inject__ = ['transforms']
    __share__ = ['num_landmarks']
    
    def __init__(self, img_folder, ann_file, transforms=None, num_landmarks=5, 
                 return_visibility=False):
        self.img_folder = img_folder
        self.transforms = transforms
        self.num_landmarks = num_landmarks
        self.return_visibility = return_visibility
        
        # Load annotations
        if not os.path.exists(ann_file):
            raise FileNotFoundError(
                f"Annotation file '{ann_file}' not found. "
                "Please check the dataset path.")

        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_ids = list(self.annotations.keys())
            
        # Create dummy data if needed
        if len(self.image_ids) == 0:
            print("Creating dummy face dataset for testing...")
            self.image_ids = ['dummy_0']
            self.annotations = {
                'dummy_0': {
                    'filename': 'dummy.jpg',
                    'boxes': [[100, 100, 200, 200]],  # xyxy format
                    'landmarks': [[150, 120, 170, 120, 160, 140, 150, 160, 170, 160]]
                }
            }
    
    def __len__(self):
        return max(1, len(self.image_ids))
    
    def __getitem__(self, idx):
        if len(self.image_ids) == 0:
            # Return dummy data
            img = Image.new('RGB', (640, 640), color='white')
            target = self._create_dummy_target()
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target
        
        img_id = self.image_ids[idx % len(self.image_ids)]
        ann = self.annotations[img_id]
        
        # Load image
        img_path = os.path.join(self.img_folder, ann['filename'])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
        else:
            img = Image.new('RGB', (640, 640), color='white')
            w, h = 640, 640
        
        # Process boxes - ensure correct format
        boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
        
        if boxes.numel() > 0:
            # Assume input is xyxy in pixel coordinates
            if boxes.shape[1] == 4:
                # Normalize to [0, 1]
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w) / w
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h) / h
                
                # Convert xyxy to cxcywh
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                width = (x2 - x1).clamp(min=0)
                height = (y2 - y1).clamp(min=0)
                
                boxes = torch.stack([cx, cy, width, height], dim=1)
        
        # Process landmarks
        landmarks = torch.tensor(ann['landmarks'], dtype=torch.float32)
        if landmarks.numel() > 0:
            # Normalize landmarks
            landmarks = landmarks.reshape(-1, self.num_landmarks * 2)
            landmarks[:, 0::2] = landmarks[:, 0::2] / w
            landmarks[:, 1::2] = landmarks[:, 1::2] / h
        
        # Create labels (all faces are class 0)
        labels = torch.zeros(len(boxes), dtype=torch.int64)
        
        # Calculate area in normalized coordinates
        if boxes.numel() > 0:
            area = boxes[:, 2] * boxes[:, 3]  # width * height in normalized coords
        else:
            area = torch.zeros(0, dtype=torch.float32)
        
        target = {
            'boxes': boxes,
            'landmarks': landmarks,
            'labels': labels,
            'image_id': torch.tensor([int(idx)]),
            'orig_size': torch.tensor([w, h]),
            'size': torch.tensor([w, h]),
            'area': area,
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
        }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def _create_dummy_target(self):
        """Create dummy target for testing"""
        # Box in normalized cxcywh format
        boxes = torch.tensor([[0.5, 0.5, 0.3, 0.3]], dtype=torch.float32)
        landmarks = torch.tensor([[0.45, 0.45, 0.55, 0.45, 0.5, 0.5, 0.45, 0.55, 0.55, 0.55]], 
                                dtype=torch.float32)
        
        return {
            'boxes': boxes,
            'landmarks': landmarks,
            'labels': torch.zeros(1, dtype=torch.int64),
            'image_id': torch.tensor([0]),
            'orig_size': torch.tensor([640, 640]),
            'size': torch.tensor([640, 640]),
            'area': torch.tensor([0.09]),  # 0.3 * 0.3
            'iscrowd': torch.zeros(1, dtype=torch.int64),
        }