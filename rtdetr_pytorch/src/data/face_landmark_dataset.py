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
        
        # For now, create dummy data if annotation file doesn't exist
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                self.annotations = json.load(f)
            self.image_ids = list(self.annotations.keys())
        else:
            print(f"Warning: Annotation file {ann_file} not found. Using dummy data.")
            self.annotations = {}
            self.image_ids = []
        
        # If no annotations, create dummy data
        if len(self.image_ids) == 0:
            print("Creating dummy face dataset for testing...")
            self.image_ids = ['dummy_0']
            self.annotations = {
                'dummy_0': {
                    'filename': 'dummy.jpg',
                    'boxes': [[100, 100, 200, 200]],
                    'landmarks': [[150, 120, 170, 120, 160, 140, 150, 160, 170, 160]]
                }
            }
    
    def __len__(self):
        return max(1, len(self.image_ids))  # At least 1 for testing
    
    def __getitem__(self, idx):
        if len(self.image_ids) == 0:
            # Return dummy data
            img = Image.new('RGB', (640, 640), color='white')
            target = {
                'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
                'landmarks': torch.tensor([[0.234, 0.187, 0.265, 0.187, 0.25, 0.218, 0.234, 0.25, 0.265, 0.25]], dtype=torch.float32),
                'labels': torch.zeros(1, dtype=torch.int64),
                'image_id': torch.tensor([0]),
                'orig_size': torch.tensor([640, 640]),
                'size': torch.tensor([640, 640]),
                'area': torch.tensor([10000.0]),
                'iscrowd': torch.zeros(1, dtype=torch.int64),
            }
            
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            
            return img, target
        
        img_id = self.image_ids[idx % len(self.image_ids)]
        ann = self.annotations[img_id]
        
        # Try to load image, use dummy if not found
        img_path = os.path.join(self.img_folder, ann['filename'])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
        else:
            img = Image.new('RGB', (640, 640), color='white')
            w, h = 640, 640
        
        # Prepare targets
        boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
        landmarks = torch.tensor(ann['landmarks'], dtype=torch.float32)
        
        # Normalize landmarks
        landmarks_normalized = landmarks.clone()
        landmarks_normalized[:, 0::2] /= w
        landmarks_normalized[:, 1::2] /= h
        
        labels = torch.zeros(len(boxes), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'landmarks': landmarks_normalized,
            'labels': labels,
            'image_id': torch.tensor([int(idx)]),
            'orig_size': torch.tensor([w, h]),
            'size': torch.tensor([w, h]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
        }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
