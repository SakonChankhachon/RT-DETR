import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
train_loader = cfg.train_dataloader

# Get one batch
for images, targets in train_loader:
    # Visualize first image
    img = images[0].permute(1, 2, 0).cpu().numpy()
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = img * std.numpy() + mean.numpy()
    img = (img * 255).clip(0, 255).astype('uint8')
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    
    # Draw ground truth boxes
    target = targets[0]
    H, W = img.shape[:2]
    
    if 'boxes' in target:
        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        
        print(f"Number of faces: {len(boxes)}")
        print(f"Labels: {labels}")
        print(f"Box format (cxcywh normalized): {boxes[0] if len(boxes) > 0 else 'No boxes'}")
        
        for box, label in zip(boxes, labels):
            # Convert from cxcywh to xyxy
            cx, cy, w, h = box
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            w_pix = w * W
            h_pix = h * H
            
            rect = patches.Rectangle((x1, y1), w_pix, h_pix,
                                   linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(x1, y1-5, f'Face (label={label})', color='red')
    
    plt.title('Ground Truth')
    plt.axis('off')
    plt.savefig('gt_visualization.png')
    print("Saved to gt_visualization.png")
    break