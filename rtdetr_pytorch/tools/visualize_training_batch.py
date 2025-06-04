# visualize_training_batch.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
train_loader = cfg.train_dataloader

for images, targets in train_loader:
    # Visualize first 4 images
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(4, len(images))):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        
        # Denormalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = img * std + mean
        img = (img * 255).clip(0, 255).astype('uint8')
        
        axes[i].imshow(img)
        
        # Draw boxes
        if i < len(targets) and 'boxes' in targets[i]:
            boxes = targets[i]['boxes'].cpu().numpy()
            H, W = img.shape[:2]
            
            for box in boxes:
                cx, cy, w, h = box
                x1 = (cx - w/2) * W
                y1 = (cy - h/2) * H
                w_pix = w * W
                h_pix = h * H
                
                rect = patches.Rectangle((x1, y1), w_pix, h_pix,
                                       linewidth=2, edgecolor='r', facecolor='none')
                axes[i].add_patch(rect)
        
        axes[i].set_title(f'Image {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_batch_vis.png')
    print("Saved visualization to training_batch_vis.png")
    break