import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
model = cfg.model
criterion = cfg.criterion

# Load checkpoint to check
checkpoint_path = './output/rtdetr_r50vd_face_landmark/checkpoint0098.pth'
if os.path.exists(checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint epoch: {state.get('last_epoch', 'unknown')}")
    
    # Check score head weights
    if 'model' in state:
        # Find score head weights
        for k, v in state['model'].items():
            if 'score_head' in k and 'weight' in k:
                print(f"\n{k}:")
                print(f"  Shape: {v.shape}")
                print(f"  Range: [{v.min():.3f}, {v.max():.3f}]")
                print(f"  Mean: {v.mean():.3f}")
                print(f"  Std: {v.std():.3f}")

# Test with real data
train_loader = cfg.train_dataloader
model.train()

for i, (images, targets) in enumerate(train_loader):
    # Forward pass
    outputs = model(images, targets)
    loss_dict = criterion(outputs, targets)
    
    print("\nLoss breakdown:")
    for k, v in loss_dict.items():
        if 'loss' in k:
            print(f"  {k}: {v.item():.4f}")
    
    # Check predictions
    with torch.no_grad():
        pred_scores = torch.sigmoid(outputs['pred_logits'])
        print(f"\nPrediction scores:")
        print(f"  Max: {pred_scores.max():.3f}")
        print(f"  Mean: {pred_scores.mean():.3f}")
        print(f"  % > 0.5: {(pred_scores > 0.5).float().mean()*100:.1f}%")
    
    break