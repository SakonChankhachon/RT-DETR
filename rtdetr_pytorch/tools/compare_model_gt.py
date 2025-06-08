import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
import matplotlib.pyplot as plt
from src.core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
model = cfg.model
model.eval()

# Get one batch
val_loader = cfg.val_dataloader
for images, targets in val_loader:
    # Check ground truth
    print("Ground Truth:")
    for i, target in enumerate(targets[:2]):
        print(f"\nImage {i}:")
        print(f"  Boxes shape: {target['boxes'].shape}")
        print(f"  Boxes: {target['boxes']}")
        print(f"  Labels: {target['labels']}")
    
    # Run model
    with torch.no_grad():
        outputs = model(images)
        
    print("\nModel Outputs:")
    print(f"  Logits shape: {outputs['pred_logits'].shape}")
    print(f"  Boxes shape: {outputs['pred_boxes'].shape}")
    
    # Check top predictions
    scores = torch.sigmoid(outputs['pred_logits'])
    for i in range(min(2, len(images))):
        top_scores, top_idx = scores[i, :, 0].topk(5)
        print(f"\nImage {i} - Top 5 predictions:")
        for j, (score, idx) in enumerate(zip(top_scores, top_idx)):
            box = outputs['pred_boxes'][i, idx]
            print(f"  {j+1}. Score: {score:.3f}, Box: {box}")
    
    break