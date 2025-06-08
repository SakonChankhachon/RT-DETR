import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
import matplotlib.pyplot as plt

# Load checkpoint
checkpoint = torch.load('./output/rtdetr_r50vd_face_landmark/checkpoint0099.pth', 
                       map_location='cpu')

# Check score head weights
model_state = checkpoint['model']
for k, v in model_state.items():
    if 'dec_score_head' in k and 'weight' in k:
        print(f"{k}:")
        print(f"  Mean: {v.mean():.3f}, Std: {v.std():.3f}")
        print(f"  Range: [{v.min():.3f}, {v.max():.3f}]")

# Visualize score predictions
if 'dec_score_head.5.weight' in model_state:
    weights = model_state['dec_score_head.5.weight']
    plt.figure(figsize=(10, 6))
    plt.hist(weights.flatten().numpy(), bins=50)
    plt.title('Score Head Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.savefig('score_head_weights.png')
    print("Saved weight distribution plot")