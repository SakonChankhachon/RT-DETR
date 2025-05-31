import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

import torch
from src.core import YAMLConfig

print("Loading config...")
cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')

print("Creating model...")
model = cfg.model
model.eval()

print("\nModel architecture:")
print(f"- Decoder type: {type(model.decoder).__name__}")
print(f"- Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check if model has landmark heads
if hasattr(model.decoder, 'dec_landmark_head'):
    print(f"✅ Model has landmark heads: {len(model.decoder.dec_landmark_head)} layers")
    print(f"✅ Number of landmarks: {model.decoder.num_landmarks}")
else:
    print("❌ Model does not have landmark heads")

# Test forward pass
print("\nTesting forward pass...")
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 640, 640)
    outputs = model(dummy_input)
    
print("\nModel outputs:")
for key, value in outputs.items():
    if isinstance(value, torch.Tensor):
        print(f"- {key}: shape {value.shape}")
    else:
        print(f"- {key}: {type(value)}")

if 'pred_landmarks' in outputs:
    print(f"\n✅ Model outputs landmarks!")
    print(f"   Shape: {outputs['pred_landmarks'].shape}")
    print(f"   Expected: [batch_size, num_queries, num_landmarks * 2]")
    print(f"   Actual: [{outputs['pred_landmarks'].shape[0]}, {outputs['pred_landmarks'].shape[1]}, {outputs['pred_landmarks'].shape[2]}]")
else:
    print("\n❌ No landmarks in output")

# Check criterion
print("\n\nChecking criterion...")
criterion = cfg.criterion
print(f"Criterion type: {type(criterion).__name__}")

if hasattr(criterion, 'losses'):
    print(f"Losses: {criterion.losses}")
    if 'landmarks' in criterion.losses:
        print("✅ Landmark loss is included")
    else:
        print("❌ Landmark loss is NOT included")

if hasattr(criterion, 'weight_dict'):
    print("\nLoss weights:")
    for k, v in criterion.weight_dict.items():
        print(f"  - {k}: {v}")
