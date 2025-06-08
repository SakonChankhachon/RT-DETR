# tools/test_model_with_fixed_data.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from src.core import YAMLConfig

# Load config and create model
cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
model = cfg.model
model.eval()

# Load checkpoint with error handling
checkpoint_path = './output/rtdetr_r50vd_face_landmark/checkpoint0099.pth'
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to load state dict with error handling
    try:
        model.load_state_dict(checkpoint['model'])
        print("✅ Model loaded successfully!")
    except RuntimeError as e:
        print(f"❌ Error loading model: {e}")
        
        # Try loading with strict=False to see what's missing
        incompatible = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"\nMissing keys: {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")
        
        # Manually filter and load compatible weights
        model_state = model.state_dict()
        checkpoint_state = checkpoint['model']
        
        # Filter out problematic keys
        filtered_state = {}
        for k, v in checkpoint_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                try:
                    # Test if we can assign this tensor
                    test_tensor = model_state[k].clone()
                    test_tensor.copy_(v)
                    filtered_state[k] = v
                except Exception as e:
                    print(f"Skipping {k}: {e}")
        
        # Load filtered state
        model.load_state_dict(filtered_state, strict=False)
        print(f"\n✅ Loaded {len(filtered_state)}/{len(checkpoint_state)} parameters")
else:
    print(f"Checkpoint not found at {checkpoint_path}")

# Test with dummy data
print("\nTesting model with dummy data...")
dummy_input = torch.randn(1, 3, 640, 640)

with torch.no_grad():
    outputs = model(dummy_input)
    
print("\nModel outputs:")
for key, value in outputs.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
    elif isinstance(value, list):
        print(f"  {key}: list of {len(value)} items")

# Test specific components
if 'pred_landmarks' in outputs:
    print(f"\n✅ Model outputs landmarks!")
    print(f"   Shape: {outputs['pred_landmarks'].shape}")
    print(f"   Expected: [1, 300, 10] (batch_size=1, num_queries=300, num_landmarks*2=10)")
else:
    print("\n❌ No landmarks in output")

# Check if model has landmark heads
if hasattr(model, 'decoder') and hasattr(model.decoder, 'dec_landmark_heads'):
    print(f"\n✅ Model has {len(model.decoder.dec_landmark_heads)} landmark decoder heads")
    
    # Check first landmark head
    first_head = model.decoder.dec_landmark_heads[0]
    print(f"   First head type: {type(first_head).__name__}")
    
    # Check buffers
    if hasattr(first_head, 'grid_x'):
        print(f"   grid_x shape: {first_head.grid_x.shape}")
        print(f"   grid_x is contiguous: {first_head.grid_x.is_contiguous()}")

# Test postprocessor
print("\n\nTesting postprocessor...")
postprocessor = cfg.postprocessor
orig_sizes = torch.tensor([[640, 640]])

try:
    results = postprocessor(outputs, orig_sizes)
    print("✅ Postprocessor works!")
    
    if isinstance(results, list) and len(results) > 0:
        result = results[0]
        print(f"\nFirst result keys: {result.keys()}")
        
        if 'landmarks' in result:
            print(f"✅ Postprocessor outputs landmarks!")
            print(f"   Landmarks shape: {result['landmarks'].shape}")
        else:
            print("❌ No landmarks in postprocessor output")
            
except Exception as e:
    print(f"❌ Postprocessor error: {e}")
    import traceback
    traceback.print_exc()

print("\n\nDone!")