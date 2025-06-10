# tools/debug_landmark_loss.py
"""
Debug why landmark loss is so high
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
from src.core import YAMLConfig

def debug_landmark_loss():
    print("🔍 Debugging Landmark Loss...")
    
    # Load config
    cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
    model = cfg.model
    criterion = cfg.criterion
    
    # Load checkpoint
    checkpoint_path = './output/rtdetr_r50vd_face_landmark_v2/checkpoint0149.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('last_epoch', '?')}")
    else:
        print("⚠️ No checkpoint found, using random weights")
    
    # Get one batch of data
    train_loader = cfg.train_dataloader
    for images, targets in train_loader:
        break
    
    print(f"\n📊 Data Analysis:")
    print(f"  Batch size: {len(images)}")
    print(f"  Image shape: {images.shape}")
    
    # Analyze targets
    for i, target in enumerate(targets[:2]):
        print(f"\n  Target {i}:")
        print(f"    Boxes shape: {target['boxes'].shape}")
        print(f"    Boxes range: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}]")
        
        if 'landmarks' in target:
            lmks = target['landmarks']
            print(f"    Landmarks shape: {lmks.shape}")
            print(f"    Landmarks range: [{lmks.min():.3f}, {lmks.max():.3f}]")
            print(f"    First landmark set: {lmks[0] if len(lmks) > 0 else 'None'}")
        else:
            print(f"    ❌ No landmarks in target!")
    
    # Test model forward pass
    print(f"\n🤖 Model Forward Pass:")
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    
    print(f"  Output keys: {outputs.keys()}")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
    
    # Check predictions
    if 'pred_landmarks' in outputs:
        pred_lmks = outputs['pred_landmarks']
        print(f"\n  Predicted landmarks analysis:")
        print(f"    Shape: {pred_lmks.shape}")
        print(f"    Range: [{pred_lmks.min():.3f}, {pred_lmks.max():.3f}]")
        print(f"    Mean: {pred_lmks.mean():.3f}")
        print(f"    Std: {pred_lmks.std():.3f}")
        
        # Check if landmarks are reasonable
        if pred_lmks.max() > 10 or pred_lmks.min() < -10:
            print("    ⚠️ Landmark values seem unreasonable!")
        else:
            print("    ✅ Landmark values seem reasonable")
    else:
        print("    ❌ No pred_landmarks in outputs!")
    
    # Test loss computation
    print(f"\n💸 Loss Computation:")
    model.train()
    outputs = model(images, targets)
    loss_dict = criterion(outputs, targets)
    
    print(f"  Loss breakdown:")
    for key, value in loss_dict.items():
        if 'loss' in key:
            print(f"    {key}: {value.item():.4f}")
    
    # Focus on landmark loss
    if 'loss_landmarks' in loss_dict:
        landmark_loss = loss_dict['loss_landmarks'].item()
        print(f"\n  🎯 Landmark Loss Analysis:")
        print(f"    Value: {landmark_loss:.4f}")
        
        if landmark_loss > 1000:
            print("    ❌ PROBLEM: Landmark loss is extremely high!")
            print("    Possible causes:")
            print("      1. Landmarks not normalized properly")
            print("      2. Loss function scale issue")
            print("      3. Target format mismatch")
            
            # Check loss weights
            print(f"\n    Loss weights in criterion:")
            for k, v in criterion.weight_dict.items():
                print(f"      {k}: {v}")
        else:
            print("    ✅ Landmark loss seems reasonable")
    
    # Check if criterion has the right loss function
    print(f"\n🔧 Criterion Analysis:")
    print(f"  Criterion type: {type(criterion).__name__}")
    print(f"  Losses: {criterion.losses}")
    print(f"  Number of landmarks: {getattr(criterion, 'num_landmarks', 'Not set')}")
    
    # Test landmark loss function directly
    if hasattr(criterion, 'loss_landmarks'):
        print(f"\n🧪 Direct Landmark Loss Test:")
        
        # Create dummy data
        dummy_indices = [(torch.tensor([0]), torch.tensor([0]))]  # Match first target
        dummy_num_boxes = 1
        
        try:
            landmark_loss_dict = criterion.loss_landmarks(
                outputs, targets, dummy_indices, dummy_num_boxes
            )
            print(f"  Direct loss result: {landmark_loss_dict}")
        except Exception as e:
            print(f"  ❌ Error in direct loss: {e}")

def check_data_format():
    """Check if data format is causing the issue"""
    print(f"\n📋 Checking Data Format...")
    
    cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
    dataset = cfg.train_dataloader.dataset
    
    # Check raw data (no transforms)
    dataset.transforms = None
    img, target = dataset[0]
    
    print(f"  Raw data format:")
    print(f"    Image: {type(img)}, size: {img.size if hasattr(img, 'size') else 'N/A'}")
    print(f"    Boxes: {target['boxes']}")
    print(f"    Landmarks: {target.get('landmarks', 'Not found')}")
    
    # Check after transforms
    dataset.transforms = cfg.train_dataloader.dataset.transforms
    img, target = dataset[0]
    
    print(f"\n  After transforms:")
    print(f"    Image: {type(img)}, shape: {img.shape if hasattr(img, 'shape') else 'N/A'}")
    print(f"    Boxes: {target['boxes']}")
    print(f"    Landmarks: {target.get('landmarks', 'Not found')}")

def fix_loss_weights():
    """Suggest loss weight fixes"""
    print(f"\n🔧 Suggested Loss Weight Fixes:")
    
    print(f"""
The landmark loss is too high. Try these fixes:

1. **Reduce landmark loss weight**:
   In your config file, change:
   ```yaml
   weight_dict: {{
     loss_vfl: 1.0,
     loss_bbox: 5.0,
     loss_giou: 2.0,
     loss_landmarks: 0.1,  # ← Reduce from 5.0 to 0.1
   }}
   ```

2. **Check landmark normalization**:
   Ensure landmarks are in [0,1] range, not pixel coordinates.

3. **Use different loss function**:
   Try SmoothL1Loss instead of L1Loss for landmarks.

4. **Add gradient clipping**:
   ```yaml
   clip_max_norm: 1.0  # ← Add this
   ```

5. **Check coordinate system**:
   Make sure predicted and target landmarks use the same coordinate system.
""")

if __name__ == '__main__':
    debug_landmark_loss()
    check_data_format()
    fix_loss_weights()