# tools/comprehensive_landmark_debug.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from src.core import YAMLConfig

def main():
    print("🔍 Comprehensive Landmark Debug")
    print("=" * 50)
    
    cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
    
    # Test 1: Single sample with transforms
    print("\n1️⃣ Testing single sample normalization:")
    dataset = cfg.train_dataloader.dataset
    img, target = dataset[0]
    
    print(f"  Image: {type(img)}, shape: {img.shape if torch.is_tensor(img) else 'PIL'}")
    print(f"  Boxes range: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}]")
    if 'landmarks' in target:
        print(f"  Landmarks range: [{target['landmarks'].min():.3f}, {target['landmarks'].max():.3f}]")
        normalized = target['landmarks'].max() <= 1.0
        print(f"  Landmarks normalized: {'✅' if normalized else '❌'}")
    
    # Test 2: Batch from dataloader
    print("\n2️⃣ Testing batch from dataloader:")
    train_loader = cfg.train_dataloader
    for images, targets in train_loader:
        print(f"  Batch size: {len(targets)}")
        
        # Check all targets in batch
        all_normalized = True
        landmark_ranges = []
        
        for i, target in enumerate(targets[:3]):  # Check first 3
            if 'landmarks' in target and target['landmarks'].numel() > 0:
                lmk_min, lmk_max = target['landmarks'].min().item(), target['landmarks'].max().item()
                landmark_ranges.append((lmk_min, lmk_max))
                if lmk_max > 1.0:
                    all_normalized = False
                    print(f"    Target {i}: landmarks [{lmk_min:.3f}, {lmk_max:.3f}] ❌")
                else:
                    print(f"    Target {i}: landmarks [{lmk_min:.3f}, {lmk_max:.3f}] ✅")
        
        print(f"  All landmarks normalized: {'✅' if all_normalized else '❌'}")
        break
    
    # Test 3: Model prediction vs target alignment
    print("\n3️⃣ Testing model prediction alignment:")
    model = cfg.model
    model.eval()
    
    with torch.no_grad():
        outputs = model(images[:2])  # Just 2 samples
        
    pred_landmarks = outputs['pred_landmarks']
    print(f"  Predicted landmarks range: [{pred_landmarks.min():.3f}, {pred_landmarks.max():.3f}]")
    
    # Compare with first target
    if len(targets) > 0 and 'landmarks' in targets[0]:
        target_landmarks = targets[0]['landmarks']
        print(f"  Target landmarks range: [{target_landmarks.min():.3f}, {target_landmarks.max():.3f}]")
        
        # Check coordinate alignment
        pred_sample = pred_landmarks[0, 0, :]  # First query of first image
        target_sample = target_landmarks[0, :]  # First face of first image
        
        diff = torch.abs(pred_sample - target_sample).mean()
        print(f"  Mean coordinate difference: {diff:.3f}")
        
        if diff < 0.5:
            print("  ✅ Predictions and targets are in similar coordinate space")
        else:
            print("  ⚠️ Large difference between predictions and targets")
    
    # Test 4: Loss computation test
    print("\n4️⃣ Testing loss computation:")
    criterion = cfg.criterion
    
    try:
        with torch.no_grad():
            loss_dict = criterion(outputs, targets[:2])
        
        landmark_loss = loss_dict.get('loss_landmarks', 0)
        print(f"  Landmark loss: {landmark_loss:.4f}")
        
        if landmark_loss < 50:
            print("  ✅ Landmark loss is reasonable")
        elif landmark_loss < 500:
            print("  ⚠️ Landmark loss is high but manageable")
        else:
            print("  ❌ Landmark loss is extremely high")
            
    except Exception as e:
        print(f"  ❌ Error computing loss: {e}")
    
    # Test 5: Config verification
    print("\n5️⃣ Verifying config settings:")
    criterion_cfg = cfg.yaml_cfg.get('PolarLandmarkCriterion', {})
    weight_dict = criterion_cfg.get('weight_dict', {})
    
    landmark_weight = weight_dict.get('loss_landmarks', 'Not set')
    print(f"  Landmark loss weight: {landmark_weight}")
    
    if isinstance(landmark_weight, (int, float)) and landmark_weight <= 1.0:
        print("  ✅ Landmark weight is reasonable")
    else:
        print("  ⚠️ Consider reducing landmark weight")
    
    # Summary
    print("\n📋 Summary and Recommendations:")
    
    if all_normalized and landmark_loss < 50:
        print("  ✅ Everything looks good! Training should be stable.")
    elif all_normalized:
        print("  ✅ Normalization is correct")
        print("  💡 Consider reducing landmark loss weight for better balance")
    else:
        print("  ❌ Normalization issues detected")
        print("  🔧 Check transform pipeline in dataloader config")
    
    print("\n🎯 Recommended loss weights for current setup:")
    print("  loss_vfl: 1.0")
    print("  loss_bbox: 5.0") 
    print("  loss_giou: 2.0")
    print("  loss_landmarks: 1.0  # Adjust based on loss magnitude")

if __name__ == '__main__':
    main()