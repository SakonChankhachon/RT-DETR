# tools/test_emergency_fix.py
"""
ทดสอบ emergency fix สำหรับลด validation loss
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from src.core import YAMLConfig

def test_emergency_fix():
    """ทดสอบการแก้ไข emergency"""
    
    print("🚨 Testing Emergency Fix")
    print("=" * 50)
    
    # Load fixed config
    config_path = 'configs/rtdetr/rtdetr_r50vd_face_landmark_emergency_fix.yml'
    
    try:
        cfg = YAMLConfig(config_path)
        print("✅ Config loaded successfully")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False
    
    # Test dataset
    print("\n📊 Testing dataset:")
    try:
        val_dataset = cfg.val_dataloader.dataset
        print(f"  Dataset size: {len(val_dataset)}")
        
        # Test first few samples
        for i in range(min(3, len(val_dataset))):
            img, target = val_dataset[i]
            
            print(f"\n  Sample {i}:")
            print(f"    Boxes range: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}]")
            
            if target['boxes'].max() > 1.0:
                print(f"    ❌ Boxes still not normalized!")
                return False
            
            if 'landmarks' in target:
                print(f"    Landmarks range: [{target['landmarks'].min():.3f}, {target['landmarks'].max():.3f}]")
                if target['landmarks'].max() > 1.0:
                    print(f"    ❌ Landmarks still not normalized!")
                    return False
            
        print("  ✅ All samples properly normalized")
        
    except Exception as e:
        print(f"  ❌ Dataset test failed: {e}")
        return False
    
    # Test model
    print("\n🤖 Testing model:")
    try:
        model = cfg.model
        print("  ✅ Model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"  ✅ Forward pass successful")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape}")
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        return False
    
    # Test loss computation
    print("\n💰 Testing loss computation:")
    try:
        criterion = cfg.criterion
        
        # Create dummy targets
        targets = [{
            'boxes': torch.tensor([[0.5, 0.5, 0.3, 0.3]]),  # cxcywh normalized
            'landmarks': torch.tensor([[0.4, 0.4, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.6, 0.6]]),
            'labels': torch.zeros(1, dtype=torch.int64),
            'image_id': torch.tensor([0]),
            'orig_size': torch.tensor([640, 640]),
            'size': torch.tensor([640, 640]),
            'area': torch.tensor([0.09]),
            'iscrowd': torch.zeros(1, dtype=torch.int64),
        }]
        
        with torch.no_grad():
            loss_dict = criterion(outputs, targets)
        
        print("  Loss breakdown:")
        total_loss = 0
        for name, value in loss_dict.items():
            if name in criterion.weight_dict:
                weight = criterion.weight_dict[name]
                weighted = value * weight
                total_loss += weighted
                print(f"    {name}: {value:.4f} × {weight} = {weighted:.4f}")
        
        print(f"  Total loss: {total_loss:.4f}")
        
        if total_loss < 10:
            print("  ✅ Loss is reasonable now!")
        elif total_loss < 50:
            print("  ⚠️  Loss is better but still high")
        else:
            print("  ❌ Loss is still too high")
            return False
            
    except Exception as e:
        print(f"  ❌ Loss test failed: {e}")
        return False
    
    return True


def run_quick_training_test():
    """ทดสอบการ training สั้นๆ"""
    
    print("\n🏃 Quick training test:")
    
    config_path = 'configs/rtdetr/rtdetr_r50vd_face_landmark_emergency_fix.yml'
    cfg = YAMLConfig(config_path)
    
    model = cfg.model
    criterion = cfg.criterion
    train_loader = cfg.train_dataloader
    
    # Simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00002)
    
    model.train()
    
    # Train for 5 steps
    for i, (images, targets) in enumerate(train_loader):
        if i >= 5:
            break
        
        print(f"  Step {i+1}/5")
        
        # Fix targets
        fixed_targets = []
        for target in targets:
            # Ensure normalization
            if 'boxes' in target and target['boxes'].max() > 1.0:
                if 'orig_size' in target:
                    w, h = target['orig_size'].tolist()
                    target['boxes'][:, [0, 2]] /= w
                    target['boxes'][:, [1, 3]] /= h
                    target['boxes'] = torch.clamp(target['boxes'], 0, 1)
            
            if 'landmarks' in target and target['landmarks'].max() > 1.0:
                if 'orig_size' in target:
                    w, h = target['orig_size'].tolist()
                    target['landmarks'][:, 0::2] /= w
                    target['landmarks'][:, 1::2] /= h
                    target['landmarks'] = torch.clamp(target['landmarks'], 0, 1)
            
            fixed_targets.append(target)
        
        try:
            # Forward
            outputs = model(images, fixed_targets)
            loss_dict = criterion(outputs, fixed_targets)
            
            # Total loss
            total_loss = sum(loss_dict[k] * criterion.weight_dict[k] 
                           for k in loss_dict.keys() if k in criterion.weight_dict)
            
            print(f"    Loss: {total_loss:.4f}")
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if total_loss > 100:
                print(f"    ⚠️  Loss still high: {total_loss:.4f}")
            else:
                print(f"    ✅ Loss reasonable: {total_loss:.4f}")
        
        except Exception as e:
            print(f"    ❌ Training step failed: {e}")
            return False
    
    print("  ✅ Quick training test passed!")
    return True


if __name__ == '__main__':
    print("เริ่มทดสอบ Emergency Fix...")
    
    # Test 1: Basic functionality
    if not test_emergency_fix():
        print("\n❌ Basic test failed!")
        sys.exit(1)
    
    # Test 2: Quick training
    if not run_quick_training_test():
        print("\n❌ Training test failed!")
        sys.exit(1)
    
    print("\n🎉 All tests passed!")
    print("\n📋 Next steps:")
    print("1. Copy the fixed transform classes to src/data/transforms.py")
    print("2. Create the emergency config files")
    print("3. Run training with:")
    print("   python tools/train_face_progressive.py --config configs/rtdetr/rtdetr_r50vd_face_landmark_emergency_fix.yml --debug")