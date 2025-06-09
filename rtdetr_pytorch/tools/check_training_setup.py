# tools/check_training_setup.py
"""
Check training setup before starting new training
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import matplotlib.pyplot as plt
from src.core import YAMLConfig
import numpy as np


def check_dataset(cfg):
    """Check dataset configuration and samples"""
    print("=" * 60)
    print("CHECKING DATASET")
    print("=" * 60)
    
    # Check train dataset
    train_loader = cfg.train_dataloader
    val_loader = cfg.val_dataloader
    
    print(f"\nTrain dataset:")
    print(f"  Size: {len(train_loader.dataset)} images")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Num workers: {train_loader.num_workers}")
    
    print(f"\nValidation dataset:")
    print(f"  Size: {len(val_loader.dataset)} images")
    print(f"  Batch size: {val_loader.batch_size}")
    
    # Check a few samples
    print("\nChecking train samples...")
    total_boxes = 0
    total_landmarks = 0
    boxes_per_image = []
    
    for i, (images, targets) in enumerate(train_loader):
        if i >= 5:  # Check first 5 batches
            break
            
        batch_boxes = sum(len(t['boxes']) for t in targets)
        batch_landmarks = sum(len(t['landmarks']) if 'landmarks' in t else 0 for t in targets)
        
        total_boxes += batch_boxes
        total_landmarks += batch_landmarks
        boxes_per_image.extend([len(t['boxes']) for t in targets])
        
        print(f"  Batch {i}: {len(images)} images, {batch_boxes} boxes, {batch_landmarks} landmark sets")
        
        # Check first target in detail
        if i == 0 and len(targets) > 0:
            t = targets[0]
            print(f"\n  First target detail:")
            print(f"    Keys: {t.keys()}")
            print(f"    Boxes shape: {t['boxes'].shape}")
            print(f"    Labels: {t['labels'].unique()}")
            if 'landmarks' in t:
                print(f"    Landmarks shape: {t['landmarks'].shape}")
    
    if len(boxes_per_image) > 0:
        print(f"\n  Statistics:")
        print(f"    Average boxes per image: {np.mean(boxes_per_image):.2f}")
        print(f"    Min boxes per image: {min(boxes_per_image)}")
        print(f"    Max boxes per image: {max(boxes_per_image)}")
        print(f"    Images with no boxes: {sum(1 for b in boxes_per_image if b == 0)}")
    
    return len(boxes_per_image) > 0 and sum(boxes_per_image) > 0


def check_model(cfg):
    """Check model configuration"""
    print("\n" + "=" * 60)
    print("CHECKING MODEL")
    print("=" * 60)
    
    model = cfg.model
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    # Check key components
    print(f"\nModel components:")
    print(f"  Backbone: {type(model.backbone).__name__}")
    print(f"  Encoder: {type(model.encoder).__name__}")
    print(f"  Decoder: {type(model.decoder).__name__}")
    
    # Check decoder configuration
    if hasattr(model.decoder, 'num_classes'):
        print(f"  Number of classes: {model.decoder.num_classes}")
    if hasattr(model.decoder, 'num_landmarks'):
        print(f"  Number of landmarks: {model.decoder.num_landmarks}")
    if hasattr(model.decoder, 'num_queries'):
        print(f"  Number of queries: {model.decoder.num_queries}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    
    try:
        with torch.no_grad():
            outputs = model(dummy_input)
        print("  ✅ Forward pass successful")
        print(f"  Output keys: {outputs.keys()}")
        if 'pred_landmarks' in outputs:
            print(f"  ✅ Model outputs landmarks")
        else:
            print(f"  ❌ Model does NOT output landmarks")
        return True
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        return False


def check_training_config(cfg):
    """Check training configuration"""
    print("\n" + "=" * 60)
    print("CHECKING TRAINING CONFIG")
    print("=" * 60)
    
    print(f"\nTraining settings:")
    print(f"  Epochs: {cfg.epoches}")
    print(f"  Learning rate: {cfg.yaml_cfg.get('optimizer', {}).get('lr', 'Not set')}")
    print(f"  Batch size: {cfg.yaml_cfg.get('train_dataloader', {}).get('batch_size', 'Not set')}")
    print(f"  Use EMA: {cfg.use_ema}")
    print(f"  Use AMP: {cfg.use_amp}")
    print(f"  Gradient clipping: {cfg.clip_max_norm}")
    
    # Check optimizer
    print(f"\nOptimizer config:")
    opt_cfg = cfg.yaml_cfg.get('optimizer', {})
    print(f"  Type: {opt_cfg.get('type', 'Not set')}")
    print(f"  Base LR: {opt_cfg.get('lr', 'Not set')}")
    if 'params' in opt_cfg:
        print(f"  Parameter groups: {len(opt_cfg['params'])}")
    
    # Check loss weights
    print(f"\nLoss weights:")
    criterion_cfg = cfg.yaml_cfg.get('PolarLandmarkCriterion', {})
    if 'weight_dict' in criterion_cfg:
        for k, v in criterion_cfg['weight_dict'].items():
            print(f"  {k}: {v}")
    
    return True


def visualize_samples(cfg, num_samples=4):
    """Visualize some training samples"""
    print("\n" + "=" * 60)
    print("VISUALIZING SAMPLES")
    print("=" * 60)
    
    train_loader = cfg.train_dataloader
    
    # Get one batch
    for images, targets in train_loader:
        break
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = (img * 255).clip(0, 255).astype(np.uint8)
        
        axes[i].imshow(img)
        
        # Draw boxes
        if i < len(targets) and 'boxes' in targets[i]:
            boxes = targets[i]['boxes'].cpu().numpy()
            H, W = img.shape[:2]
            
            for box in boxes:
                # Convert normalized cxcywh to pixel xyxy
                cx, cy, w, h = box
                x1 = (cx - w/2) * W
                y1 = (cy - h/2) * H
                w_pix = w * W
                h_pix = h * H
                
                from matplotlib.patches import Rectangle
                rect = Rectangle((x1, y1), w_pix, h_pix,
                               linewidth=2, edgecolor='r', facecolor='none')
                axes[i].add_patch(rect)
            
            axes[i].set_title(f'Image {i}: {len(boxes)} faces')
        else:
            axes[i].set_title(f'Image {i}: No annotations')
        
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_samples_check.png')
    print(f"\nSaved visualization to training_samples_check.png")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                       default='configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize training samples')
    
    args = parser.parse_args()
    
    print(f"Checking training setup for: {args.config}\n")
    
    # Load config
    cfg = YAMLConfig(args.config)
    
    # Run checks
    dataset_ok = check_dataset(cfg)
    model_ok = check_model(cfg)
    config_ok = check_training_config(cfg)
    
    if args.visualize:
        visualize_samples(cfg)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    checks = {
        "Dataset": dataset_ok,
        "Model": model_ok,
        "Config": config_ok
    }
    
    all_ok = all(checks.values())
    
    for name, status in checks.items():
        status_str = "✅ OK" if status else "❌ FAILED"
        print(f"  {name}: {status_str}")
    
    if all_ok:
        print("\n✅ All checks passed! Ready to train.")
        print("\nTo start training, run:")
        print(f"  python tools/train_polar_landmarks.py -c {args.config}")
    else:
        print("\n❌ Some checks failed. Please fix the issues before training.")
    
    return all_ok


if __name__ == '__main__':
    main()