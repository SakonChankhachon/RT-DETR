#!/usr/bin/env python
"""
Improve Validation Performance for Polar Landmark Training
Diagnoses issues and provides solutions for low validation scores
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def diagnose_validation_issues():
    """Diagnose common issues causing low validation performance"""
    
    print("🔍 Diagnosing Validation Performance Issues...")
    print("=" * 60)
    
    issues_found = []
    solutions = []
    
    # Check 1: Model outputs
    print("\n1️⃣ Checking Model Outputs...")
    try:
        from src.core import YAMLConfig
        
        # Try to load the current config
        config_files = [
            'configs/rtdetr/rtdetr_r50vd_face_landmark_optimized.yml',
            'configs/rtdetr/rtdetr_r50vd_face_landmark.yml',
            'configs/rtdetr/emergency_final/main.yml'
        ]
        
        cfg = None
        for config_file in config_files:
            if os.path.exists(config_file):
                cfg = YAMLConfig(config_file)
                print(f"✅ Loaded config: {config_file}")
                break
        
        if cfg is None:
            issues_found.append("No valid config file found")
            solutions.append("Create a proper configuration file")
            return issues_found, solutions
        
        model = cfg.model
        model.eval()
        
        # Test model with dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"   Model outputs: {list(outputs.keys())}")
        
        # Check landmark outputs
        if 'pred_landmarks' in outputs:
            landmarks = outputs['pred_landmarks']
            print(f"   ✅ Landmarks shape: {landmarks.shape}")
            print(f"   ✅ Landmarks range: [{landmarks.min():.3f}, {landmarks.max():.3f}]")
            
            # Check if landmarks are reasonable
            if landmarks.max() > 2.0 or landmarks.min() < -1.0:
                issues_found.append("Landmark predictions are out of reasonable range")
                solutions.append("Add sigmoid activation to landmark heads")
        else:
            issues_found.append("Model not outputting landmarks")
            solutions.append("Check if RTDETRTransformerPolarLandmark is being used")
        
    except Exception as e:
        issues_found.append(f"Model loading error: {e}")
        solutions.append("Fix model configuration or import issues")
    
    # Check 2: Dataset quality
    print("\n2️⃣ Checking Dataset Quality...")
    try:
        if cfg and hasattr(cfg, 'val_dataloader'):
            val_loader = cfg.val_dataloader
            
            sample_count = 0
            total_faces = 0
            landmark_issues = 0
            
            for images, targets in val_loader:
                sample_count += len(targets)
                
                for target in targets:
                    if 'boxes' in target:
                        total_faces += len(target['boxes'])
                    
                    if 'landmarks' in target:
                        landmarks = target['landmarks']
                        if landmarks.numel() > 0:
                            # Check landmark quality
                            if landmarks.max() > 1.1 or landmarks.min() < -0.1:
                                landmark_issues += 1
                
                if sample_count >= 10:  # Check first 10 samples
                    break
            
            print(f"   Samples checked: {sample_count}")
            print(f"   Total faces: {total_faces}")
            print(f"   Landmark issues: {landmark_issues}")
            
            if total_faces == 0:
                issues_found.append("No faces found in validation dataset")
                solutions.append("Check dataset annotations and paths")
            
            if landmark_issues > sample_count * 0.5:
                issues_found.append("Many landmarks are out of normalized range")
                solutions.append("Fix landmark normalization in dataset")
        
    except Exception as e:
        issues_found.append(f"Dataset checking error: {e}")
        solutions.append("Fix dataset configuration")
    
    # Check 3: Training configuration
    print("\n3️⃣ Checking Training Configuration...")
    try:
        if cfg and hasattr(cfg, 'criterion'):
            criterion = cfg.criterion
            
            if hasattr(criterion, 'weight_dict'):
                weights = criterion.weight_dict
                print(f"   Loss weights: {weights}")
                
                # Check for overly high landmark weights
                landmark_weight = weights.get('loss_landmarks', 0)
                detection_weight = weights.get('loss_vfl', 1)
                
                if landmark_weight > detection_weight * 10:
                    issues_found.append("Landmark loss weight too high compared to detection")
                    solutions.append("Reduce landmark loss weight to 1-5x detection weight")
                
                if landmark_weight == 0:
                    issues_found.append("Landmark loss weight is zero")
                    solutions.append("Set landmark loss weight to 1.0-5.0")
    
    except Exception as e:
        issues_found.append(f"Criterion checking error: {e}")
        solutions.append("Fix criterion configuration")
    
    return issues_found, solutions

def create_improved_config():
    """Create an improved configuration for better validation performance"""
    
    print("\n🔧 Creating Improved Configuration...")
    
    config_dir = Path("configs/rtdetr/polar_improved")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create improved main config
    improved_config = """__include__: [
  '../runtime.yml',
  './optimizer_improved.yml',
]

# Task and model setup
task: detection
model: RTDETR
criterion: PolarLandmarkCriterion
postprocessor: RTDETRFacePostProcessor

# Model architecture
RTDETR: 
  backbone: PResNet
  encoder: HybridEncoder
  decoder: RTDETRTransformerPolarLandmark
  multi_scale: [640]  # Fixed size for better stability

PResNet:
  depth: 50
  variant: d
  freeze_at: 0
  return_idx: [1, 2, 3]
  num_stages: 4
  freeze_norm: True
  pretrained: True 

HybridEncoder:
  in_channels: [512, 1024, 2048]
  feat_strides: [8, 16, 32]
  hidden_dim: 256
  use_encoder_idx: [2]
  num_encoder_layers: 1
  nhead: 8
  dim_feedforward: 1024
  dropout: 0.
  enc_act: 'gelu'
  pe_temperature: 10000
  expansion: 1.0
  depth_mult: 1
  act: 'silu'
  eval_spatial_size: [640, 640]

RTDETRTransformerPolarLandmark:
  in_channels: [512, 1024, 2048]
  feat_channels: [512, 1024, 2048]
  feat_strides: [8, 16, 32]
  hidden_dim: 256
  num_levels: 3
  num_queries: 300
  num_decoder_layers: 6
  num_denoising: 50  # Reduced for stability
  eval_idx: -1
  eval_spatial_size: [640, 640]
  num_classes: 1
  num_landmarks: 5
  num_orientations: 8
  heatmap_size: 32  # Reduced for better performance
  landmark_loss_weight: 1.0

# Global settings
use_focal_loss: True
num_classes: 1
num_landmarks: 5
num_orientations: 8
heatmap_size: 32

# Post-processing
RTDETRFacePostProcessor:
  num_top_queries: 300
  num_classes: 1
  num_landmarks: 5

# Loss function with balanced weights
PolarLandmarkCriterion:
  num_classes: 1
  num_landmarks: 5
  num_orientations: 8
  heatmap_size: 32
  heatmap_sigma: 2.0
  heatmap_loss_type: 'mse'  # More stable than focal
  
  # Balanced loss weights
  weight_dict: {
    loss_vfl: 1.0,        # Face detection
    loss_bbox: 5.0,       # Box regression
    loss_giou: 2.0,       # Box IoU
    loss_landmarks: 2.0,  # Coordinate loss
    loss_heatmap: 1.0,    # Heatmap loss
    loss_orientation: 0.5, # Orientation loss
    loss_radius: 0.5      # Radius loss
  }

  losses: ['vfl', 'boxes', 'landmarks']
  alpha: 0.75
  gamma: 2.0

  matcher:
    type: HungarianMatcher
    weight_dict: { cost_class: 2, cost_bbox: 5, cost_giou: 2 }
    alpha: 0.25
    gamma: 2.0

# Dataset configuration
train_dataloader: 
  type: DataLoader
  dataset: 
    type: FaceLandmarkDataset
    img_folder: ./dataset/faces/train/images/
    ann_file: ./dataset/faces/train/annotations.json
    num_landmarks: 5
    return_visibility: False
    transforms:
      type: Compose
      ops:
        - {type: RandomPhotometricDistort, p: 0.3}  # Reduced augmentation
        - {type: RandomHorizontalFlip, p: 0.5}
        - {type: Resize, size: [640, 640]}
        - {type: ToImageTensor}
        - {type: ConvertDtype}
        - {type: SanitizeBoundingBox, min_size: 1}
        - {type: ConvertBox, out_fmt: 'cxcywh', normalize: True}
        - {type: SanitizeLandmarks, min_val: 0.0, max_val: 1.0}
  shuffle: True
  batch_size: 4  # Smaller batch for stability
  num_workers: 2
  drop_last: False
  collate_fn: default_collate_fn

val_dataloader:
  type: DataLoader
  dataset: 
    type: FaceLandmarkDataset
    img_folder: ./dataset/faces/val/images/
    ann_file: ./dataset/faces/val/annotations.json
    num_landmarks: 5
    return_visibility: False
    transforms:
      type: Compose
      ops:
        - {type: Resize, size: [640, 640]}
        - {type: ToImageTensor}
        - {type: ConvertDtype}
        - {type: ConvertBox, out_fmt: 'cxcywh', normalize: True}
        - {type: SanitizeLandmarks, min_val: 0.0, max_val: 1.0}
  shuffle: False
  batch_size: 2
  num_workers: 2
  drop_last: False
  collate_fn: default_collate_fn

# Training settings
output_dir: ./output/rtdetr_polar_improved
epoches: 50  # Longer training
checkpoint_step: 5
log_step: 10

# Tuning from COCO pretrained for better initialization
tuning: https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
"""

    # Save improved config
    with open(config_dir / "main.yml", "w") as f:
        f.write(improved_config)
    
    # Create improved optimizer config
    optimizer_config = """use_ema: True 
ema:
  type: ModelEMA
  decay: 0.9999
  warmups: 1000  # Reduced warmup

find_unused_parameters: False  # Better for memory
clip_max_norm: 0.1

# Conservative optimizer settings
optimizer:
  type: AdamW
  lr: 0.00005  # Lower learning rate for stability
  betas: [0.9, 0.999]
  weight_decay: 0.0001

# Gradual learning rate decay
lr_scheduler:
  type: MultiStepLR
  milestones: [30, 45]
  gamma: 0.1

# Runtime settings
sync_bn: False
use_amp: False  # Disable for better precision during debugging
"""

    with open(config_dir / "optimizer_improved.yml", "w") as f:
        f.write(optimizer_config)
    
    print(f"✅ Created improved config: {config_dir}/main.yml")
    return config_dir / "main.yml"

def create_validation_script():
    """Create a script to validate training progress"""
    
    validation_script = '''#!/usr/bin/env python
"""
Validation and Debugging Script for Polar Landmarks
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.core import YAMLConfig

def validate_model_outputs(config_path, checkpoint_path=None):
    """Validate model outputs and visualize predictions"""
    
    print("🔍 Validating Model Performance...")
    
    # Load config and model
    cfg = YAMLConfig(config_path)
    model = cfg.model
    model.eval()
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    
    # Get validation data
    val_loader = cfg.val_dataloader
    
    print("\\n📊 Model Performance Analysis:")
    
    total_faces = 0
    detected_faces = 0
    landmark_errors = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if i >= 10:  # Test first 10 batches
                break
            
            # Forward pass
            outputs = model(images)
            
            # Post-process
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = cfg.postprocessor(outputs, orig_sizes)
            
            for j, (result, target) in enumerate(zip(results, targets)):
                total_faces += len(target['boxes'])
                
                # Count detections with score > 0.5
                scores = result['scores']
                detected_faces += (scores > 0.5).sum().item()
                
                # Calculate landmark errors if available
                if 'landmarks' in result and 'landmarks' in target:
                    pred_landmarks = result['landmarks'][scores > 0.5]
                    target_landmarks = target['landmarks']
                    
                    if len(pred_landmarks) > 0 and len(target_landmarks) > 0:
                        # Simple matching: use first prediction and ground truth
                        pred_lmk = pred_landmarks[0].reshape(-1, 2)
                        target_lmk = target_landmarks[0].reshape(-1, 2)
                        
                        # Convert to pixel coordinates
                        orig_w, orig_h = target['orig_size'].tolist()
                        target_lmk[:, 0] *= orig_w
                        target_lmk[:, 1] *= orig_h
                        
                        # Calculate error
                        error = torch.norm(pred_lmk - target_lmk, dim=1).mean()
                        landmark_errors.append(error.item())
    
    # Print results
    print(f"   Total ground truth faces: {total_faces}")
    print(f"   Detected faces (>0.5): {detected_faces}")
    print(f"   Detection recall: {detected_faces/max(total_faces,1):.3f}")
    
    if landmark_errors:
        mean_error = np.mean(landmark_errors)
        print(f"   Average landmark error: {mean_error:.2f} pixels")
        print(f"   Landmark error std: {np.std(landmark_errors):.2f} pixels")
    else:
        print("   No landmark predictions found")
    
    return {
        'detection_recall': detected_faces/max(total_faces,1),
        'landmark_error': np.mean(landmark_errors) if landmark_errors else float('inf')
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    
    args = parser.parse_args()
    validate_model_outputs(args.config, args.checkpoint)
'''
    
    with open("tools/validate_polar_landmarks.py", "w") as f:
        f.write(validation_script)
    
    os.chmod("tools/validate_polar_landmarks.py", 0o755)
    print("✅ Created validation script: tools/validate_polar_landmarks.py")

def main():
    """Main function to improve validation performance"""
    
    print("🚀 Polar Landmark Validation Improvement Tool")
    print("=" * 60)
    
    # Step 1: Diagnose issues
    issues, solutions = diagnose_validation_issues()
    
    if issues:
        print("\n⚠️  Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n💡 Suggested Solutions:")
        for i, solution in enumerate(solutions, 1):
            print(f"   {i}. {solution}")
    else:
        print("\n✅ No obvious issues found!")
    
    # Step 2: Create improved configuration
    improved_config_path = create_improved_config()
    
    # Step 3: Create validation tools
    create_validation_script()
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Test improved config:")
    print(f"   python tools/train.py -c {improved_config_path} --test-only")
    print(f"")
    print(f"2. Train with improved settings:")
    print(f"   python tools/train_optimized_face_landmarks.py --config {improved_config_path}")
    print(f"")
    print(f"3. Validate training progress:")
    print(f"   python tools/validate_polar_landmarks.py --config {improved_config_path} --checkpoint path/to/checkpoint.pth")
    print(f"")
    print(f"4. Monitor training closely:")
    print(f"   - Watch for loss convergence")
    print(f"   - Check landmark error decreases")
    print(f"   - Ensure detection recall improves")
    
    print(f"\n🔧 Key Improvements Made:")
    print(f"   ✅ Reduced heatmap size (32x32) for better performance")
    print(f"   ✅ Balanced loss weights")
    print(f"   ✅ Lower learning rate for stability")
    print(f"   ✅ Smaller batch size to avoid memory issues")
    print(f"   ✅ Better data normalization")
    print(f"   ✅ COCO pretrained initialization")

if __name__ == "__main__":
    main()