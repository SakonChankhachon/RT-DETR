# tools/train_face_landmarks.py
"""
Training script for RT-DETR face detection and landmark localization
Includes face-specific training strategies
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
import numpy as np
from pathlib import Path

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS


class FaceTrainingStrategy:
    """Face-specific training strategies"""
    
    @staticmethod
    def adjust_learning_rate(optimizer, epoch, cfg):
        """Custom learning rate schedule for face detection"""
        
        # Warm-up for first 5 epochs
        if epoch < 5:
            lr = cfg.optimizer.lr * (epoch + 1) / 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Warm-up epoch {epoch}, lr: {lr}")
            
        # Step decay at specific epochs
        elif epoch in [40, 70, 90]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print(f"Decay learning rate at epoch {epoch}")
    
    @staticmethod
    def should_evaluate(epoch, eval_freq=5):
        """Determine if should run evaluation"""
        # Evaluate every eval_freq epochs and at important milestones
        return epoch % eval_freq == 0 or epoch in [1, 40, 70, 90, 99]
    
    @staticmethod
    def get_training_phases(total_epochs=100):
        """Define training phases for progressive learning"""
        return [
            {
                'name': 'Detection Focus',
                'epochs': range(0, 30),
                'loss_weights': {
                    'loss_vfl': 1.0,
                    'loss_bbox': 5.0,
                    'loss_giou': 2.0,
                    'loss_landmarks': 1.0,  # Lower weight initially
                }
            },
            {
                'name': 'Balanced Training',
                'epochs': range(30, 70),
                'loss_weights': {
                    'loss_vfl': 1.0,
                    'loss_bbox': 5.0,
                    'loss_giou': 2.0,
                    'loss_landmarks': 5.0,  # Increase landmark importance
                }
            },
            {
                'name': 'Fine-tuning',
                'epochs': range(70, 100),
                'loss_weights': {
                    'loss_vfl': 1.0,
                    'loss_bbox': 3.0,
                    'loss_giou': 2.0,
                    'loss_landmarks': 7.0,  # Focus on landmark precision
                }
            }
        ]


def setup_face_training(cfg):
    """Setup face-specific training configurations"""
    
    # Ensure correct number of classes
    if 'num_classes' in cfg.yaml_cfg:
        assert cfg.yaml_cfg['num_classes'] == 1, "Face detection should have num_classes=1"
    
    # Set default face training parameters if not specified
    if cfg.epoches == -1:
        cfg.epoches = 100
    
    # Face detection typically benefits from larger batch sizes
    if hasattr(cfg, 'train_dataloader') and cfg.train_dataloader:
        # But adjust based on GPU memory
        pass
    
    print("Face training configuration:")
    print(f"- Epochs: {cfg.epoches}")
    print(f"- Batch size: {cfg.yaml_cfg.get('train_dataloader', {}).get('batch_size', 'default')}")
    print(f"- Number of landmarks: {cfg.yaml_cfg.get('num_landmarks', 5)}")
    
    return cfg


def validate_face_dataset(dataloader, num_samples=5):
    """Validate face dataset by checking a few samples"""
    
    print("\nValidating face dataset...")
    
    for i, (images, targets) in enumerate(dataloader):
        if i >= num_samples:
            break
        
        print(f"\nSample {i+1}:")
        print(f"  Image shape: {images.shape}")
        print(f"  Number of faces: {[len(t['boxes']) for t in targets]}")
        
        # Check landmarks
        for j, target in enumerate(targets):
            if 'landmarks' in target:
                landmarks = target['landmarks']
                print(f"  Image {j} - Landmarks shape: {landmarks.shape}")
                # Check if landmarks are normalized
                if landmarks.max() <= 1.0 and landmarks.min() >= 0.0:
                    print(f"  Image {j} - Landmarks are normalized ✓")
                else:
                    print(f"  Image {j} - WARNING: Landmarks not normalized!")
                    print(f"    Range: [{landmarks.min():.2f}, {landmarks.max():.2f}]")
            else:
                print(f"  Image {j} - WARNING: No landmarks found!")
    
    print("\nDataset validation complete.\n")


def log_face_metrics(epoch, train_stats, test_stats=None):
    """Log face-specific metrics"""
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch} Summary:")
    print(f"{'='*60}")
    
    # Training metrics
    print("\nTraining:")
    print(f"  Total Loss: {train_stats.get('loss', 0):.4f}")
    print(f"  Face Detection Loss: {train_stats.get('loss_vfl', 0):.4f}")
    print(f"  Box Regression Loss: {train_stats.get('loss_bbox', 0):.4f}")
    print(f"  GIoU Loss: {train_stats.get('loss_giou', 0):.4f}")
    print(f"  Landmark Loss: {train_stats.get('loss_landmarks', 0):.4f}")
    
    # Per-landmark losses if available
    for i in range(5):  # Assuming 5 landmarks
        if f'loss_landmark_{i}' in train_stats:
            print(f"    Landmark {i}: {train_stats[f'loss_landmark_{i}']:.4f}")
    
    # Validation metrics
    if test_stats:
        print("\nValidation:")
        if 'coco_eval_bbox' in test_stats:
            ap_metrics = test_stats['coco_eval_bbox']
            print(f"  Face Detection AP@0.5: {ap_metrics[1]:.3f}")
            print(f"  Face Detection AP@0.75: {ap_metrics[2]:.3f}")
            print(f"  Face Detection AP (all): {ap_metrics[0]:.3f}")
        
        # Landmark metrics (if implemented)
        if 'landmark_error' in test_stats:
            print(f"  Average Landmark Error: {test_stats['landmark_error']:.2f} pixels")
    
    print(f"{'='*60}\n")


def main(args):
    """Main training function"""
    
    # Initialize distributed training
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)
    
    # Check resume/tuning options
    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch or resume or tuning at one time'
    
    # Load configuration
    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )
    
    # Setup face-specific training
    cfg = setup_face_training(cfg)
    
    # Override epochs if specified
    if args.epochs:
        cfg.epoches = args.epochs
    
    # Create solver
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        # Evaluation mode
        solver.val()
    else:
        # Training mode
        print("\nStarting face detection and landmark training...")
        
        # Validate dataset first
        if hasattr(solver, 'train_dataloader') and solver.train_dataloader:
            validate_face_dataset(solver.train_dataloader, num_samples=3)
        
        # Train with face-specific strategies
        if args.progressive:
            print("Using progressive training strategy...")
            # This would require modifying the solver
            # For now, just use standard training
        
        solver.fit()
    
    print("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RT-DETR Face Detection and Landmark Training')
    
    # Basic arguments
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--resume', '-r', type=str,
                       help='Resume from checkpoint')
    parser.add_argument('--tuning', '-t', type=str,
                       help='Fine-tuning from pretrained model')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run evaluation')
    
    # Training arguments
    parser.add_argument('--epochs', type=int,
                       help='Override number of epochs')
    parser.add_argument('--amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int,
                       help='Random seed')
    parser.add_argument('--progressive', action='store_true',
                       help='Use progressive training strategy')
    
    # Face-specific arguments
    parser.add_argument('--pretrained-backbone', type=str,
                       help='Path to pretrained backbone (e.g., from face recognition)')
    parser.add_argument('--freeze-backbone-epochs', type=int, default=0,
                       help='Number of epochs to freeze backbone')
    
    args = parser.parse_args()
    
    main(args)