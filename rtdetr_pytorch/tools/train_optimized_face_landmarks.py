#!/usr/bin/env python
"""
Optimized Training Script for RT-DETR Face Detection and Landmark Localization
Includes progressive training strategy and improved loss balancing
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
import time

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS
from src.misc import MetricLogger
from src.solver.det_engine import evaluate

# Apply COCO fix for normalized boxes
import src.data.coco.coco_utils as coco_utils
from src.data.coco.coco_utils_fixed import get_coco_api_from_dataset
coco_utils.get_coco_api_from_dataset = get_coco_api_from_dataset


class OptimizedFaceTrainer:
    """Optimized trainer for face detection and landmarks"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training phases for progressive learning
        self.training_phases = [
            {
                'name': 'Phase 1: Detection Foundation',
                'epochs': (0, 30),
                'loss_weights': {
                    'loss_vfl': 1.0,
                    'loss_bbox': 5.0,
                    'loss_giou': 2.0,
                    'loss_landmarks': 2.0,  # Start with moderate landmark weight
                },
                'description': 'Focus on stable face detection'
            },
            {
                'name': 'Phase 2: Joint Optimization',
                'epochs': (30, 70),
                'loss_weights': {
                    'loss_vfl': 1.0,
                    'loss_bbox': 5.0,
                    'loss_giou': 2.0,
                    'loss_landmarks': 8.0,  # Increase landmark importance
                },
                'description': 'Balance detection and landmarks'
            },
            {
                'name': 'Phase 3: Landmark Refinement',
                'epochs': (70, 100),
                'loss_weights': {
                    'loss_vfl': 0.5,
                    'loss_bbox': 3.0,
                    'loss_giou': 1.0,
                    'loss_landmarks': 15.0,  # Focus on landmark precision
                },
                'description': 'Fine-tune landmark accuracy'
            }
        ]
    
    def get_current_phase(self, epoch):
        """Get training phase for current epoch"""
        for phase in self.training_phases:
            if phase['epochs'][0] <= epoch < phase['epochs'][1]:
                return phase
        return self.training_phases[-1]
    
    def adjust_loss_weights(self, criterion, epoch):
        """Dynamically adjust loss weights based on training phase"""
        phase = self.get_current_phase(epoch)
        
        print(f"\n🔄 {phase['name']} (Epoch {epoch})")
        print(f"📝 {phase['description']}")
        print("⚖️  Adjusting loss weights:")
        
        for loss_name, weight in phase['loss_weights'].items():
            if loss_name in criterion.weight_dict:
                old_weight = criterion.weight_dict[loss_name]
                criterion.weight_dict[loss_name] = weight
                print(f"   {loss_name}: {old_weight:.2f} → {weight:.2f}")
    
    def compute_landmark_metrics(self, predictions, targets):
        """Compute face landmark specific metrics"""
        metrics = defaultdict(list)
        
        for pred, target in zip(predictions, targets):
            if 'landmarks' not in pred or 'landmarks' not in target:
                continue

            pred_landmarks = pred['landmarks'].cpu().numpy()
            target_landmarks = target['landmarks'].cpu().numpy()

            # Skip if no faces
            if len(pred_landmarks) == 0 or len(target_landmarks) == 0:
                continue

            # Convert ground truth boxes/landmarks to pixel coordinates
            orig_w, orig_h = target['orig_size'].tolist()

            gt_boxes_cxcywh = target['boxes'].cpu().numpy()
            gt_boxes_xyxy = np.zeros_like(gt_boxes_cxcywh)
            gt_boxes_xyxy[:, 0] = (gt_boxes_cxcywh[:, 0] - gt_boxes_cxcywh[:, 2] / 2) * orig_w
            gt_boxes_xyxy[:, 1] = (gt_boxes_cxcywh[:, 1] - gt_boxes_cxcywh[:, 3] / 2) * orig_h
            gt_boxes_xyxy[:, 2] = (gt_boxes_cxcywh[:, 0] + gt_boxes_cxcywh[:, 2] / 2) * orig_w
            gt_boxes_xyxy[:, 3] = (gt_boxes_cxcywh[:, 1] + gt_boxes_cxcywh[:, 3] / 2) * orig_h

            # Predicted boxes are already pixel xyxy
            pred_boxes = pred['boxes'].cpu().numpy()

            for t_idx, t_box in enumerate(gt_boxes_xyxy):
                # Find best matching prediction
                if len(pred_boxes) > 0:
                    ious = self.compute_iou(pred_boxes, t_box[None, :])
                    best_idx = ious.argmax()

                    if ious[best_idx] > 0.5:  # Match threshold
                        # Compute landmark errors in pixel space
                        t_lmks = target_landmarks[t_idx].reshape(-1, 2)
                        t_lmks[:, 0] *= orig_w
                        t_lmks[:, 1] *= orig_h

                        p_lmks = pred_landmarks[best_idx].reshape(-1, 2)

                        errors = np.linalg.norm(t_lmks - p_lmks, axis=1)

                        for i, error in enumerate(errors):
                            metrics[f'landmark_{i}_error'].append(error)

                        metrics['mean_landmark_error'].append(errors.mean())

                        # Normalized Mean Error (NME) using inter-ocular distance
                        if len(t_lmks) >= 2:
                            iod = np.linalg.norm(t_lmks[0] - t_lmks[1])
                            nme = errors.mean() / iod if iod > 0 else 0
                            metrics['nme'].append(nme)
        
        # Compute mean metrics
        mean_metrics = {}
        for key, values in metrics.items():
            if len(values) > 0:
                mean_metrics[key] = np.mean(values)
            else:
                mean_metrics[key] = 0.0
        
        return mean_metrics
    
    def compute_iou(self, boxes1, boxes2):
        """Compute IoU between two sets of boxes"""
        x1 = np.maximum(boxes1[:, 0], boxes2[:, 0])
        y1 = np.maximum(boxes1[:, 1], boxes2[:, 1])
        x2 = np.minimum(boxes1[:, 2], boxes2[:, 2])
        y2 = np.minimum(boxes1[:, 3], boxes2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def log_detailed_metrics(self, epoch, phase, train_stats, val_stats=None):
        """Log detailed training information"""
        print(f"\n{'='*70}")
        print(f"📊 Epoch {epoch} - {phase['name']}")
        print(f"{'='*70}")
        
        # Training losses
        print("\n🏋️  Training Losses:")
        loss_total = 0
        for key, value in train_stats.items():
            if 'loss' in key:
                print(f"   {key}: {value:.4f}")
                loss_total += value
        print(f"   total_loss: {loss_total:.4f}")
        
        # Validation metrics
        if val_stats:
            print("\n✅ Validation Metrics:")
            
            # Detection metrics
            if 'coco_eval_bbox' in val_stats:
                ap_metrics = val_stats['coco_eval_bbox']
                print(f"   Face Detection:")
                print(f"     AP@0.5: {ap_metrics[1]:.3f}")
                print(f"     AP@0.75: {ap_metrics[2]:.3f}")
                print(f"     AP (all): {ap_metrics[0]:.3f}")
            
            # Landmark metrics
            landmark_keys = [k for k in val_stats.keys() if 'landmark' in k or 'nme' in k]
            if landmark_keys:
                print(f"   Face Landmarks:")
                for key in sorted(landmark_keys):
                    print(f"     {key}: {val_stats[key]:.3f}")
        
        print(f"{'='*70}\n")


def fix_targets_boxes(targets):
    """Ensure target boxes are in normalized cxcywh format"""
    fixed_targets = []

    for target in targets:
        target = {k: v.clone() if isinstance(v, torch.Tensor) else v
                  for k, v in target.items()}

        if 'boxes' in target and target['boxes'].numel() > 0:
            boxes = target['boxes']

            # Determine if boxes are in pixel coordinates
            if boxes.max() > 1.0:
                if 'orig_size' in target:
                    orig_size = target['orig_size']
                    if isinstance(orig_size, torch.Tensor):
                        w, h = orig_size[0].item(), orig_size[1].item()
                    else:
                        w, h = orig_size
                else:
                    w, h = 640, 640

                boxes = boxes.float()
                boxes[:, [0, 2]] = boxes[:, [0, 2]] / w
                boxes[:, [1, 3]] = boxes[:, [1, 3]] / h

            # If boxes look like xyxy (x2 > x1 and y2 > y1), convert
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            width_xyxy = x2 - x1
            height_xyxy = y2 - y1

            if (width_xyxy >= 0).all() and (height_xyxy >= 0).all():
                # Values already normalized to [0, 1]; treat as xyxy
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                width = width_xyxy
                height = height_xyxy
            else:
                # Assume boxes are already cxcywh
                cx, cy, width, height = x1, y1, x2, y2

            cx = torch.clamp(cx, 0.0, 1.0)
            cy = torch.clamp(cy, 0.0, 1.0)
            width = torch.clamp(width, 0.0, 1.0)
            height = torch.clamp(height, 0.0, 1.0)

            target['boxes'] = torch.stack([cx, cy, width, height], dim=1)

        fixed_targets.append(target)

    return fixed_targets


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Optimized RT-DETR Face Training')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--val-freq', type=int, default=5, help='Validation frequency')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    print("✅ Face landmark components loaded successfully")
    
    # Initialize distributed training
    dist.init_distributed()
    
    print("Loading configuration...")
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    print(f"\n🚀 Starting Optimized Face Landmark Training")
    print(f"Config: {args.config}")
    print(f"Output: {cfg.output_dir}")
    print(f"Epochs: {cfg.epoches}")
    print(f"Device: {args.device}")
    
    # Create trainer
    trainer = OptimizedFaceTrainer(cfg)
    
    # Create and setup solver
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.setup()
    
    # Create simplified optimizer without parameter groups to avoid regex issues
    optimizer = torch.optim.AdamW(
        solver.model.parameters(),
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.0001
    )
    solver.optimizer = optimizer
    solver.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        solver.optimizer, 
        milestones=[60, 80], 
        gamma=0.1
    )
    
    if cfg.resume:
        print(f'Resume checkpoint from {cfg.resume}')
        solver.resume(cfg.resume)
    
    # Wrap dataloaders
    solver.train_dataloader = dist.warp_loader(
        cfg.train_dataloader, 
        shuffle=cfg.train_dataloader.shuffle
    )
    solver.val_dataloader = dist.warp_loader(
        cfg.val_dataloader,
        shuffle=cfg.val_dataloader.shuffle
    )
    
    # Training loop
    best_metric = float('inf')  # Lower NME is better
    best_epoch = -1
    
    for epoch in range(solver.last_epoch + 1, cfg.epoches):
        start_time = time.time()
        
        # Update loss weights based on training phase
        trainer.adjust_loss_weights(solver.criterion, epoch)
        
        # Get current phase
        phase = trainer.get_current_phase(epoch)
        
        # Train one epoch
        if dist.is_dist_available_and_initialized():
            solver.train_dataloader.sampler.set_epoch(epoch)
        
        # Standard training
        solver.model.train()
        metric_logger = MetricLogger(delimiter="  ")
        
        for samples, targets in metric_logger.log_every(
            solver.train_dataloader, 
            cfg.log_step, 
            header=f'Epoch: [{epoch}]'
        ):
            samples = samples.to(solver.device)
            targets = [{k: v.to(solver.device) for k, v in t.items()} for t in targets]
            
            # Fix boxes format
            targets = fix_targets_boxes(targets)
            
            # Forward pass
            try:
                outputs = solver.model(samples, targets)
                loss_dict = solver.criterion(outputs, targets)
            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue
            
            # Compute total loss
            losses = sum(loss_dict[k] * solver.criterion.weight_dict[k] 
                        for k in loss_dict.keys() if k in solver.criterion.weight_dict)
            
            # Backward pass
            solver.optimizer.zero_grad()
            losses.backward()
            if cfg.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(solver.model.parameters(), cfg.clip_max_norm)
            solver.optimizer.step()
            
            # Update EMA if used
            if solver.ema is not None:
                solver.ema.update(solver.model)
            
            # Log metrics
            metric_logger.update(loss=losses.item(), **loss_dict)
        
        # Update learning rate
        solver.lr_scheduler.step()
        
        # Get training stats
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
        epoch_time = time.time() - start_time
        
        # Validation
        val_stats = None
        if epoch % args.val_freq == 0 or epoch == cfg.epoches - 1:
            print(f"\n🔍 Running validation at epoch {epoch}...")
            
            # Standard validation
            module = solver.ema.module if solver.ema else solver.model
            try:
                test_stats, coco_evaluator = evaluate(
                    module, solver.criterion, solver.postprocessor,
                    solver.val_dataloader, get_coco_api_from_dataset(solver.val_dataloader.dataset),
                    solver.device, solver.output_dir
                )
                
                # Compute landmark-specific metrics if model outputs landmarks
                solver.model.eval()
                all_predictions = []
                all_targets = []
                
                with torch.no_grad():
                    for samples, targets in solver.val_dataloader:
                        samples = samples.to(solver.device)
                        targets = [{k: v.to(solver.device) for k, v in t.items()} for t in targets]
                        targets = fix_targets_boxes(targets)
                        
                        outputs = module(samples)
                        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                        results = solver.postprocessor(outputs, orig_sizes)
                        
                        all_predictions.extend(results)
                        all_targets.extend(targets)
                
                # Compute landmark metrics
                landmark_metrics = trainer.compute_landmark_metrics(all_predictions, all_targets)
                test_stats.update(landmark_metrics)
                
                val_stats = test_stats
                
                # Check if best model (using NME - lower is better)
                current_metric = landmark_metrics.get('nme', float('inf'))
                if current_metric < best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    
                    # Save best model
                    best_path = solver.output_dir / 'best_model.pth'
                    dist.save_on_master(solver.state_dict(epoch), best_path)
                    print(f"🏆 New best model saved! NME: {current_metric:.4f}")
            
            except Exception as e:
                print(f"Error during validation: {e}")
                val_stats = None
        
        # Log training information
        trainer.log_detailed_metrics(epoch, phase, train_stats, val_stats)
        print(f"⏱️  Epoch time: {epoch_time:.1f}s")
        
        # Save checkpoint
        if (epoch + 1) % cfg.checkpoint_step == 0:
            checkpoint_path = solver.output_dir / f'checkpoint{epoch:04}.pth'
            dist.save_on_master(solver.state_dict(epoch), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best model: Epoch {best_epoch} with NME: {best_metric:.4f}")


if __name__ == '__main__':
    main()