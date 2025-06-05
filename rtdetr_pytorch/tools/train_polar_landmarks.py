#!/usr/bin/env python
"""
Training script for RT-DETR with Polar Heatmap Face Landmarks
Implements progressive training strategy and landmark-specific optimizations
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

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS
from src.misc import MetricLogger
# Apply COCO fix for normalized boxes
import src.data.coco.coco_utils as coco_utils
from src.data.coco.coco_utils_fixed import get_coco_api_from_dataset
coco_utils.get_coco_api_from_dataset = get_coco_api_from_dataset

class PolarLandmarkTrainer:
    """Custom trainer for polar heatmap-based face landmarks"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training phases for progressive learning
        self.training_phases = [
            {
                'name': 'Phase 1: Detection Focus',
                'epochs': (0, 30),
                'lr_scale': 1.0,
                'loss_weights': {
                    'loss_vfl': 1.0,
                    'loss_bbox': 5.0,
                    'loss_giou': 2.0,
                    'loss_landmarks': 1.0,
                    'loss_heatmap': 0.5,
                    'loss_orientation': 0.1,
                    'loss_radius': 0.1,
                },
                'freeze_landmark_heads': False,
            },
            {
                'name': 'Phase 2: Joint Learning',
                'epochs': (30, 60),
                'lr_scale': 1.0,
                'loss_weights': {
                    'loss_vfl': 1.0,
                    'loss_bbox': 5.0,
                    'loss_giou': 2.0,
                    'loss_landmarks': 5.0,
                    'loss_heatmap': 2.0,
                    'loss_orientation': 1.0,
                    'loss_radius': 1.0,
                },
                'freeze_landmark_heads': False,
            },
            {
                'name': 'Phase 3: Landmark Refinement',
                'epochs': (60, 100),
                'lr_scale': 0.1,
                'loss_weights': {
                    'loss_vfl': 0.5,
                    'loss_bbox': 3.0,
                    'loss_giou': 1.0,
                    'loss_landmarks': 10.0,
                    'loss_heatmap': 5.0,
                    'loss_orientation': 2.0,
                    'loss_radius': 2.0,
                },
                'freeze_landmark_heads': False,
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
        
        print(f"\n{phase['name']} (Epoch {epoch})")
        print("Adjusting loss weights:")
        
        for loss_name, weight in phase['loss_weights'].items():
            if loss_name in criterion.weight_dict:
                old_weight = criterion.weight_dict[loss_name]
                criterion.weight_dict[loss_name] = weight
                print(f"  {loss_name}: {old_weight:.2f} -> {weight:.2f}")
    
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
            
            # Simple matching based on IoU
            pred_boxes = pred['boxes'].cpu().numpy()
            target_boxes = target['boxes'].cpu().numpy()
            
            for t_idx, t_box in enumerate(target_boxes):
                # Find best matching prediction
                if len(pred_boxes) > 0:
                    ious = self.compute_iou(pred_boxes, t_box[None, :])
                    best_idx = ious.argmax()
                    
                    if ious[best_idx] > 0.5:  # Match threshold
                        # Compute landmark errors
                        t_lmks = target_landmarks[t_idx].reshape(-1, 2)
                        p_lmks = pred_landmarks[best_idx].reshape(-1, 2)
                        
                        # Per-landmark error (in pixels, assuming normalized coords)
                        errors = np.linalg.norm(t_lmks - p_lmks, axis=1) * 640  # Scale to image size
                        
                        for i, error in enumerate(errors):
                            metrics[f'landmark_{i}_error'].append(error)
                        
                        metrics['mean_landmark_error'].append(errors.mean())
                        
                        # Normalized Mean Error (NME)
                        # Using inter-ocular distance as normalization
                        if len(t_lmks) >= 2:
                            iod = np.linalg.norm(t_lmks[0] - t_lmks[1])  # Distance between eyes
                            nme = errors.mean() / (iod * 640) if iod > 0 else 0
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
    
    def log_training_info(self, epoch, phase, train_stats, val_stats=None):
        """Log detailed training information"""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} - {phase['name']}")
        print(f"{'='*60}")
        
        # Training losses
        print("\nTraining Losses:")
        for key, value in train_stats.items():
            if 'loss' in key:
                print(f"  {key}: {value:.4f}")
        
        # Validation metrics
        if val_stats:
            print("\nValidation Metrics:")
            
            # Detection metrics
            if 'coco_eval_bbox' in val_stats:
                ap_metrics = val_stats['coco_eval_bbox']
                print(f"  Face Detection:")
                print(f"    AP@0.5: {ap_metrics[1]:.3f}")
                print(f"    AP@0.75: {ap_metrics[2]:.3f}")
                print(f"    AP (all): {ap_metrics[0]:.3f}")
            
            # Landmark metrics
            landmark_keys = [k for k in val_stats.keys() if 'landmark' in k or 'nme' in k]
            if landmark_keys:
                print(f"  Face Landmarks:")
                for key in sorted(landmark_keys):
                    print(f"    {key}: {val_stats[key]:.3f}")
        
        print(f"{'='*60}\n")
    
    def visualize_predictions(self, samples, outputs, targets, results, save_dir):
        """
        Draw boxes and landmarks on images and save to a directory.
        
        Args:
            samples: Batch of images tensor [batch_size, 3, H, W]
            outputs: Model outputs dict with 'pred_boxes', 'pred_landmarks'
            targets: Ground truth targets
            results: Post-processed results
            save_dir: Directory to save visualizations
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get batch size
        batch_size = samples.shape[0]
        
        for i in range(min(batch_size, 4)):  # Visualize only first 4 images
            # Get image tensor and convert to numpy
            img = samples[i].detach().cpu().permute(1, 2, 0).numpy()
            
            # Denormalize if needed (assuming standard ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = (img * 255).clip(0, 255).astype(np.uint8)
            
            # Get image dimensions
            H_img, W_img = img.shape[:2]
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot original image with ground truth
            ax1.imshow(img)
            ax1.set_title('Ground Truth')
            ax1.axis('off')
            
            # Draw ground truth boxes and landmarks
            if i < len(targets):
                target = targets[i]
                if 'boxes' in target and 'landmarks' in target:
                    gt_boxes = target['boxes'].cpu().numpy()
                    gt_landmarks = target['landmarks'].cpu().numpy()
                    
                    for j, box in enumerate(gt_boxes):
                        # Box is in cxcywh format, convert to xyxy
                        cx, cy, w, h = box
                        x1 = (cx - w/2) * W_img
                        y1 = (cy - h/2) * H_img
                        w_pixel = w * W_img
                        h_pixel = h * H_img
                        
                        rect = patches.Rectangle(
                            (x1, y1), w_pixel, h_pixel,
                            linewidth=2, edgecolor='g', facecolor='none'
                        )
                        ax1.add_patch(rect)
                        
                        # Draw landmarks if available
                        if j < len(gt_landmarks):
                            lmks = gt_landmarks[j].reshape(-1, 2)
                            # Convert normalized to pixel coordinates
                            lmks[:, 0] *= W_img
                            lmks[:, 1] *= H_img
                            ax1.scatter(lmks[:, 0], lmks[:, 1], c='g', s=30, marker='o')
            
            # Plot predictions
            ax2.imshow(img)
            ax2.set_title('Predictions')
            ax2.axis('off')
            
            # Draw predicted boxes and landmarks
            if i < len(results):
                result = results[i]
                pred_boxes = result['boxes'].cpu().numpy()
                pred_scores = result['scores'].cpu().numpy()
                pred_landmarks = result.get('landmarks', None)
                
                # Filter by score threshold
                score_threshold = 0.3
                keep = pred_scores > score_threshold
                
                if keep.any():
                    pred_boxes = pred_boxes[keep]
                    pred_scores = pred_scores[keep]
                    if pred_landmarks is not None:
                        pred_landmarks = pred_landmarks[keep].cpu().numpy()
                    
                    for j, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
                        # Box should be in xyxy format from postprocessor
                        x1, y1, x2, y2 = box
                        
                        rect = patches.Rectangle(
                            (x1, y1), x2-x1, y2-y1,
                            linewidth=2, edgecolor='r', facecolor='none'
                        )
                        ax2.add_patch(rect)
                        
                        # Add score text
                        ax2.text(x1, y1-5, f'{score:.2f}', color='r', fontsize=10)
                        
                        # Draw landmarks if available
                        if pred_landmarks is not None and j < len(pred_landmarks):
                            lmks = pred_landmarks[j].reshape(-1, 2)
                            ax2.scatter(lmks[:, 0], lmks[:, 1], c='r', s=30, marker='o')
                            
                            # Draw landmark indices
                            for k, (x, y) in enumerate(lmks):
                                ax2.text(x+2, y+2, str(k), color='r', fontsize=8)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'visualization_{i:04d}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Saved {min(batch_size, 4)} visualizations to {save_dir}")

# เพิ่ม function นี้ก่อน main() function
def fix_targets_boxes(targets):
    """Ensure target boxes are in normalized cxcywh format.

    The FaceLandmarkDataset already returns boxes as normalized
    center\-x, center\-y, width and height. However, older annotation
    formats may use pixel\-space ``xyxy`` coordinates. This helper now
    checks the range of each box and only converts when needed.
    """

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

def main(args):
    """Main training function"""
    
    # Initialize distributed training
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)
    
    # Load configuration
    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )
    
    # Override epochs if specified
    if args.epochs:
        cfg.epoches = args.epochs
    
    # Create custom trainer
    trainer = PolarLandmarkTrainer(cfg)
    
    # Create solver with modified training loop
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        print("Running evaluation only...")
        solver.val()
        return
    
    # Custom training loop with progressive learning
    print("Starting Polar Heatmap Face Landmark Training...")
    print(f"Total epochs: {cfg.epoches}")
    print(f"Number of training phases: {len(trainer.training_phases)}")
    

    # tools/train_polar_landmarks.py ประมาณบรรทัด 280 ก่อน solver.setup()
    # debug
    print("DEBUG yaml_cfg type =", type(cfg.yaml_cfg))
    print("DEBUG yaml_cfg value =", cfg.yaml_cfg)   # พิมพ์ออกทั้งก้อนก่อน




    # Setup solver for training
    solver.setup()
    solver.optimizer = cfg.optimizer
    solver.lr_scheduler = cfg.lr_scheduler
    
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
    best_metric = 0.0
    best_epoch = -1
    
    for epoch in range(solver.last_epoch + 1, cfg.epoches):
        # Update loss weights based on training phase
        trainer.adjust_loss_weights(solver.criterion, epoch)
        
        # Get current phase
        phase = trainer.get_current_phase(epoch)
        
        # Adjust learning rate if needed
        if hasattr(phase, 'lr_scale'):
            for param_group in solver.optimizer.param_groups:
                param_group['lr'] *= phase['lr_scale']
        
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
            #targets = fix_targets_boxes(targets)
            
            # Forward pass
            outputs = solver.model(samples, targets)
            loss_dict = solver.criterion(outputs, targets)
            
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
        
        # Validation
        val_stats = None
        if epoch % args.val_freq == 0 or epoch == cfg.epoches - 1:
            print(f"\nRunning validation at epoch {epoch}...")
            
            # Standard validation
            module = solver.ema.module if solver.ema else solver.model
            test_stats, coco_evaluator = evaluate(
                module, solver.criterion, solver.postprocessor,
                solver.val_dataloader, get_coco_api_from_dataset(solver.val_dataloader.dataset),
                solver.device, solver.output_dir
            )
            
            # Compute landmark-specific metrics
            solver.model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for samples, targets in solver.val_dataloader:
                    samples = samples.to(solver.device)
                    targets = [{k: v.to(solver.device) for k, v in t.items()} for t in targets]
                     # Fix boxes format
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
            
            # Check if best model
            current_metric = landmark_metrics.get('nme', 1.0)  # Lower is better for NME
            if current_metric < best_metric or best_metric == 0:
                best_metric = current_metric
                best_epoch = epoch
                
                # Save best model
                best_path = solver.output_dir / 'best_model.pth'
                dist.save_on_master(solver.state_dict(epoch), best_path)
                print(f"New best model saved! NME: {current_metric:.4f}")
        
        # Log training information
        trainer.log_training_info(epoch, phase, train_stats, val_stats)
        
        # Save checkpoint
        if (epoch + 1) % cfg.checkpoint_step == 0:
            checkpoint_path = solver.output_dir / f'checkpoint{epoch:04}.pth'
            dist.save_on_master(solver.state_dict(epoch), checkpoint_path)
        
        # Visualize predictions (optional)
        # Visualize predictions (optional)
        if args.visualize and epoch % args.vis_freq == 0:
            print(f"Visualization is currently disabled. Skipping...")
            # TODO: Fix visualization later
            pass

            
            print(f"\nTraining completed!")
            print(f"Best model: Epoch {best_epoch} with NME: {best_metric:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RT-DETR Polar Heatmap Face Landmark Training')
    
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
    
    # Validation and visualization
    parser.add_argument('--val-freq', type=int, default=5,
                       help='Validation frequency')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization')
    parser.add_argument('--vis-freq', type=int, default=10,
                       help='Visualization frequency')
    
    args = parser.parse_args()
    
    # Import required modules
    #from src.data import get_coco_api_from_dataset
    from src.solver.det_engine import evaluate
    
    main(args)