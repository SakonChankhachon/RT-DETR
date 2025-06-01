"""
Criterion for polar heatmap-based face landmark detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

from src.core import register
from .rtdetr_criterion import SetCriterion
from .matcher import HungarianMatcher


def gaussian_2d(shape, sigma=1.0):
    """Generate 2D Gaussian kernel"""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def generate_gaussian_heatmaps(landmarks, heatmap_size=64, sigma=2.0):
    """Generate Gaussian heatmaps for landmarks
    
    Args:
        landmarks: [batch_size, num_landmarks, 2] normalized coordinates
        heatmap_size: Size of output heatmap
        sigma: Gaussian sigma
        
    Returns:
        heatmaps: [batch_size, num_landmarks, heatmap_size, heatmap_size]
    """
    batch_size, num_landmarks = landmarks.shape[:2]
    device = landmarks.device
    
    heatmaps = torch.zeros(batch_size, num_landmarks, heatmap_size, heatmap_size, device=device)
    
    # Generate Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = gaussian_2d((kernel_size, kernel_size), sigma)
    kernel = torch.from_numpy(kernel).float().to(device)
    
    for b in range(batch_size):
        for l in range(num_landmarks):
            # Convert normalized coordinates to heatmap coordinates
            x = int(landmarks[b, l, 0] * heatmap_size)
            y = int(landmarks[b, l, 1] * heatmap_size)
            
            # Skip if outside bounds
            if x < 0 or x >= heatmap_size or y < 0 or y >= heatmap_size:
                continue
                
            # Calculate kernel bounds
            x_min = max(0, x - kernel_size // 2)
            x_max = min(heatmap_size, x + kernel_size // 2 + 1)
            y_min = max(0, y - kernel_size // 2)
            y_max = min(heatmap_size, y + kernel_size // 2 + 1)
            
            # Calculate kernel region
            kernel_x_min = max(0, kernel_size // 2 - x)
            kernel_x_max = kernel_x_min + (x_max - x_min)
            kernel_y_min = max(0, kernel_size // 2 - y)
            kernel_y_max = kernel_y_min + (y_max - y_min)
            
            # Place kernel
            heatmaps[b, l, y_min:y_max, x_min:x_max] = kernel[
                kernel_y_min:kernel_y_max,
                kernel_x_min:kernel_x_max
            ]
    
    return heatmaps


@register
class PolarLandmarkCriterion(SetCriterion):
    """Criterion for polar heatmap-based face landmark detection"""
    
    __share__ = ['num_classes', 'num_landmarks']
    
    def __init__(self,
                 matcher,
                 weight_dict,
                 losses,
                 num_landmarks=5,
                 num_orientations=8,
                 heatmap_size=64,
                 heatmap_sigma=2.0,
                 heatmap_loss_type='mse',
                 alpha=0.2,
                 gamma=2.0,
                 eos_coef=1e-4,
                 num_classes=1):
        
        # Ensure landmarks is in losses
        if 'landmarks' not in losses:
            losses = list(losses) + ['landmarks']
        
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            eos_coef=eos_coef,
            num_classes=num_classes
        )
        
        self.num_landmarks = num_landmarks
        self.num_orientations = num_orientations
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_loss_type = heatmap_loss_type
        
        # Add default weights if not specified
        default_weights = {
            'loss_landmarks': 5.0,
            'loss_heatmap': 2.0,
            'loss_orientation': 1.0,
            'loss_radius': 1.0,
        }
        
        for k, v in default_weights.items():
            if k not in self.weight_dict:
                self.weight_dict[k] = v
    
    def loss_landmarks(self, outputs, targets, indices, num_boxes, log=True):
        """Compute all landmark-related losses"""
        
        losses = {}
        
        # Get predictions
        pred_landmarks = outputs.get('pred_landmarks')
        if pred_landmarks is None:
            # Return zero losses if no landmarks
            device = outputs['pred_logits'].device
            losses['loss_landmarks'] = torch.tensor(0.0, device=device)
            losses['loss_heatmap'] = torch.tensor(0.0, device=device)
            losses['loss_orientation'] = torch.tensor(0.0, device=device)
            losses['loss_radius'] = torch.tensor(0.0, device=device)
            return losses
        
        # Get matched predictions and targets
        idx = self._get_src_permutation_idx(indices)
        src_landmarks = pred_landmarks[idx]
        src_boxes = outputs['pred_boxes'][idx]
        
        # Get target landmarks and boxes
        target_landmarks = []
        target_boxes = []
        for t, (_, i) in zip(targets, indices):
            if 'landmarks' in t and len(t['landmarks']) > 0:
                target_landmarks.append(t['landmarks'][i])
                target_boxes.append(t['boxes'][i])
            else:
                # Skip if no landmarks
                continue
        
        if len(target_landmarks) == 0:
            # No targets, return zero losses
            device = pred_landmarks.device
            losses['loss_landmarks'] = torch.tensor(0.0, device=device)
            losses['loss_heatmap'] = torch.tensor(0.0, device=device)
            losses['loss_orientation'] = torch.tensor(0.0, device=device)
            losses['loss_radius'] = torch.tensor(0.0, device=device)
            return losses
        
        target_landmarks = torch.cat(target_landmarks, dim=0)
        target_boxes = torch.cat(target_boxes, dim=0)
        
        # 1. Direct coordinate loss (main loss)
        coord_loss = F.l1_loss(src_landmarks, target_landmarks, reduction='none')
        losses['loss_landmarks'] = coord_loss.sum() / max(num_boxes, 1)
        
        # 2. Heatmap loss (if heatmaps are provided in outputs)
        if 'pred_heatmaps' in outputs:
            pred_heatmaps = outputs['pred_heatmaps'][idx]
            
            # Generate target heatmaps
            target_landmarks_2d = target_landmarks.view(-1, self.num_landmarks, 2)
            target_heatmaps = generate_gaussian_heatmaps(
                target_landmarks_2d, 
                self.heatmap_size, 
                self.heatmap_sigma
            )
            
            if self.heatmap_loss_type == 'mse':
                heatmap_loss = F.mse_loss(pred_heatmaps, target_heatmaps, reduction='mean')
            else:  # focal loss for heatmaps
                heatmap_loss = self._focal_loss_for_heatmap(pred_heatmaps, target_heatmaps)
            
            losses['loss_heatmap'] = heatmap_loss
        
        # 3. Orientation loss (if orientations are provided)
        if 'pred_orientations' in outputs:
            pred_orientations = outputs['pred_orientations'][idx]
            
            # Compute target orientations from landmark positions relative to box center
            cx, cy = target_boxes[:, 0], target_boxes[:, 1]
            target_orientations = []
            
            for i in range(self.num_landmarks):
                lmk_x = target_landmarks[:, i*2]
                lmk_y = target_landmarks[:, i*2 + 1]
                
                # Compute angle from box center to landmark
                angle = torch.atan2(lmk_y - cy, lmk_x - cx)
                # Convert to [0, 2π]
                angle = (angle + 2 * np.pi) % (2 * np.pi)
                # Convert to bin index
                bin_idx = (angle / (2 * np.pi) * self.num_orientations).long()
                target_orientations.append(bin_idx)
            
            target_orientations = torch.stack(target_orientations, dim=1)
            
            # Cross entropy loss for orientation
            orientation_loss = F.cross_entropy(
                pred_orientations.view(-1, self.num_orientations),
                target_orientations.view(-1),
                reduction='mean'
            )
            losses['loss_orientation'] = orientation_loss
        
        # 4. Radius loss (if radii are provided)
        if 'pred_radii' in outputs:
            pred_radii = outputs['pred_radii'][idx]
            
            # Compute target radii
            cx, cy, w, h = target_boxes.unbind(-1)
            target_radii = []
            
            for i in range(self.num_landmarks):
                lmk_x = target_landmarks[:, i*2]
                lmk_y = target_landmarks[:, i*2 + 1]
                
                # Compute distance from box center to landmark
                dx = (lmk_x - cx) / w
                dy = (lmk_y - cy) / h
                radius = torch.sqrt(dx**2 + dy**2)
                target_radii.append(radius)
            
            target_radii = torch.stack(target_radii, dim=1)
            
            # L1 loss for radius
            radius_loss = F.l1_loss(pred_radii, target_radii, reduction='mean')
            losses['loss_radius'] = radius_loss
        
        return losses
    
    def _focal_loss_for_heatmap(self, pred, target, alpha=2, beta=4):
        """Focal loss for heatmap regression"""
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        
        neg_weights = torch.pow(1 - target, beta)
        
        pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
        neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_mask
        
        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if loss == 'landmarks':
            return self.loss_landmarks(outputs, targets, indices, num_boxes, **kwargs)
        else:
            return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)