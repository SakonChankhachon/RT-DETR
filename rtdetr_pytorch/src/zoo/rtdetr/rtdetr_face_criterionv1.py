"""Loss functions for face detection and landmark localization"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register
from .rtdetr_criterion import SetCriterion


@register
class FaceLandmarkCriterion(SetCriterion):
    """Criterion for face detection and landmark localization"""
    
    __share__ = ['num_classes', 'num_landmarks']
    
    def __init__(self, 
                 matcher, 
                 weight_dict, 
                 losses, 
                 num_landmarks=5,
                 landmark_loss_type='l1',
                 alpha=0.2, 
                 gamma=2.0, 
                 eos_coef=1e-4, 
                 num_classes=1):
        
        # Add landmarks to losses if not present
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
        self.landmark_loss_type = landmark_loss_type
        
        # Add landmark loss weight if not specified
        if 'loss_landmarks' not in self.weight_dict:
            self.weight_dict['loss_landmarks'] = 5.0
    
    def loss_landmarks(self, outputs, targets, indices, num_boxes, log=True):
        """Compute landmark localization loss"""
        # Check if landmarks are in outputs
        if 'pred_landmarks' not in outputs:
            # Return zero loss if no landmarks
            device = outputs['pred_logits'].device if 'pred_logits' in outputs else 'cpu'
            return {'loss_landmarks': torch.tensor(0.0, device=device)}
        
        idx = self._get_src_permutation_idx(indices)
        src_landmarks = outputs['pred_landmarks'][idx]
        
        # Get target landmarks
        target_landmarks = []
        for t, (_, i) in zip(targets, indices):
            if 'landmarks' in t and len(t['landmarks']) > 0:
                target_landmarks.append(t['landmarks'][i])
            else:
                # Create dummy landmarks if not present
                num_targets = len(i)
                dummy = torch.zeros(num_targets, self.num_landmarks * 2, device=src_landmarks.device)
                target_landmarks.append(dummy)
        
        if len(target_landmarks) == 0:
            # No targets, return zero loss
            return {'loss_landmarks': torch.tensor(0.0, device=src_landmarks.device)}
        
        target_landmarks = torch.cat(target_landmarks, dim=0)
        
        # Compute loss
        if src_landmarks.numel() == 0 or target_landmarks.numel() == 0:
            return {'loss_landmarks': torch.tensor(0.0, device=src_landmarks.device)}
        
        if self.landmark_loss_type == 'l1':
            loss_landmarks = F.l1_loss(src_landmarks, target_landmarks, reduction='none')
        elif self.landmark_loss_type == 'l2':
            loss_landmarks = F.mse_loss(src_landmarks, target_landmarks, reduction='none')
        else:
            loss_landmarks = F.smooth_l1_loss(src_landmarks, target_landmarks, reduction='none')
        
        losses = {'loss_landmarks': loss_landmarks.sum() / max(num_boxes, 1)}
        
        return losses
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if loss == 'landmarks':
            return self.loss_landmarks(outputs, targets, indices, num_boxes, **kwargs)
        else:
            return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
