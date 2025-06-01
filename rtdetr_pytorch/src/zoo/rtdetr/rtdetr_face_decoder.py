"""
Polar Heatmap-based Face Landmark Decoder for RT-DETR
Uses polar coordinates and orientation tokens for better landmark detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List

from src.core import register
from .rtdetr_decoder import RTDETRTransformer, MLP
from .utils import inverse_sigmoid, bias_init_with_prob


class PolarHeatmapHead(nn.Module):
    """Polar coordinate heatmap head for landmark detection"""
    
    def __init__(self, 
                 hidden_dim: int = 256,
                 num_landmarks: int = 5,
                 num_orientations: int = 8,  # Number of orientation bins
                 heatmap_size: int = 64,     # Size of heatmap
                 temperature: float = 1.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_landmarks = num_landmarks
        self.num_orientations = num_orientations
        self.heatmap_size = heatmap_size
        self.temperature = temperature
        
        # Heatmap prediction head
        self.heatmap_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_landmarks * heatmap_size * heatmap_size)
        )
        
        # Orientation prediction head (polar angle)
        self.orientation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_landmarks * num_orientations)
        )
        
        # Radial distance prediction head
        self.radius_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_landmarks)  # One radius per landmark
        )
        
        # Create coordinate grids for heatmap
        self._create_coordinate_grids()
        
    def _create_coordinate_grids(self):
        """Create coordinate grids for heatmap generation"""
        # Create normalized grid coordinates [-1, 1]
        x = torch.linspace(-1, 1, self.heatmap_size)
        y = torch.linspace(-1, 1, self.heatmap_size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        self.register_buffer('grid_x', xx)
        self.register_buffer('grid_y', yy)
        
        # Convert to polar coordinates
        self.register_buffer('grid_r', torch.sqrt(xx**2 + yy**2))
        self.register_buffer('grid_theta', torch.atan2(yy, xx))
        
    def forward(self, features: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Query features [batch_size, num_queries, hidden_dim]
            boxes: Predicted boxes [batch_size, num_queries, 4] in cxcywh format
            
        Returns:
            heatmaps: [batch_size, num_queries, num_landmarks, H, W]
            orientations: [batch_size, num_queries, num_landmarks, num_orientations]
            radii: [batch_size, num_queries, num_landmarks]
        """
        batch_size, num_queries = features.shape[:2]
        
        # Predict heatmaps
        heatmaps = self.heatmap_head(features)
        heatmaps = heatmaps.view(batch_size, num_queries, self.num_landmarks, 
                                 self.heatmap_size, self.heatmap_size)
        heatmaps = F.sigmoid(heatmaps)  # Normalize to [0, 1]
        
        # Predict orientations
        orientations = self.orientation_head(features)
        orientations = orientations.view(batch_size, num_queries, self.num_landmarks, 
                                       self.num_orientations)
        orientations = F.softmax(orientations / self.temperature, dim=-1)
        
        # Predict radii (relative to box size)
        radii = self.radius_head(features)
        radii = radii.view(batch_size, num_queries, self.num_landmarks)
        radii = F.sigmoid(radii) * 0.5  # Limit to half box size
        
        return heatmaps, orientations, radii
    
    def decode_landmarks(self, heatmaps, orientations, radii, boxes):
        """Decode landmarks from polar representation
        
        Args:
            heatmaps: [batch_size, num_queries, num_landmarks, H, W]
            orientations: [batch_size, num_queries, num_landmarks, num_orientations]
            radii: [batch_size, num_queries, num_landmarks]
            boxes: [batch_size, num_queries, 4] in cxcywh format
            
        Returns:
            landmarks: [batch_size, num_queries, num_landmarks * 2] in normalized coordinates
        """
        batch_size, num_queries = boxes.shape[:2]
        device = boxes.device
        
        # Get box centers and sizes
        cx, cy, w, h = boxes.unbind(-1)
        
        landmarks = []
        
        for i in range(self.num_landmarks):
            # Get peak location from heatmap
            heatmap = heatmaps[:, :, i]  # [batch_size, num_queries, H, W]
            heatmap_flat = heatmap.view(batch_size, num_queries, -1)
            peak_idx = heatmap_flat.argmax(dim=-1)  # [batch_size, num_queries]
            
            # Convert to 2D coordinates
            peak_y = peak_idx // self.heatmap_size
            peak_x = peak_idx % self.heatmap_size
            
            # Normalize to [-1, 1]
            peak_x_norm = 2.0 * peak_x.float() / (self.heatmap_size - 1) - 1.0
            peak_y_norm = 2.0 * peak_y.float() / (self.heatmap_size - 1) - 1.0
            
            # Get orientation (polar angle)
            orientation_probs = orientations[:, :, i]  # [batch_size, num_queries, num_orientations]
            angle_bins = torch.linspace(0, 2*math.pi, self.num_orientations + 1)[:-1].to(device)
            angle = (orientation_probs @ angle_bins) # Weighted average of angles
            
            # Get radius
            r = radii[:, :, i]  # [batch_size, num_queries]
            
            # Convert polar to cartesian (relative to box)
            dx = r * torch.cos(angle) * w
            dy = r * torch.sin(angle) * h
            
            # Combine heatmap peak with polar offset
            # Use heatmap for coarse location, polar for fine adjustment
            heatmap_weight = 0.7
            polar_weight = 0.3
            
            lmk_x = cx + heatmap_weight * (peak_x_norm * w * 0.5) + polar_weight * dx
            lmk_y = cy + heatmap_weight * (peak_y_norm * h * 0.5) + polar_weight * dy
            
            landmarks.extend([lmk_x, lmk_y])
        
        # Stack landmarks
        landmarks = torch.stack(landmarks, dim=-1)  # [batch_size, num_queries, num_landmarks * 2]
        
        # Normalize to [0, 1] if needed
        landmarks = torch.sigmoid(landmarks)
        
        return landmarks


@register
class RTDETRTransformerPolarLandmark(RTDETRTransformer):
    """RT-DETR Transformer with Polar Heatmap-based Landmark Detection"""
    
    __share__ = ['num_classes', 'num_landmarks']
    
    def __init__(self,
                 num_classes=1,
                 num_landmarks=5,
                 num_orientations=8,
                 heatmap_size=64,
                 landmark_loss_weight=5.0,
                 **kwargs):
        
        # Remove num_landmarks from kwargs to avoid conflict
        kwargs.pop('num_landmarks', None)
        
        super().__init__(num_classes=num_classes, **kwargs)
        
        self.num_landmarks = num_landmarks
        self.num_orientations = num_orientations
        self.heatmap_size = heatmap_size
        self.landmark_loss_weight = landmark_loss_weight
        
        # Replace landmark heads with polar heatmap heads
        self.dec_landmark_heads = nn.ModuleList([
            PolarHeatmapHead(
                hidden_dim=self.hidden_dim,
                num_landmarks=num_landmarks,
                num_orientations=num_orientations,
                heatmap_size=heatmap_size
            ) for _ in range(self.num_decoder_layers)
        ])
        
        # Encoder landmark head
        self.enc_landmark_head = PolarHeatmapHead(
            hidden_dim=self.hidden_dim,
            num_landmarks=num_landmarks,
            num_orientations=num_orientations,
            heatmap_size=heatmap_size
        )
        
    def forward(self, feats, targets=None):
        """Forward pass with polar landmark prediction"""
        
        # Get encoder features and initial predictions
        memory, spatial_shapes, level_start_index = self._get_encoder_input(feats)
        
        # Prepare denoising if training
        if self.training and self.num_denoising > 0 and targets is not None:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets, self.num_classes, self.num_queries,
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale
                )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
        
        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)
        
        # Modified decoder forward to collect intermediate features
        out_bboxes = []
        out_logits = []
        out_landmarks = []
        
        output = target
        ref_points_detach = F.sigmoid(init_ref_points_unact)
        
        for i, layer in enumerate(self.decoder.layers):
            # Get query features
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = self.query_pos_head(ref_points_detach)
            
            # Transformer layer
            output = layer(output, ref_points_input, memory,
                          memory_spatial_shapes, memory_level_start_index,
                          attn_mask, memory_mask=None, query_pos_embed=query_pos_embed)
            
            # Predict boxes and classes
            inter_ref_bbox = F.sigmoid(self.dec_bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            inter_class_logits = self.dec_score_head[i](output)
            
            # Predict landmarks using polar heatmap
            heatmaps, orientations, radii = self.dec_landmark_heads[i](output, inter_ref_bbox)
            landmarks = self.dec_landmark_heads[i].decode_landmarks(
                heatmaps, orientations, radii, inter_ref_bbox
            )
            
            if self.training:
                out_logits.append(inter_class_logits)
                out_landmarks.append(landmarks)
                if i == 0:
                    out_bboxes.append(inter_ref_bbox)
                else:
                    out_bboxes.append(F.sigmoid(self.dec_bbox_head[i](output) + inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                out_logits.append(inter_class_logits)
                out_bboxes.append(inter_ref_bbox)
                out_landmarks.append(landmarks)
                break
            
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox
        
        # Stack outputs
        out_bboxes = torch.stack(out_bboxes)
        out_logits = torch.stack(out_logits)
        out_landmarks = torch.stack(out_landmarks)
        
        # Split denoising if needed
        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_landmarks, out_landmarks = torch.split(out_landmarks, dn_meta['dn_num_split'], dim=2)
        
        # Prepare outputs
        out = {
            'pred_logits': out_logits[-1],
            'pred_boxes': out_bboxes[-1],
            'pred_landmarks': out_landmarks[-1]
        }
        
        # Add auxiliary outputs if training
        if self.training and self.aux_loss:
            out['aux_outputs'] = []
            for i in range(len(out_logits) - 1):
                aux_out = {
                    'pred_logits': out_logits[i],
                    'pred_boxes': out_bboxes[i],
                    'pred_landmarks': out_landmarks[i]
                }
                out['aux_outputs'].append(aux_out)
            
            # Add encoder outputs
            enc_heatmaps, enc_orientations, enc_radii = self.enc_landmark_head(
                self.enc_output(memory), enc_topk_bboxes
            )
            enc_landmarks = self.enc_landmark_head.decode_landmarks(
                enc_heatmaps, enc_orientations, enc_radii, enc_topk_bboxes
            )
            
            out['aux_outputs'].append({
                'pred_logits': enc_topk_logits,
                'pred_boxes': enc_topk_bboxes,
                'pred_landmarks': enc_landmarks
            })
            
            # Add denoising outputs if available
            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = []
                for i in range(len(dn_out_logits)):
                    dn_aux_out = {
                        'pred_logits': dn_out_logits[i],
                        'pred_boxes': dn_out_bboxes[i],
                        'pred_landmarks': dn_out_landmarks[i]
                    }
                    out['dn_aux_outputs'].append(dn_aux_out)
                out['dn_meta'] = dn_meta
        
        return out


class PolarLandmarkLoss(nn.Module):
    """Loss function for polar heatmap-based landmarks"""
    
    def __init__(self, 
                 num_landmarks=5,
                 heatmap_loss_weight=1.0,
                 orientation_loss_weight=0.5,
                 radius_loss_weight=0.5,
                 coordinate_loss_weight=1.0):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.heatmap_loss_weight = heatmap_loss_weight
        self.orientation_loss_weight = orientation_loss_weight
        self.radius_loss_weight = radius_loss_weight
        self.coordinate_loss_weight = coordinate_loss_weight
        
    def forward(self, pred_landmarks, target_landmarks, pred_boxes, target_boxes):
        """Compute polar landmark loss
        
        Args:
            pred_landmarks: Predicted landmarks [batch_size, num_queries, num_landmarks * 2]
            target_landmarks: Target landmarks [batch_size, num_landmarks * 2]
            pred_boxes: Predicted boxes for computing relative positions
            target_boxes: Target boxes
            
        Returns:
            Dictionary of losses
        """
        # Direct coordinate loss (L1)
        coord_loss = F.l1_loss(pred_landmarks, target_landmarks, reduction='mean')
        
        # Additional losses can be computed here if needed
        # For example: visibility loss, confidence loss, etc.
        
        losses = {
            'loss_landmarks': coord_loss * self.coordinate_loss_weight,
        }
        
        return losses