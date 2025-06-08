"""
Polar Heatmap-based Face Landmark Decoder for RT-DETR
Uses polar coordinates and orientation tokens for better landmark detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List

from src.core import register
from .rtdetr_decoder import RTDETRTransformer, get_contrastive_denoising_training_group
from .utils import inverse_sigmoid


class PolarHeatmapHead(nn.Module):
    """Polar coordinate heatmap head for landmark detection"""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_landmarks: int = 5,
        num_orientations: int = 8,  # Number of orientation bins
        heatmap_size: int = 64,     # Size of heatmap
        temperature: float = 1.0
    ):
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
        x = torch.linspace(-1, 1, self.heatmap_size)
        y = torch.linspace(-1, 1, self.heatmap_size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        # Clone tensors to avoid memory aliasing issues
        self.register_buffer('grid_x', xx.clone())
        self.register_buffer('grid_y', yy.clone())
        self.register_buffer('grid_r', torch.sqrt(xx**2 + yy**2))
        self.register_buffer('grid_theta', torch.atan2(yy, xx))

    def forward(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        heatmaps = heatmaps.view(
            batch_size,
            num_queries,
            self.num_landmarks,
            self.heatmap_size,
            self.heatmap_size
        )
        heatmaps = torch.sigmoid(heatmaps)

        # Predict orientations
        orientations = self.orientation_head(features)
        orientations = orientations.view(
            batch_size,
            num_queries,
            self.num_landmarks,
            self.num_orientations
        )
        orientations = torch.softmax(orientations / self.temperature, dim=-1)

        # Predict radii (relative to box size)
        radii = self.radius_head(features)
        radii = radii.view(batch_size, num_queries, self.num_landmarks)
        radii = torch.sigmoid(radii) * 0.5  # Limit to half box size

        return heatmaps, orientations, radii

    def decode_landmarks(
        self,
        heatmaps: torch.Tensor,
        orientations: torch.Tensor,
        radii: torch.Tensor,
        boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode landmarks from polar representation

        Args:
            heatmaps: [batch_size, num_queries, num_landmarks, H, W]
            orientations: [batch_size, num_queries, num_landmarks, num_orientations]
            radii: [batch_size, num_queries, num_landmarks]
            boxes: [batch_size, num_queries, 4] in cxcywh format

        Returns:
            landmarks: [batch_size, num_queries, num_landmarks * 2] in normalized [0,1]
        """
        batch_size = heatmaps.shape[0]
        device = boxes.device

        # Extract box centers and sizes
        cx, cy, w, h = boxes.unbind(-1)  # each [B, Q]

        landmarks = []
        for i in range(self.num_landmarks):
            heatmap = heatmaps[:, :, i]  # [B, Q, H, W]
            B, Q, H, W = heatmap.shape

            # Flatten H*W
            heatmap_flat = heatmap.contiguous().view(B, Q, H * W)
            peak_idx = heatmap_flat.argmax(dim=-1)  # [B, Q]

            peak_y = peak_idx // W  # [B, Q]
            peak_x = peak_idx % W   # [B, Q]

            peak_x_norm = 2.0 * peak_x.float() / (W - 1) - 1.0
            peak_y_norm = 2.0 * peak_y.float() / (H - 1) - 1.0

            # Orientation (polar angle)
            orientation_probs = orientations[:, :, i]  # [B, Q, O]
            angle_bins = torch.linspace(0, 2 * math.pi, self.num_orientations + 1)[:-1].to(device)
            angle = orientation_probs @ angle_bins  # [B, Q]

            # Radius
            r = radii[:, :, i]  # [B, Q]

            dx = r * torch.cos(angle) * w  # [B, Q]
            dy = r * torch.sin(angle) * h  # [B, Q]

            heatmap_weight = 0.7
            polar_weight = 0.3

            lmk_x = cx + heatmap_weight * (peak_x_norm * w * 0.5) + polar_weight * dx
            lmk_y = cy + heatmap_weight * (peak_y_norm * h * 0.5) + polar_weight * dy

            landmarks.extend([lmk_x, lmk_y])

        landmarks = torch.stack(landmarks, dim=-1)  # [B, Q, 2*L]
        landmarks = torch.sigmoid(landmarks)  # normalize to [0,1]
        return landmarks


@register
class RTDETRTransformerPolarLandmark(RTDETRTransformer):
    """RT-DETR Transformer with Polar Heatmap-based Landmark Detection"""

    __share__ = ['num_classes', 'num_landmarks']

    def __init__(
        self,
        num_classes=1,
        num_landmarks=5,
        num_orientations=8,
        heatmap_size=64,
        landmark_loss_weight=5.0,
        in_channels=[512, 1024, 2048],  # Changed to list to match parent
        # Include all parent class parameters
        hidden_dim=256,
        num_queries=300,
        position_embed_type='sine',
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        **kwargs
    ):
        # Make feat_channels match in_channels if not provided
        if 'feat_channels' not in kwargs and feat_channels == [512, 1024, 2048]:
            feat_channels = in_channels
            
        # Initialize parent class with all parameters
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            in_channels=in_channels,
            num_queries=num_queries,
            position_embed_type=position_embed_type,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_decoder_points=num_decoder_points,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learnt_init_query=learnt_init_query,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss
        )
        
        # Debug: Check what attributes were set by parent
        print(f"DEBUG: After parent init, eval_idx = {getattr(self, 'eval_idx', 'NOT SET')}")
        print(f"DEBUG: num_decoder_layers = {self.num_decoder_layers}")
        print(f"DEBUG: decoder type = {type(self.decoder)}")

        # Face landmark specific attributes
        self.num_landmarks = num_landmarks
        self.num_orientations = num_orientations
        self.heatmap_size = heatmap_size
        self.landmark_loss_weight = landmark_loss_weight
        
        # Ensure eval_idx is set (inherited from parent)
        if not hasattr(self, 'eval_idx'):
            print(f"WARNING: eval_idx not set by parent, setting to {eval_idx}")
            self.eval_idx = eval_idx
        
        # Also ensure decoder has eval_idx
        if hasattr(self, 'decoder') and hasattr(self.decoder, 'eval_idx'):
            print(f"DEBUG: decoder.eval_idx = {self.decoder.eval_idx}")
        else:
            print(f"WARNING: decoder does not have eval_idx")

        # Replace each decoder layer's landmark head
        self.dec_landmark_heads = nn.ModuleList([
            PolarHeatmapHead(
                hidden_dim=self.hidden_dim,
                num_landmarks=num_landmarks,
                num_orientations=num_orientations,
                heatmap_size=heatmap_size
            )
            for _ in range(self.num_decoder_layers)
        ])

        # Encoder landmark head (only used if aux_loss is True)
        self.enc_landmark_head = PolarHeatmapHead(
            hidden_dim=self.hidden_dim,
            num_landmarks=num_landmarks,
            num_orientations=num_orientations,
            heatmap_size=heatmap_size
        )

    def forward(self, feats: List[torch.Tensor], targets=None):
        """
        Forward pass with polar landmark prediction.

        Args:
            feats: list of multi-scale feature tensors from backbone,
                   each is [B, C_i, H_i, W_i]
            targets: (optional) ground-truth annotations for training

        Returns:
            A dict containing:
              - 'pred_logits': [B, Q, num_classes]
              - 'pred_boxes' : [B, Q, 4]
              - 'pred_landmarks': [B, Q, num_landmarks*2]
            If training & aux_loss, also returns 'aux_outputs' list.
        """
        # 1) Encode all multi-scale features:
        memory, spatial_shapes, level_start_index = self._get_encoder_input(feats)

        # 2) Prepare denoising (if enabled)
        if self.training and self.num_denoising > 0 and targets is not None:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale
                )
        else:
            denoising_class = None
            denoising_bbox_unact = None
            attn_mask = None
            dn_meta = None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(
                memory,
                spatial_shapes,
                denoising_class,
                denoising_bbox_unact
            )

        out_bboxes = []
        out_logits = []
        out_landmarks = []

        output = target
        ref_points_detach = torch.sigmoid(init_ref_points_unact)

        # 3) Pass through each TransformerDecoder layer
        for i, layer in enumerate(self.decoder.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = self.query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                spatial_shapes,
                level_start_index,
                attn_mask,
                memory_mask=None,
                query_pos_embed=query_pos_embed
            )

            # Box & class predictions
            inter_ref_bbox = torch.sigmoid(
                self.dec_bbox_head[i](output) +
                inverse_sigmoid(ref_points_detach)
            )
            inter_class_logits = self.dec_score_head[i](output)

            # Landmark prediction via PolarHeatmapHead
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
                    out_bboxes.append(torch.sigmoid(
                        self.dec_bbox_head[i](output) +
                        inverse_sigmoid(ref_points)
                    ))
            else:
                # validation: if this is the chosen eval_idx, or if eval_idx < 0 (use last layer),
                # always fall back to the very last layer
                last_layer = self.num_decoder_layers - 1
                if i == self.eval_idx or i == last_layer:
                    out_logits.append(inter_class_logits)
                    out_bboxes.append(inter_ref_bbox)
                    out_landmarks.append(landmarks)
                    break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        # 4) Stack across decoder layers
        out_bboxes = torch.stack(out_bboxes)        # [num_layers, B, Q, 4]
        out_logits = torch.stack(out_logits)        # [num_layers, B, Q, num_classes]
        out_landmarks = torch.stack(out_landmarks)  # [num_layers, B, Q, num_landmarks*2]

        # 5) If denoising, split off the "denoising" portion
        if self.training and dn_meta is not None:
            dn_split = dn_meta['dn_num_split']
            (
                dn_out_bboxes, out_bboxes
            ) = torch.split(out_bboxes, dn_split, dim=2)
            (
                dn_out_logits, out_logits
            ) = torch.split(out_logits, dn_split, dim=2)
            (
                dn_out_landmarks, out_landmarks
            ) = torch.split(out_landmarks, dn_split, dim=2)

        # 6) Final predictions (last decoder layer)
        out = {
            'pred_logits': out_logits[-1],       # [B, Q, num_classes]
            'pred_boxes': out_bboxes[-1],        # [B, Q, 4]
            'pred_landmarks': out_landmarks[-1]  # [B, Q, num_landmarks*2]
        }

        # 7) If training & aux_loss, append each intermediate layer's outputs
        if self.training and self.aux_loss:
            out['aux_outputs'] = []
            for j in range(len(out_logits) - 1):
                aux_out = {
                    'pred_logits': out_logits[j],
                    'pred_boxes': out_bboxes[j],
                    'pred_landmarks': out_landmarks[j]
                }
                out['aux_outputs'].append(aux_out)

            # Skip encoder auxiliary outputs for now to avoid dimension mismatch
            # The encoder operates on all spatial positions (8400) while decoder
            # operates on selected queries (300), making direct landmark prediction difficult

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = []
                for j in range(len(dn_out_logits)):
                    dn_aux_out = {
                        'pred_logits': dn_out_logits[j],
                        'pred_boxes': dn_out_bboxes[j],
                        'pred_landmarks': dn_out_landmarks[j]
                    }
                    out['dn_aux_outputs'].append(dn_aux_out)
                out['dn_meta'] = dn_meta

        return out


class PolarLandmarkLoss(nn.Module):
    """Loss function for polar heatmap-based landmarks"""

    def __init__(
        self,
        num_landmarks=5,
        heatmap_loss_weight=1.0,
        orientation_loss_weight=0.5,
        radius_loss_weight=0.5,
        coordinate_loss_weight=1.0
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.heatmap_loss_weight = heatmap_loss_weight
        self.orientation_loss_weight = orientation_loss_weight
        self.radius_loss_weight = radius_loss_weight
        self.coordinate_loss_weight = coordinate_loss_weight

    def forward(
        self,
        pred_landmarks: torch.Tensor,
        target_landmarks: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> dict:
        """
        Compute polar landmark loss

        Args:
            pred_landmarks: [batch_size, num_queries, num_landmarks * 2]
            target_landmarks: [batch_size, num_landmarks * 2]
            pred_boxes: [batch_size, num_queries, 4]
            target_boxes: [batch_size, ???, 4]

        Returns:
            Dictionary containing:
              - 'loss_landmarks' : L1 between predicted & target landmark coordinates
        """
        coord_loss = F.l1_loss(pred_landmarks, target_landmarks, reduction='mean')
        losses = {
            'loss_landmarks': coord_loss * self.coordinate_loss_weight,
        }
        return losses