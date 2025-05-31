"""Modified RT-DETR decoder for face landmark detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict

from .rtdetr_decoder import (
    RTDETRTransformer, TransformerDecoder, TransformerDecoderLayer,
    MLP, MSDeformableAttention, get_contrastive_denoising_training_group
)
from .utils import inverse_sigmoid, bias_init_with_prob
from src.core import register


@register
class RTDETRTransformerFaceLandmark(RTDETRTransformer):
    """RT-DETR Transformer modified for face detection and landmark localization"""
    
    __share__ = ['num_classes', 'num_landmarks']
    
    def __init__(self,
                 num_classes=1,  # Only face class
                 num_landmarks=5,  # Number of landmark points
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
                 landmark_noise_scale=0.1,  # Noise scale for landmarks
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True):
        
        # Initialize parent class
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
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
        
        self.num_landmarks = num_landmarks
        self.landmark_noise_scale = landmark_noise_scale
        
        # Add landmark prediction heads for each decoder layer
        self.dec_landmark_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, num_landmarks * 2, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        
        # Encoder landmark head
        self.enc_landmark_head = MLP(hidden_dim, hidden_dim, num_landmarks * 2, num_layers=3)
        
        # Initialize landmark heads
        self._reset_landmark_parameters()
    
    def _reset_landmark_parameters(self):
        """Initialize landmark prediction heads"""
        for landmark_head in self.dec_landmark_head:
            init.constant_(landmark_head.layers[-1].weight, 0)
            init.constant_(landmark_head.layers[-1].bias, 0)
        
        init.constant_(self.enc_landmark_head.layers[-1].weight, 0)
        init.constant_(self.enc_landmark_head.layers[-1].bias, 0)
    
    def forward(self, feats, targets=None):
        # Use parent class forward pass
        outputs = super().forward(feats, targets)
        
        # Get dimensions
        if 'pred_logits' in outputs:
            batch_size = outputs['pred_logits'].shape[0]
            num_queries = outputs['pred_logits'].shape[1]
        else:
            # Handle case where there's no pred_logits
            return outputs
        
        # Predict landmarks for main output
        # Get the decoder output from the last layer
        if hasattr(self, 'dec_landmark_head') and len(self.dec_landmark_head) > 0:
            # For now, use dummy landmarks until we implement proper prediction
            pred_landmarks = torch.zeros(
                batch_size, num_queries, self.num_landmarks * 2,
                device=outputs['pred_logits'].device,
                dtype=outputs['pred_logits'].dtype
            )
            outputs['pred_landmarks'] = pred_landmarks
        
        # Also add landmarks to auxiliary outputs
        if 'aux_outputs' in outputs:
            for i, aux_out in enumerate(outputs['aux_outputs']):
                if 'pred_logits' in aux_out:
                    # Add dummy landmarks to each auxiliary output
                    aux_landmarks = torch.zeros(
                        batch_size, num_queries, self.num_landmarks * 2,
                        device=aux_out['pred_logits'].device,
                        dtype=aux_out['pred_logits'].dtype
                    )
                    aux_out['pred_landmarks'] = aux_landmarks
        
        # Add landmarks to encoder output (if exists)
        if 'enc_aux_outputs' in outputs:
            for enc_out in outputs['enc_aux_outputs']:
                if 'pred_logits' in enc_out:
                    enc_landmarks = torch.zeros(
                        batch_size, num_queries, self.num_landmarks * 2,
                        device=enc_out['pred_logits'].device,
                        dtype=enc_out['pred_logits'].dtype
                    )
                    enc_out['pred_landmarks'] = enc_landmarks
        
        return outputs
