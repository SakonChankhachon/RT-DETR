# src/zoo/rtdetr/rtdetr_face_postprocessor.py
"""Post-processor for face detection and landmark localization"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from src.core import register
from .rtdetr_postprocessor import RTDETRPostProcessor


@register
class RTDETRFacePostProcessor(RTDETRPostProcessor):
    """Post-processor for face detection and landmarks"""
    
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'num_landmarks']
    
    def __init__(self, 
                 num_classes=1, 
                 use_focal_loss=True, 
                 num_top_queries=300,
                 num_landmarks=5,
                 remap_mscoco_category=False):
        super().__init__(
            num_classes=num_classes,
            use_focal_loss=use_focal_loss,
            num_top_queries=num_top_queries,
            remap_mscoco_category=remap_mscoco_category
        )
        self.num_landmarks = num_landmarks
    
    def forward(self, outputs, orig_target_sizes):
        """Process model outputs to final predictions"""
        
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        landmarks = outputs['pred_landmarks']
        
        # Convert box format
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        
        # Process landmarks
        landmarks_pred = landmarks.clone()
        # Reshape landmarks to [batch, num_queries, num_landmarks, 2]
        landmarks_pred = landmarks_pred.reshape(
            landmarks_pred.shape[0], 
            landmarks_pred.shape[1], 
            self.num_landmarks, 
            2
        )
        
        # Denormalize landmarks (they are in [0,1] range)
        # landmarks are normalized, so multiply by image size
        for i in range(landmarks_pred.shape[0]):
            landmarks_pred[i, :, :, 0] *= orig_target_sizes[i, 0]  # x coordinates
            landmarks_pred[i, :, :, 1] *= orig_target_sizes[i, 1]  # y coordinates
        
        # Flatten landmarks back to [batch, num_queries, num_landmarks * 2]
        landmarks_pred = landmarks_pred.reshape(
            landmarks_pred.shape[0],
            landmarks_pred.shape[1],
            -1
        )
        
        # Get scores and labels
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            landmarks = landmarks_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, landmarks_pred.shape[-1]))
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            landmarks = landmarks_pred
            
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
                landmarks = torch.gather(landmarks, dim=1, index=index.unsqueeze(-1).tile(1, 1, landmarks.shape[-1]))
        
        # For deployment mode
        if self.deploy_mode:
            return labels, boxes, scores, landmarks
        
        # Create results
        results = []
        for lab, box, sco, lmk in zip(labels, boxes, scores, landmarks):
            result = dict(
                labels=lab,
                boxes=box,
                scores=sco,
                landmarks=lmk.reshape(-1, self.num_landmarks, 2)  # Reshape for easier use
            )
            results.append(result)
        
        return results
    
    def deploy(self):
        """Set deploy mode"""
        self.eval()
        self.deploy_mode = True
        return self