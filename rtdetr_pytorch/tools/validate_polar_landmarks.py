#!/usr/bin/env python
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
    
    print("\n📊 Model Performance Analysis:")
    
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
