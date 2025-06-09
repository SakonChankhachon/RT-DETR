# tools/evaluate_face_model.py
"""
Evaluate face detection model on validation dataset
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import src.misc.dist as dist
from src.core import YAMLConfig
from src.data import get_coco_api_from_dataset, CocoEvaluator
from src.solver.det_engine import evaluate


def compute_landmark_metrics(predictions, targets):
    """Compute face landmark specific metrics"""
    
    metrics = defaultdict(list)
    matched_count = 0
    total_gt = 0
    
    for pred, target in zip(predictions, targets):
        if 'landmarks' not in target or 'landmarks' not in pred:
            continue
        
        # Get dimensions
        orig_w, orig_h = target['orig_size'].tolist()
        
        # Convert normalized cxcywh to pixel xyxy for GT boxes
        gt_boxes_cxcywh = target['boxes'].cpu().numpy()
        gt_boxes_xyxy = np.zeros_like(gt_boxes_cxcywh)
        gt_boxes_xyxy[:, 0] = (gt_boxes_cxcywh[:, 0] - gt_boxes_cxcywh[:, 2]/2) * orig_w
        gt_boxes_xyxy[:, 1] = (gt_boxes_cxcywh[:, 1] - gt_boxes_cxcywh[:, 3]/2) * orig_h
        gt_boxes_xyxy[:, 2] = (gt_boxes_cxcywh[:, 0] + gt_boxes_cxcywh[:, 2]/2) * orig_w
        gt_boxes_xyxy[:, 3] = (gt_boxes_cxcywh[:, 1] + gt_boxes_cxcywh[:, 3]/2) * orig_h
        
        # Predicted boxes are already in pixel xyxy
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_landmarks = pred['landmarks'].cpu().numpy()
        
        # GT landmarks
        gt_landmarks = target['landmarks'].cpu().numpy()
        
        total_gt += len(gt_boxes_xyxy)
        
        # Match predictions to ground truth
        for gt_idx, gt_box in enumerate(gt_boxes_xyxy):
            if len(pred_boxes) == 0:
                continue
                
            # Calculate IoU
            ious = compute_iou_vectorized(pred_boxes, gt_box[None, :])
            
            # Find best match
            best_idx = ious.argmax()
            best_iou = ious[best_idx]
            
            if best_iou > 0.5 and pred_scores[best_idx] > 0.3:
                matched_count += 1
                
                # Compute landmark errors
                gt_lmks = gt_landmarks[gt_idx].reshape(-1, 2)
                pred_lmks = pred_landmarks[best_idx].reshape(-1, 2)
                
                # Convert GT landmarks to pixel coordinates
                gt_lmks[:, 0] *= orig_w
                gt_lmks[:, 1] *= orig_h
                
                # Compute errors
                errors = np.linalg.norm(gt_lmks - pred_lmks, axis=1)
                
                # Store individual landmark errors
                for i, error in enumerate(errors):
                    metrics[f'landmark_{i}_error'].append(error)
                
                # Mean error
                metrics['mean_landmark_error'].append(errors.mean())
                
                # Normalized Mean Error (NME) using inter-ocular distance
                if len(gt_lmks) >= 2:
                    iod = np.linalg.norm(gt_lmks[0] - gt_lmks[1])
                    if iod > 0:
                        nme = errors.mean() / iod
                        metrics['nme'].append(nme)
    
    # Compute summary statistics
    summary = {
        'detection_recall': matched_count / max(total_gt, 1),
        'matched_faces': matched_count,
        'total_gt_faces': total_gt
    }
    
    # Average metrics
    for key, values in metrics.items():
        if len(values) > 0:
            summary[f'avg_{key}'] = np.mean(values)
            summary[f'std_{key}'] = np.std(values)
    
    return summary


def compute_iou_vectorized(boxes1, boxes2):
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate face detection model')
    parser.add_argument('--config', type=str, 
                       default='configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
    parser.add_argument('--checkpoint', type=str,
                       default='./output/rtdetr_r50vd_face_landmark/checkpoint0099.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    
    args = parser.parse_args()
    
    # Initialize
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load config
    print("Loading configuration...")
    cfg = YAMLConfig(args.config)
    
    # Setup model
    print("Setting up model...")
    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.get('last_epoch', -1)
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        print(f"Warning: No checkpoint found at {args.checkpoint}")
        epoch = -1
    
    model.eval()
    
    # Setup dataloader
    print("Setting up validation dataloader...")
    val_dataloader = dist.warp_loader(
        cfg.val_dataloader,
        shuffle=False
    )
    
    # Get COCO GT
    base_ds = get_coco_api_from_dataset(val_dataloader.dataset)
    
    print(f"Validation dataset: {len(val_dataloader.dataset)} images")
    
    # Run standard COCO evaluation
    print("\nRunning COCO evaluation...")
    test_stats, coco_evaluator = evaluate(
        model, criterion, postprocessor,
        val_dataloader, base_ds, device, None
    )
    
    # Print COCO metrics
    print("\nCOCO Detection Metrics:")
    if 'coco_eval_bbox' in test_stats:
        ap_metrics = test_stats['coco_eval_bbox']
        print(f"  AP@[0.50:0.95]: {ap_metrics[0]:.3f}")
        print(f"  AP@0.50: {ap_metrics[1]:.3f}")
        print(f"  AP@0.75: {ap_metrics[2]:.3f}")
        print(f"  AP (small): {ap_metrics[3]:.3f}")
        print(f"  AP (medium): {ap_metrics[4]:.3f}")
        print(f"  AP (large): {ap_metrics[5]:.3f}")
    
    # Run landmark evaluation
    print("\nComputing landmark metrics...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for samples, targets in tqdm(val_dataloader, desc="Processing"):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(samples)
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_sizes)
            
            all_predictions.extend(results)
            all_targets.extend(targets)
    
    # Compute landmark metrics
    landmark_metrics = compute_landmark_metrics(all_predictions, all_targets)
    
    # Print landmark metrics
    print("\nLandmark Metrics:")
    print(f"  Detection Recall: {landmark_metrics['detection_recall']:.3f}")
    print(f"  Matched Faces: {landmark_metrics['matched_faces']}/{landmark_metrics['total_gt_faces']}")
    
    if 'avg_nme' in landmark_metrics:
        print(f"  Average NME: {landmark_metrics['avg_nme']:.4f} ± {landmark_metrics['std_nme']:.4f}")
    
    if 'avg_mean_landmark_error' in landmark_metrics:
        print(f"  Average Landmark Error: {landmark_metrics['avg_mean_landmark_error']:.2f} ± "
              f"{landmark_metrics['std_mean_landmark_error']:.2f} pixels")
    
    # Per-landmark errors
    print("\n  Per-Landmark Errors (pixels):")
    landmark_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
    for i, name in enumerate(landmark_names):
        if f'avg_landmark_{i}_error' in landmark_metrics:
            avg_err = landmark_metrics[f'avg_landmark_{i}_error']
            std_err = landmark_metrics[f'std_landmark_{i}_error']
            print(f"    {name}: {avg_err:.2f} ± {std_err:.2f}")
    
    # Save results
    results = {
        'epoch': epoch,
        'coco_metrics': test_stats,
        'landmark_metrics': landmark_metrics
    }
    
    output_file = args.checkpoint.replace('.pth', '_eval_results.pth')
    torch.save(results, output_file)
    print(f"\nSaved evaluation results to {output_file}")


if __name__ == '__main__':
    main()