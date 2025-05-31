# tools/infer_face_landmarks.py
"""
Inference script for RT-DETR face detection and landmark localization
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
from src.core import YAMLConfig


def draw_face_landmarks(image, boxes, landmarks, scores, threshold=0.5, 
                        landmark_names=['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']):
    """Draw face boxes and landmarks on image
    
    Args:
        image: PIL Image or numpy array
        boxes: Face bounding boxes [N, 4]
        landmarks: Face landmarks [N, num_landmarks, 2]
        scores: Detection scores [N]
        threshold: Score threshold for visualization
        landmark_names: Names for each landmark point
    """
    
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Make a copy to avoid modifying original
    image = image.copy()
    
    # Define colors for different landmarks
    landmark_colors = [
        (255, 0, 0),    # Red for left eye
        (0, 255, 0),    # Green for right eye
        (0, 0, 255),    # Blue for nose
        (255, 255, 0),  # Yellow for left mouth
        (0, 255, 255),  # Cyan for right mouth
    ]
    
    for box, lmks, score in zip(boxes, landmarks, scores):
        if score < threshold:
            continue
        
        # Draw face box
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw landmarks
        for i, (x, y) in enumerate(lmks):
            x, y = int(x), int(y)
            color = landmark_colors[i % len(landmark_colors)]
            
            # Draw landmark point
            cv2.circle(image, (x, y), 3, color, -1)
            cv2.circle(image, (x, y), 4, (255, 255, 255), 1)  # White border
            
            # Add landmark name if provided
            if i < len(landmark_names):
                cv2.putText(image, landmark_names[i], (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw detection score
        cv2.putText(image, f'{score:.2f}', (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw connections between landmarks (optional)
        if len(lmks) >= 5:
            # Connect eyes
            cv2.line(image, tuple(lmks[0].astype(int)), tuple(lmks[1].astype(int)), 
                    (200, 200, 200), 1)
            # Connect mouth corners
            cv2.line(image, tuple(lmks[3].astype(int)), tuple(lmks[4].astype(int)), 
                    (200, 200, 200), 1)
    
    return image


class FaceLandmarkDetector:
    """Face detection and landmark localization using RT-DETR"""
    
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        """Initialize detector
        
        Args:
            config_path: Path to model config file
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device)
        
        # Load config and model
        cfg = YAMLConfig(config_path, resume=checkpoint_path)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        
        # Load model
        cfg.model.load_state_dict(state)
        
        # Create inference model
        self.model = nn.ModuleDict({
            'model': cfg.model.deploy(),
            'postprocessor': cfg.postprocessor.deploy()
        }).to(self.device)
        
        self.model.eval()
        
        # Image preprocessing
        self.transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        
        # Get number of landmarks from config
        self.num_landmarks = cfg.yaml_cfg.get('num_landmarks', 5)
    
    @torch.no_grad()
    def detect(self, image, conf_threshold=0.5):
        """Detect faces and landmarks in image
        
        Args:
            image: PIL Image or numpy array
            conf_threshold: Confidence threshold for detections
            
        Returns:
            boxes: Face bounding boxes [N, 4]
            landmarks: Face landmarks [N, num_landmarks, 2]
            scores: Detection scores [N]
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get original size
        w, h = image.size
        orig_size = torch.tensor([w, h])[None].to(self.device)
        
        # Preprocess image
        img_tensor = self.transforms(image)[None].to(self.device)
        
        # Run inference
        outputs = self.model['model'](img_tensor)
        results = self.model['postprocessor'](outputs, orig_size)
        
        # Extract results
        if isinstance(results, tuple):  # Deploy mode
            labels, boxes, scores, landmarks = results
            # Convert to numpy
            boxes = boxes[0].cpu().numpy()
            scores = scores[0].cpu().numpy()
            landmarks = landmarks[0].cpu().numpy().reshape(-1, self.num_landmarks, 2)
        else:  # Normal mode
            result = results[0]
            boxes = result['boxes'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            landmarks = result['landmarks'].cpu().numpy()
        
        # Filter by confidence threshold
        mask = scores > conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        landmarks = landmarks[mask]
        
        return boxes, landmarks, scores
    
    def detect_batch(self, images, conf_threshold=0.5):
        """Detect faces in batch of images
        
        Args:
            images: List of PIL Images or numpy arrays
            conf_threshold: Confidence threshold
            
        Returns:
            List of (boxes, landmarks, scores) for each image
        """
        results = []
        for image in images:
            boxes, landmarks, scores = self.detect(image, conf_threshold)
            results.append((boxes, landmarks, scores))
        return results


def process_image(detector, image_path, output_path=None, conf_threshold=0.5, show=True):
    """Process a single image"""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"Processing {image_path}...")
    
    # Detect faces and landmarks
    boxes, landmarks, scores = detector.detect(image, conf_threshold)
    print(f"Detected {len(boxes)} faces")
    
    # Visualize results
    vis_image = draw_face_landmarks(image, boxes, landmarks, scores, conf_threshold)
    
    # Save or show result
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"Saved result to {output_path}")
    
    if show:
        cv2.imshow('Face Detection and Landmarks', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return boxes, landmarks, scores


def process_video(detector, video_path, output_path=None, conf_threshold=0.5, show=True):
    """Process video for face detection and landmarks"""
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if output path provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, landmarks, scores = detector.detect(frame_rgb, conf_threshold)
        
        # Draw results
        vis_frame = draw_face_landmarks(frame_rgb, boxes, landmarks, scores, conf_threshold)
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        
        # Add frame info
        cv2.putText(vis_frame, f'Frame: {frame_count} | Faces: {len(boxes)}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if output_path:
            out.write(vis_frame)
        
        if show:
            cv2.imshow('Face Detection and Landmarks', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames")


def process_webcam(detector, conf_threshold=0.5):
    """Real-time face detection from webcam"""
    
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        start_time = cv2.getTickCount()
        boxes, landmarks, scores = detector.detect(frame_rgb, conf_threshold)
        end_time = cv2.getTickCount()
        
        # Calculate FPS
        fps = cv2.getTickFrequency() / (end_time - start_time)
        
        # Draw results
        vis_frame = draw_face_landmarks(frame_rgb, boxes, landmarks, scores, conf_threshold)
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        
        # Add info
        cv2.putText(vis_frame, f'FPS: {fps:.1f} | Faces: {len(boxes)}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection and Landmarks (Webcam)', vis_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            frame_count += 1
            filename = f'webcam_capture_{frame_count}.jpg'
            cv2.imwrite(filename, vis_frame)
            print(f"Saved {filename}")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='RT-DETR Face Detection and Landmark Inference')
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('-r', '--resume', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('-i', '--input', type=str, default=None,
                       help='Input image or video path')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output path for results')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam for real-time detection')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display results')
    
    args = parser.parse_args()
    
    # Initialize detector
    print("Loading model...")
    detector = FaceLandmarkDetector(
        args.config, 
        args.resume,
        device=args.device
    )
    print(f"Model loaded with {detector.num_landmarks} landmarks")
    
    # Process input
    if args.webcam:
        process_webcam(detector, args.conf_threshold)
    elif args.input:
        if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Process image
            process_image(
                detector, 
                args.input, 
                args.output,
                args.conf_threshold,
                show=not args.no_show
            )
        elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Process video
            process_video(
                detector,
                args.input,
                args.output,
                args.conf_threshold,
                show=not args.no_show
            )
        else:
            print(f"Unsupported file format: {args.input}")
    else:
        print("No input provided. Use --input for image/video or --webcam for live detection")


if __name__ == '__main__':
    main()