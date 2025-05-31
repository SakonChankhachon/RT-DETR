# tools/prepare_face_dataset.py
"""
Script to prepare face detection datasets for RT-DETR training
Supports: WIDER Face, WFLW, 300W, AFLW, etc.
"""

import os
import json
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import cv2


def prepare_widerface(data_root, output_dir):
    """Convert WIDER Face dataset to RT-DETR format"""
    
    for split in ['train', 'val']:
        print(f"Processing WIDER Face {split} set...")
        
        # Read annotation file
        ann_file = os.path.join(data_root, 'wider_face_split', f'wider_face_{split}_bbx_gt.txt')
        
        annotations = {}
        with open(ann_file, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                # Image path
                img_name = lines[i].strip()
                i += 1
                
                # Number of faces
                num_faces = int(lines[i].strip())
                i += 1
                
                boxes = []
                for j in range(num_faces):
                    # bbox format: x y w h blur expression illumination invalid occlusion pose
                    parts = lines[i + j].strip().split()
                    x, y, w, h = map(int, parts[:4])
                    
                    # Skip invalid boxes
                    if w > 0 and h > 0:
                        # Convert to x1,y1,x2,y2
                        boxes.append([x, y, x + w, y + h])
                
                i += num_faces
                
                if len(boxes) > 0:
                    image_id = img_name.replace('/', '_').replace('.jpg', '')
                    annotations[image_id] = {
                        'filename': img_name,
                        'boxes': boxes,
                        'landmarks': []  # WIDER Face doesn't have landmarks
                    }
        
        # Add dummy landmarks (center of face)
        for img_id, ann in annotations.items():
            landmarks = []
            for box in ann['boxes']:
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                
                # Create 5 dummy landmarks
                landmark = [
                    cx - w * 0.2, cy - h * 0.2,  # left eye
                    cx + w * 0.2, cy - h * 0.2,  # right eye
                    cx, cy,                       # nose
                    cx - w * 0.15, cy + h * 0.2, # left mouth
                    cx + w * 0.15, cy + h * 0.2  # right mouth
                ]
                landmarks.append(landmark)
            ann['landmarks'] = landmarks
        
        # Save annotations
        output_file = os.path.join(output_dir, f'{split}/annotations.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Saved {len(annotations)} annotations to {output_file}")


def prepare_wflw(data_root, output_dir):
    """Convert WFLW dataset (98 landmarks) to RT-DETR format"""
    
    for split in ['train', 'test']:
        print(f"Processing WFLW {split} set...")
        
        list_file = os.path.join(data_root, 'WFLW_annotations', 'list_98pt_rect_attr_train_test', 
                                f'list_98pt_rect_attr_{split}.txt')
        
        annotations = {}
        with open(list_file, 'r') as f:
            lines = f.readlines()
            
        for line in tqdm(lines):
            parts = line.strip().split()
            
            # Parse 98 landmarks (196 values)
            landmarks_98 = list(map(float, parts[:196]))
            landmarks_98 = np.array(landmarks_98).reshape(98, 2)
            
            # Convert to 5-point landmarks (approximate)
            # Left eye: average of points 60-67
            # Right eye: average of points 68-75
            # Nose: point 54
            # Left mouth: point 76
            # Right mouth: point 82
            left_eye = landmarks_98[60:68].mean(axis=0)
            right_eye = landmarks_98[68:76].mean(axis=0)
            nose = landmarks_98[54]
            left_mouth = landmarks_98[76]
            right_mouth = landmarks_98[82]
            
            landmarks_5 = np.array([left_eye, right_eye, nose, left_mouth, right_mouth])
            landmarks_5 = landmarks_5.flatten().tolist()
            
            # Parse bounding box
            x1, y1, x2, y2 = map(float, parts[196:200])
            
            # Image path
            img_path = parts[-1]
            
            image_id = img_path.replace('/', '_').replace('.jpg', '').replace('.png', '')
            annotations[image_id] = {
                'filename': img_path,
                'boxes': [[x1, y1, x2, y2]],
                'landmarks': [landmarks_5]
            }
        
        # Save annotations
        output_file = os.path.join(output_dir, f'{split}/annotations.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Saved {len(annotations)} annotations to {output_file}")


def prepare_300w(data_root, output_dir):
    """Convert 300W dataset (68 landmarks) to RT-DETR format"""
    
    import scipy.io as sio
    
    for split in ['train', 'test']:
        print(f"Processing 300W {split} set...")
        
        if split == 'train':
            folders = ['01_Indoor', '02_Outdoor']
        else:
            folders = ['01_Challenging', '02_Common']
        
        annotations = {}
        
        for folder in folders:
            folder_path = os.path.join(data_root, folder)
            if not os.path.exists(folder_path):
                continue
                
            # List all .pts files
            pts_files = [f for f in os.listdir(folder_path) if f.endswith('.pts')]
            
            for pts_file in tqdm(pts_files):
                # Parse landmarks
                landmarks_68 = []
                with open(os.path.join(folder_path, pts_file), 'r') as f:
                    lines = f.readlines()
                    # Skip header
                    for i in range(3, 71):  # Lines 3-70 contain the 68 points
                        x, y = map(float, lines[i].strip().split())
                        landmarks_68.extend([x, y])
                
                # Convert to 5-point landmarks
                landmarks_68 = np.array(landmarks_68).reshape(68, 2)
                
                # Approximate 5-point landmarks
                left_eye = landmarks_68[36:42].mean(axis=0)
                right_eye = landmarks_68[42:48].mean(axis=0)
                nose = landmarks_68[30]
                left_mouth = landmarks_68[48]
                right_mouth = landmarks_68[54]
                
                landmarks_5 = np.array([left_eye, right_eye, nose, left_mouth, right_mouth])
                landmarks_5 = landmarks_5.flatten().tolist()
                
                # Get bounding box from landmarks
                x_coords = landmarks_68[:, 0]
                y_coords = landmarks_68[:, 1]
                x1, y1 = x_coords.min(), y_coords.min()
                x2, y2 = x_coords.max(), y_coords.max()
                
                # Add margin
                w, h = x2 - x1, y2 - y1
                margin = 0.2
                x1 -= w * margin
                x2 += w * margin
                y1 -= h * margin
                y2 += h * margin
                
                # Image filename
                img_name = pts_file.replace('.pts', '.jpg')
                if not os.path.exists(os.path.join(folder_path, img_name)):
                    img_name = pts_file.replace('.pts', '.png')
                
                image_id = f"{folder}_{img_name}".replace('.jpg', '').replace('.png', '')
                annotations[image_id] = {
                    'filename': os.path.join(folder, img_name),
                    'boxes': [[x1, y1, x2, y2]],
                    'landmarks': [landmarks_5]
                }
        
        # Save annotations
        output_file = os.path.join(output_dir, f'{split}/annotations.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Saved {len(annotations)} annotations to {output_file}")


def visualize_annotations(ann_file, img_dir, num_samples=5):
    """Visualize annotations to verify correctness"""
    
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Randomly sample some images
    import random
    sample_ids = random.sample(list(annotations.keys()), min(num_samples, len(annotations)))
    
    for img_id in sample_ids:
        ann = annotations[img_id]
        img_path = os.path.join(img_dir, ann['filename'])
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Draw boxes
        for box in ann['boxes']:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw landmarks
        for landmarks in ann['landmarks']:
            landmarks = np.array(landmarks).reshape(-1, 2)
            for i, (x, y) in enumerate(landmarks):
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.putText(img, str(i), (int(x)+5, int(y)+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Show image
        cv2.imshow(f'Sample: {img_id}', img)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['widerface', 'wflw', '300w', 'custom'],
                       help='Dataset type')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize some samples after processing')
    
    args = parser.parse_args()
    
    if args.dataset == 'widerface':
        prepare_widerface(args.data_root, args.output_dir)
    elif args.dataset == 'wflw':
        prepare_wflw(args.data_root, args.output_dir)
    elif args.dataset == '300w':
        prepare_300w(args.data_root, args.output_dir)
    else:
        print(f"Dataset {args.dataset} not implemented yet")
    
    if args.visualize:
        ann_file = os.path.join(args.output_dir, 'train/annotations.json')
        img_dir = args.data_root
        visualize_annotations(ann_file, img_dir)


if __name__ == '__main__':
    main()