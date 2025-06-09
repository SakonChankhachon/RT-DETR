#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Face Detection and Landmark Inference Script (Fixed Import Version)
This script is a modified version of simple_face_infer.py with corrected import paths
that match the actual repository structure.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
from pathlib import Path

print("Attempting to import RT-DETR modules...")

# Import face landmark components
try:
    import face_alignment
    print("✅ Face landmark components loaded successfully")
except ImportError:
    print("⚠️ Could not import face_alignment. Install with: pip install face-alignment")
    face_alignment = None

# Try multiple import paths for PResNet based on the actual directory structure
presnet_imported = False
rtdetr_imported = False

# First attempt: Try importing from corrected path (src.nn or src.core)
try:
    # Try src.nn path first (most likely location based on structure)
    from src.nn.backbone.presnet import PResNet
    presnet_imported = True
    print("✅ Imported PResNet from src.nn.backbone")
except ImportError:
    try:
        # Try src.core path next
        from src.core.backbone.presnet import PResNet
        presnet_imported = True
        print("✅ Imported PResNet from src.core.backbone")
    except ImportError:
        print("⚠️ Could not import PResNet from expected locations")

# If PResNet import failed, try alternative paths
if not presnet_imported:
    # Try direct import (in case it's in the main path)
    try:
        from presnet import PResNet
        presnet_imported = True
        print("✅ Imported PResNet directly")
    except ImportError:
        print("⚠️ Could not import PResNet directly")

# Try multiple import paths for RTDETR based on the actual directory structure
try:
    # Correct path: zoo is inside src directory
    from src.zoo.rtdetr.rtdetr import RTDETR
    rtdetr_imported = True
    print("✅ Imported RTDETR from src.zoo.rtdetr")
except ImportError:
    try:
        # Try alternative path
        from src.zoo.rtdetr import RTDETR
        rtdetr_imported = True
        print("✅ Imported RTDETR from src.zoo")
    except ImportError:
        print("⚠️ Could not import RTDETR from src.zoo paths")

# If imports are still failing, raise a clear error
if not presnet_imported or not rtdetr_imported:
    error_message = "Failed to import required modules:"
    if not presnet_imported:
        error_message += "\n- Could not find PResNet in any expected location"
    if not rtdetr_imported:
        error_message += "\n- Could not find RTDETR in any expected location"
    
    print(f"❌ {error_message}")
    print("\nDetailed directory search paths:")
    for path in sys.path:
        print(f"- {path}")
    
    print("\nTROUBLESHOOTING TIPS:")
    print("1. Make sure you're running this script from the rtdetr_pytorch directory")
    print("2. Try setting your PYTHONPATH environment variable:")
    print("   export PYTHONPATH=/path/to/rtdetr_pytorch:$PYTHONPATH")
    print("3. Check that the project structure is intact with src/ directory")
    print("4. Verify that the required modules exist in one of these locations:")
    print("   - src/nn/backbone/presnet.py")
    print("   - src/core/backbone/presnet.py")
    print("   - src/zoo/rtdetr/rtdetr.py")
    sys.exit(1)

def parse_args():
    """Parse command line arguments for face detection inference."""
    parser = argparse.ArgumentParser(description='Run face detection inference on images')
    parser.add_argument('input', type=str, help='Path to input image or directory')
    parser.add_argument('-o', '--output', type=str, default='result.jpg', 
                        help='Path to output image or directory (default: result.jpg)')
    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--nms-thresh', type=float, default=0.5,
                        help='NMS threshold for detections (default: 0.5)')
    parser.add_argument('--landmarks', action='store_true',
                        help='Enable facial landmark detection')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run inference on (default: cuda:0)')
    return parser.parse_args()

def main():
    """Main function for face detection inference."""
    args = parse_args()
    
    # Load model
    print(f"Loading RT-DETR model on {args.device}...")
    model = RTDETR()  # Initialize with appropriate parameters
    
    # Load face alignment model if landmarks are requested
    fa = None
    if args.landmarks and face_alignment:
        print("Loading face alignment model...")
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                                          device=args.device.split(':')[0])
    
    # Process input path
    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        image_paths = [input_path]
        output_path = Path(args.output)
        output_dir = output_path.parent
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each image
    for img_path in image_paths:
        print(f"Processing {img_path}...")
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Perform inference
        # Note: This is a placeholder for the actual inference code
        # You'll need to replace this with the actual model inference
        
        # Save result
        if input_path.is_dir():
            out_path = output_dir / img_path.name
        else:
            out_path = output_path
        
        cv2.imwrite(str(out_path), img)
        print(f"Result saved to {out_path}")

if __name__ == '__main__':
    main()
