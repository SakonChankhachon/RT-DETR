#!/bin/bash
# setup_face_detection.sh
# Script to set up RT-DETR for face detection and landmarks

echo "Setting up RT-DETR for Face Detection and Landmarks..."

# Create necessary directories
echo "Creating directories..."
mkdir -p src/data
mkdir -p src/zoo/rtdetr
mkdir -p configs/rtdetr/include
mkdir -p configs/dataset

# Copy compatibility files
echo "Setting up compatibility layers..."

# Create the datapoints compatibility file
cat > src/data/datapoints_compat.py << 'EOF'
"""
Compatibility layer for torchvision datapoints API
Works with both old and new versions of torchvision
"""

import torch
import torchvision
from packaging import version

# Check torchvision version
TORCHVISION_VERSION = version.parse(torchvision.__version__)
HAS_DATAPOINTS = TORCHVISION_VERSION >= version.parse("0.15.0")

if HAS_DATAPOINTS:
    from torchvision import datapoints
    from torchvision.datapoints import BoundingBoxFormat
else:
    # Create compatibility classes for older torchvision versions
    class BoundingBoxFormat:
        XYXY = 'xyxy'
        XYWH = 'xywh' 
        CXCYWH = 'cxcywh'
    
    class BoundingBox:
        """Compatibility class for BoundingBox datapoint"""
        def __init__(self, data, format, spatial_size):
            self.data = data
            self.format = format
            self.spatial_size = spatial_size
            self._tensor = data
        
        def __getattr__(self, name):
            return getattr(self._tensor, name)
        
        def __getitem__(self, key):
            return self._tensor[key]
        
        def __setitem__(self, key, value):
            self._tensor[key] = value
        
        @property
        def shape(self):
            return self._tensor.shape
        
        @property
        def device(self):
            return self._tensor.device
        
        def to(self, *args, **kwargs):
            self._tensor = self._tensor.to(*args, **kwargs)
            return self
    
    class Image:
        """Compatibility class for Image datapoint"""
        def __init__(self, data):
            self.data = data
            self._tensor = data
        
        def __getattr__(self, name):
            return getattr(self._tensor, name)
    
    class Mask:
        """Compatibility class for Mask datapoint"""
        def __init__(self, data):
            self.data = data
            self._tensor = data
        
        def __getattr__(self, name):
            return getattr(self._tensor, name)
    
    # Create a fake datapoints module
    class DatapointsCompat:
        BoundingBox = BoundingBox
        BoundingBoxFormat = BoundingBoxFormat
        Image = Image
        Mask = Mask
    
    datapoints = DatapointsCompat()

# Export the compatibility layer
__all__ = ['datapoints', 'BoundingBoxFormat', 'HAS_DATAPOINTS']
EOF

# Create transforms compatibility
cat > src/data/transforms_compat.py << 'EOF'
"""
Compatibility layer for torchvision transforms v2
"""

import torch
import torchvision
import torchvision.transforms as transforms

# Check if transforms.v2 is available
try:
    import torchvision.transforms.v2 as T
    import torchvision.transforms.v2.functional as F
    HAS_TRANSFORMS_V2 = True
except ImportError:
    # Use v1 transforms
    import torchvision.transforms as T
    import torchvision.transforms.functional as F
    HAS_TRANSFORMS_V2 = False

# Add missing transforms for older versions
if not HAS_TRANSFORMS_V2:
    if not hasattr(T, 'ToImageTensor'):
        T.ToImageTensor = T.ToTensor
    
    if not hasattr(T, 'ConvertDtype'):
        class ConvertDtype:
            def __init__(self, dtype=torch.float32):
                self.dtype = dtype
            def __call__(self, x):
                return x.to(self.dtype) if isinstance(x, torch.Tensor) else x
        T.ConvertDtype = ConvertDtype
    
    if not hasattr(T, 'SanitizeBoundingBox'):
        class SanitizeBoundingBox:
            def __init__(self, min_size=1):
                self.min_size = min_size
            def __call__(self, img, target=None):
                if target is not None:
                    return img, target
                return img
        T.SanitizeBoundingBox = SanitizeBoundingBox
    
    if not hasattr(T, 'RandomPhotometricDistort'):
        T.RandomPhotometricDistort = T.ColorJitter
    
    if not hasattr(T, 'RandomZoomOut'):
        class RandomZoomOut:
            def __init__(self, fill=0, p=0.5):
                self.fill = fill
                self.p = p
            def __call__(self, img, target=None):
                if target is not None:
                    return img, target
                return img
        T.RandomZoomOut = RandomZoomOut

# Export the compatible transforms
__all__ = ['T', 'F', 'HAS_TRANSFORMS_V2']
EOF

echo "Face detection setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy the face detection Python files to their respective locations"
echo "2. Create your face dataset in the required format"
echo "3. Run training with: python tools/train_face_landmarks.py -c configs/rtdetr/rtdetr_r50vd_face_landmark.yml"