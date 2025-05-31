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
