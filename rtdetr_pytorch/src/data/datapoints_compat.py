"""
Compatibility layer for torchvision datapoints API
Works with both old and new versions of torchvision
"""

import torch
import torchvision
from packaging import version

# Check torchvision version
TORCHVISION_VERSION = version.parse(torchvision.__version__)

# Try to import datapoints, but don't fail if it doesn't exist
try:
    from torchvision import datapoints
    from torchvision.datapoints import BoundingBoxFormat
    HAS_DATAPOINTS = True
except ImportError:
    HAS_DATAPOINTS = False
    
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
        
        @property
        def dtype(self):
            return self._tensor.dtype
        
        def to(self, *args, **kwargs):
            self._tensor = self._tensor.to(*args, **kwargs)
            return self
        
        def cpu(self):
            self._tensor = self._tensor.cpu()
            return self
        
        def cuda(self):
            self._tensor = self._tensor.cuda()
            return self
        
        def clone(self):
            return BoundingBox(self._tensor.clone(), self.format, self.spatial_size)
        
        def detach(self):
            return BoundingBox(self._tensor.detach(), self.format, self.spatial_size)
    
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
