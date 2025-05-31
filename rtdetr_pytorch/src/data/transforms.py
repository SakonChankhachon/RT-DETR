
# src/data/transforms.py
"""by lyuwenyu"""

import torch
import torch.nn as nn

import torchvision
torchvision.disable_beta_transforms_warning()

# Try to import v2 transforms
try:
    import torchvision.transforms.v2 as T
    import torchvision.transforms.v2.functional as F
    HAS_V2 = True
except ImportError:
    import torchvision.transforms as T
    import torchvision.transforms.functional as F
    HAS_V2 = False

from PIL import Image
from typing import Any, Dict, List, Optional

from src.core import register, GLOBAL_CONFIG

__all__ = ['Compose',]

# Only register transforms that actually exist
if hasattr(T, 'RandomPhotometricDistort'):
    RandomPhotometricDistort = register(T.RandomPhotometricDistort)
else:
    @register
    class RandomPhotometricDistort(nn.Module):
        def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5):
            super().__init__()
            self.color_jitter = T.ColorJitter(
                brightness=brightness, contrast=contrast,
                saturation=saturation, hue=hue)
            self.p = p
        
        def forward(self, img, target=None):
            if torch.rand(1).item() < self.p:
                img = self.color_jitter(img)
            return (img, target) if target is not None else img

if hasattr(T, 'RandomZoomOut'):
    RandomZoomOut = register(T.RandomZoomOut)
else:
    @register
    class RandomZoomOut(nn.Module):
        def __init__(self, fill=0, side_range=(1.0, 4.0), p=0.5):
            super().__init__()
            self.fill = fill
            self.side_range = side_range
            self.p = p
        
        def forward(self, img, target=None):
            return (img, target) if target is not None else img

# Standard transforms
RandomHorizontalFlip = register(T.RandomHorizontalFlip)
Resize = register(T.Resize)
RandomCrop = register(T.RandomCrop)
Normalize = register(T.Normalize)

# Unified alias for ToImageTensor
try:
    BaseToImageTensor = T.ToImageTensor
except AttributeError:
    BaseToImageTensor = T.ToTensor

@register
class ToImageTensor(BaseToImageTensor):
    """
    Alias for ToImageTensor (v2) or ToTensor (v1).
    """
    pass

# ConvertDtype transform
if hasattr(T, 'ConvertDtype'):
    ConvertDtype = register(T.ConvertDtype)
else:
    @register
    class ConvertDtype(nn.Module):
        def __init__(self, dtype=torch.float32):
            super().__init__()
            self.dtype = dtype
        
        def forward(self, img, target=None):
            if isinstance(img, torch.Tensor):
                img = img.to(self.dtype)
            return (img, target) if target is not None else img

# SanitizeBoundingBox transform
if hasattr(T, 'SanitizeBoundingBox'):
    SanitizeBoundingBox = register(T.SanitizeBoundingBox)
else:
    @register
    class SanitizeBoundingBox(nn.Module):
        def __init__(self, min_size=1):
            super().__init__()
            self.min_size = min_size
        
        def forward(self, img, target=None):
            if target is not None and 'boxes' in target:
                boxes = target['boxes']
                if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
                    w = boxes[:, 2] - boxes[:, 0]
                    h = boxes[:, 3] - boxes[:, 1]
                    keep = (w >= self.min_size) & (h >= self.min_size)
                    if keep.any():
                        target['boxes'] = boxes[keep]
                        for key in ['labels', 'area', 'iscrowd', 'masks', 'landmarks']:
                            if key in target:
                                target[key] = target[key][keep]
            return (img, target) if target is not None else img

@register
class Compose(nn.Module):
    def __init__(self, ops) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transfom = getattr(
                        GLOBAL_CONFIG[name]['_pymodule'], name)(**op)
                    transforms.append(transfom)
                elif isinstance(op, nn.Module):
                    transforms.append(op)
                else:
                    raise ValueError(f'Invalid transform spec: {op}')
        else:
            transforms = [EmptyTransform()]
        
        super().__init__()
        self.transforms = transforms

    def forward(self, img, target=None):
        for t in self.transforms:
            img, target = t(img, target) if target is not None else (t(img), None)
        return (img, target) if target is not None else img

@register
class EmptyTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *inputs):
        return inputs if len(inputs) > 1 else inputs[0]

@register
class PadToSize(nn.Module):
    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        super().__init__()
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        self.spatial_size = spatial_size
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img, target=None):
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            w, h = img.size
        pad_h = max(0, self.spatial_size[0] - h)
        pad_w = max(0, self.spatial_size[1] - w)
        if pad_h > 0 or pad_w > 0:
            padding = [0, 0, pad_w, pad_h]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            if target is not None:
                target['padding'] = torch.tensor(padding)
        return (img, target) if target is not None else img

@register
class RandomIoUCrop(nn.Module):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1,
                 min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2,
                 sampler_options: Optional[List[float]] = None,
                 trials: int = 40, p: float = 1.0):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.sampler_options = sampler_options
        self.trials = trials
        self.p = p

    def forward(self, img, target=None):
        if torch.rand(1) >= self.p:
            return (img, target) if target is not None else img
        if isinstance(img, Image.Image):
            w, h = img.size
        else:
            return (img, target) if target is not None else img
        for _ in range(self.trials):
            scale = torch.rand(1).item() * (self.max_scale - self.min_scale) + self.min_scale
            aspect_ratio = torch.rand(1).item() * (self.max_aspect_ratio - self.min_aspect_ratio) + self.min_aspect_ratio
            crop_h = int(round(h * scale))
            crop_w = int(round(w * scale / aspect_ratio))
            if crop_w <= w and crop_h <= h:
                i = torch.randint(0, h - crop_h + 1, (1,)).item()
                j = torch.randint(0, w - crop_w + 1, (1,)).item()
                img = F.crop(img, i, j, crop_h, crop_w)
                if target is not None and 'boxes' in target:
                    boxes = target['boxes']
                    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - j).clamp(min=0, max=crop_w)
                    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - i).clamp(min=0, max=crop_h)
                    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    target['boxes'] = boxes[keep]
                    for key in ['labels', 'area', 'iscrowd', 'masks', 'landmarks']:
                        if key in target:
                            target[key] = target[key][keep]
                break
        return (img, target) if target is not None else img

@register
class ConvertBox(nn.Module):
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

    def forward(self, img, target):
        if self.out_fmt and 'boxes' in target:
            boxes = target['boxes']
            if self.out_fmt == 'cxcywh':
                x1, y1, x2, y2 = boxes.unbind(-1)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                boxes = torch.stack([cx, cy, w, h], dim=-1)
            target['boxes'] = boxes
        if self.normalize and 'boxes' in target:
            boxes = target['boxes']
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                w, h = img.size
            boxes[:, [0, 2]] /= w
            boxes[:, [1, 3]] /= h
            target['boxes'] = boxes
        return (img, target) if target is not None else img

# Define ComposeFaceLandmark after Compose
@register
class ComposeFaceLandmark(Compose):
    """
    Alias for Compose, specialized for face-landmark pipelines.
    """
    def __init__(self, ops):
        super().__init__(ops)
