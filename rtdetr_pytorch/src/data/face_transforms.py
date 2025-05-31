import torch
from src.core import register

@register
class ComposeFaceLandmark(torch.nn.Module):
    """Compose transform for face landmarks"""
    
    def __init__(self, ops):
        super().__init__()
        # Import here to avoid circular imports
        from src.core import GLOBAL_CONFIG
        
        self.transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    if name in GLOBAL_CONFIG:
                        transform = GLOBAL_CONFIG[name]['_pymodule'].__dict__[name](**op)
                    else:
                        # Use standard transforms
                        import torchvision.transforms as T
                        transform = getattr(T, name, None)
                        if transform:
                            transform = transform(**op)
                    if transform:
                        self.transforms.append(transform)
                    op['type'] = name
        
        # If no transforms specified, use identity
        if len(self.transforms) == 0:
            self.transforms = [lambda x, t=None: (x, t) if t is not None else x]
    
    def forward(self, img, target=None):
        for t in self.transforms:
            if target is not None:
                img, target = t(img, target)
            else:
                img = t(img)
        
        if target is not None:
            return img, target
        return img

# Also register the standard face transforms
RandomHorizontalFlipWithLandmarks = register(ComposeFaceLandmark)
ResizeWithLandmarks = register(ComposeFaceLandmark)
NormalizeLandmarks = register(ComposeFaceLandmark)
