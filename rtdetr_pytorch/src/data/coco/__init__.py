# src/data/coco/__init__.py
# Try to import from the compatibility version first
try:
    from .coco_dataset_compat import (
        CocoDetection,
        mscoco_category2label,
        mscoco_label2category,
        mscoco_category2name,
    )
except ImportError:
    # Fall back to original if compat version doesn't exist
    from .coco_dataset import (
        CocoDetection,
        mscoco_category2label,
        mscoco_label2category,
        mscoco_category2name,
    )

from .coco_eval import *
from .coco_utils import get_coco_api_from_dataset