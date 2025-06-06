# src/data/coco/coco_utils_fixed.py
"""
Fixed COCO utilities that handle normalized cxcywh format correctly

This file should replace src/data/coco/coco_utils.py
"""

import os
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
            
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_normalized_cxcywh_to_coco_api(ds):
    """
    Convert dataset with normalized cxcywh boxes to COCO API format.
    This is needed because COCO API expects boxes in xywh pixel coordinates.
    """
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    
    for img_idx in range(len(ds)):
        # Get target data
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        
        # Get image dimensions
        if hasattr(img, 'shape'):
            # Tensor format: C, H, W
            height, width = img.shape[-2], img.shape[-1]
        else:
            # PIL format
            width, height = img.size
        
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = height
        img_dict["width"] = width
        dataset["images"].append(img_dict)
        
        # Convert boxes from normalized cxcywh to pixel xywh
        boxes = targets["boxes"].clone()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        
        # Check if boxes are normalized (values between 0 and 1)
        if boxes.numel() > 0 and boxes.max() <= 1.0:
            # Convert from normalized cxcywh to pixel coordinates
            cx = boxes[:, 0] * width
            cy = boxes[:, 1] * height
            w = boxes[:, 2] * width
            h = boxes[:, 3] * height
            
            # Convert to xywh format (top-left corner)
            x = cx - w/2
            y = cy - h/2
            
            # Stack to get xywh format
            boxes_xywh = torch.stack([x, y, w, h], dim=1)
            bboxes = boxes_xywh.tolist()
        else:
            # Assume already in correct format
            # Convert cxcywh to xywh
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x = cx - w/2
            y = cy - h/2
            boxes_xywh = torch.stack([x, y, w, h], dim=1)
            bboxes = boxes_xywh.tolist()
        
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            
            # Calculate area if not provided
            if i < len(areas):
                ann["area"] = areas[i] * width * height  # Convert normalized area to pixels
            else:
                ann["area"] = bboxes[i][2] * bboxes[i][3]  # width * height
                
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            
            dataset["annotations"].append(ann)
            ann_id += 1
    
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    """Get COCO API from dataset, handling normalized cxcywh format"""
    # FIXME: This is... awful?
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    
    # Check if it's our custom FaceLandmarkDataset
    if hasattr(dataset, '__class__') and dataset.__class__.__name__ == 'FaceLandmarkDataset':
        return convert_normalized_cxcywh_to_coco_api(dataset)
    elif isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    else:
        # Use the fixed conversion function
        return convert_normalized_cxcywh_to_coco_api(dataset)


# For backward compatibility
convert_to_coco_api = convert_normalized_cxcywh_to_coco_api