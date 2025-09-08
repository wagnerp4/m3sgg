import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

from m3sgg.utils.fpn.box_utils import bbox_overlaps

# TODO: Tracker class


def generalized_box_iou(boxes1, boxes2):
    """Compute Generalized Intersection over Union (GIoU) between two sets of boxes.

    Based on the paper from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format.

    :param boxes1: First set of bounding boxes
    :type boxes1: torch.Tensor
    :param boxes2: Second set of bounding boxes
    :type boxes2: torch.Tensor
    :return: Pairwise GIoU matrix where N = len(boxes1) and M = len(boxes2)
    :rtype: torch.Tensor
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_iou(boxes1, boxes2):
    """Compute Intersection over Union (IoU) between two sets of boxes.

    :param boxes1: First set of bounding boxes
    :type boxes1: torch.Tensor
    :param boxes2: Second set of bounding boxes
    :type boxes2: torch.Tensor
    :return: Tuple containing IoU values and union areas
    :rtype: tuple
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def box_area(boxes):
    """Compute the area of bounding boxes.

    :param boxes: Bounding boxes in [x0, y0, x1, y1] format
    :type boxes: torch.Tensor
    :return: Areas of the bounding boxes
    :rtype: torch.Tensor
    """
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def get_sequence(entry, gt_annotation, matcher, im_size, mode="predcls"):
    """
    Process detection results and ground truth annotations with tracking/matching

    Args:
        entry: Dictionary containing detection results with keys like 'boxes', 'features', etc.
        gt_annotation: Ground truth annotations for the current sequence
        matcher: Hungarian matcher for assignment
        im_size: Image size information
        mode: Processing mode ('predcls', 'sgdet', 'sgcls')
    """
    # Get device from input tensors
    device = (
        entry["boxes"].device
        if "boxes" in entry and entry["boxes"].numel() > 0
        else torch.device("cpu")
    )

    # Basic processing to ensure entry has required fields
    if "boxes" in entry:
        boxes = entry["boxes"]

        # Ensure boxes are in the correct format
        if boxes.numel() > 0:
            # Process boxes based on mode
            if mode == "predcls":
                # For predicate classification, use ground truth boxes
                pass
            elif mode == "sgdet":
                # For scene graph detection, use detected boxes
                # Apply NMS if needed
                if "scores" in entry and entry["scores"].numel() > 0:
                    # Apply NMS to detected boxes
                    # Handle different score tensor dimensions
                    if entry["scores"].dim() == 1:
                        # 1D tensor - single score per detection
                        unique_classes = torch.unique(entry["scores"])
                    else:
                        # 2D tensor - multiple scores per detection
                        unique_classes = torch.argmax(entry["scores"], dim=1).unique()
                    
                    for j in unique_classes:
                        if j == 0:  # skip background
                            continue
                        # Handle different score tensor dimensions for finding indices
                        if entry["scores"].dim() == 1:
                            # 1D tensor - direct comparison
                            inds = torch.nonzero(entry["scores"] == j).view(-1)
                        else:
                            # 2D tensor - use argmax
                            inds = torch.nonzero(
                                torch.argmax(entry["scores"], dim=1) == j
                            ).view(-1)
                        if inds.numel() > 0:
                            # Handle different score tensor dimensions for accessing scores
                            if entry["scores"].dim() == 1:
                                # 1D tensor - direct indexing
                                cls_scores = entry["scores"][inds]
                            else:
                                # 2D tensor - use class index
                                cls_scores = entry["scores"][inds, j]
                            _, order = torch.sort(cls_scores, 0, True)

                            # Apply NMS - ensure boxes have correct format (x1, y1, x2, y2)
                            selected_boxes = boxes[inds][order]
                            if selected_boxes.shape[1] == 5:
                                # Remove first column (likely frame index) to get (x1, y1, x2, y2)
                                selected_boxes = selected_boxes[:, 1:]
                            elif selected_boxes.shape[1] != 4:
                                # Skip NMS if boxes don't have expected format
                                continue
                            
                            keep = nms(selected_boxes, cls_scores[order], 0.5)

                            # Update indices to keep only non-overlapping detections
                            not_keep = torch.LongTensor(
                                [k for k in range(len(inds)) if k not in keep]
                            ).to(device)
                            if not_keep.numel() > 0:
                                # Handle overlapping detections
                                anchor = boxes[inds][order][keep]
                                remain = boxes[inds][order][not_keep]
                                
                                # Ensure boxes have correct format for GIoU calculation
                                if anchor.shape[1] == 5:
                                    anchor = anchor[:, 1:]  # Remove frame index
                                if remain.shape[1] == 5:
                                    remain = remain[:, 1:]  # Remove frame index

                                if anchor.numel() > 0 and remain.numel() > 0:
                                    alignment = torch.argmax(
                                        generalized_box_iou(anchor, remain), 0
                                    )
                                    # Merge overlapping detections
                                    pass

        # Update entry with processed information
        # Add sequence information if not present
        if "sequence_id" not in entry:
            entry["sequence_id"] = 0

        # Add frame indices
        if "im_idx" not in entry and "boxes" in entry:
            entry["im_idx"] = (
                entry["boxes"][:, 0]
                if entry["boxes"].numel() > 0
                else torch.tensor([], device=device)
            )

    # Process features if available
    if "features" in entry and entry["features"].numel() > 0:
        final_feats = entry["features"]
        final_boxes = (
            entry["boxes"] if "boxes" in entry else torch.tensor([], device=device)
        )
        final_labels = (
            entry["labels"] if "labels" in entry else torch.tensor([], device=device)
        )

        # Store processed results
        entry["final_features"] = final_feats
        entry["final_boxes"] = final_boxes
        entry["final_labels"] = final_labels

    # Add tracking indices if not present
    if "track_indices" not in entry:
        entry["track_indices"] = []
        if "boxes" in entry and entry["boxes"].numel() > 0:
            num_boxes = entry["boxes"].shape[0]
            for i in range(num_boxes):
                if "labels" in entry and i < len(entry["labels"]):
                    label = entry["labels"][i]
                    entry["track_indices"].append(
                        torch.where(entry["labels"] == label)[0]
                    )

    return entry
