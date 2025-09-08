import numpy as np
import torch
from torch.nn import functional as F

from m3sgg.utils.fpn.box_intersections_cpu.bbox import (
    bbox_intersections as bbox_intersections_np,
)
from m3sgg.utils.fpn.box_intersections_cpu.bbox import bbox_overlaps as bbox_overlaps_np


def bbox_loss(prior_boxes, deltas, gt_boxes, eps=1e-4, scale_before=1):
    """Compute bounding box regression loss.

    Computes smooth L1 loss for predicting ground truth boxes from prior boxes
    using delta transformations.

    :param prior_boxes: Prior bounding boxes of shape [num_boxes, 4] (x1, y1, x2, y2)
    :type prior_boxes: torch.Tensor
    :param deltas: Predicted box deltas of shape [num_boxes, 4] (tx, ty, th, tw)
    :type deltas: torch.Tensor
    :param gt_boxes: Ground truth boxes of shape [num_boxes, 4] (x1, y1, x2, y2)
    :type gt_boxes: torch.Tensor
    :param eps: Small epsilon value for numerical stability, defaults to 1e-4
    :type eps: float, optional
    :param scale_before: Scaling factor, defaults to 1
    :type scale_before: int, optional
    :return: Computed bounding box loss
    :rtype: torch.Tensor
    """
    prior_centers = center_size(prior_boxes)  # (cx, cy, w, h)
    gt_centers = center_size(gt_boxes)  # (cx, cy, w, h)

    center_targets = (gt_centers[:, :2] - prior_centers[:, :2]) / prior_centers[:, 2:]
    size_targets = torch.log(gt_centers[:, 2:]) - torch.log(prior_centers[:, 2:])
    all_targets = torch.cat((center_targets, size_targets), 1)

    loss = F.smooth_l1_loss(deltas, all_targets, size_average=False) / (
        eps + prior_centers.size(0)
    )

    return loss


def bbox_preds(boxes, deltas):
    """Convert predicted deltas to bounding box coordinates.

    Transforms predicted deltas along with prior boxes into
    (x1, y1, x2, y2) coordinate representation.

    :param boxes: Prior boxes in (x1, y1, x2, y2) format
    :type boxes: torch.Tensor
    :param deltas: Predicted offsets (tx, ty, tw, th)
    :type deltas: torch.Tensor
    :return: Transformed bounding boxes
    :rtype: torch.Tensor
    """

    if boxes.size(0) == 0:
        return boxes
    prior_centers = center_size(boxes)

    xys = prior_centers[:, :2] + prior_centers[:, 2:] * deltas[:, :2]

    whs = torch.exp(deltas[:, 2:]) * prior_centers[:, 2:]

    return point_form(torch.cat((xys, whs), 1))


def center_size(boxes):
    """Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0

    if isinstance(boxes, np.ndarray):
        return np.column_stack((boxes[:, :2] + 0.5 * wh, wh))
    return torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)


def point_form(boxes):
    """Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    if isinstance(boxes, np.ndarray):
        return np.column_stack(
            (
                boxes[:, :2] - 0.5 * boxes[:, 2:],
                boxes[:, :2] + 0.5 * (boxes[:, 2:] - 2.0),
            )
        )
    return torch.cat(
        (boxes[:, :2] - 0.5 * boxes[:, 2:], boxes[:, :2] + 0.5 * (boxes[:, 2:] - 2.0)),
        1,
    )  # xmax, ymax


###########################################################################
### Torch Utils, creds to Max de Groot
###########################################################################


def bbox_intersections(box_a, box_b):
    """We resize both tensors to [A,B,2.0] without new malloc:
    [A,2.0] -> [A,ĺeftright,2.0] -> [A,B,2.0]
    [B,2.0] -> [ĺeftright,B,2.0] -> [A,B,2.0]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    if isinstance(box_a, np.ndarray):
        assert isinstance(box_b, np.ndarray)
        return bbox_intersections_np(box_a, box_b)
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        box_b[:, 2:].unsqueeze(0).expand(A, B, 2),
    )
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2),
        box_b[:, :2].unsqueeze(0).expand(A, B, 2),
    )
    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def bbox_overlaps(box_a, box_b):
    """Compute Jaccard overlap (IoU) between two sets of bounding boxes.

    Calculates intersection over union (IoU) for all pairs of boxes between
    two sets. IoU = A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)

    :param box_a: First set of bounding boxes, shape [num_objects, 4]
    :type box_a: torch.Tensor
    :param box_b: Second set of bounding boxes, shape [num_priors, 4]
    :type box_b: torch.Tensor
    :return: Jaccard overlap matrix, shape [box_a.size(0), box_b.size(0)]
    :rtype: torch.Tensor
    """
    if isinstance(box_a, np.ndarray):
        assert isinstance(box_b, np.ndarray)
        return bbox_overlaps_np(box_a, box_b)

    inter = bbox_intersections(box_a, box_b)
    area_a = (
        ((box_a[:, 2] - box_a[:, 0] + 1.0) * (box_a[:, 3] - box_a[:, 1] + 1.0))
        .unsqueeze(1)
        .expand_as(inter)
    )  # [A,B]
    area_b = (
        ((box_b[:, 2] - box_b[:, 0] + 1.0) * (box_b[:, 3] - box_b[:, 1] + 1.0))
        .unsqueeze(0)
        .expand_as(inter)
    )  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def nms_overlaps(boxes):
    """get overlaps for each channel"""
    assert boxes.dim() == 3
    N = boxes.size(0)
    nc = boxes.size(1)
    max_xy = torch.min(
        boxes[:, None, :, 2:].expand(N, N, nc, 2),
        boxes[None, :, :, 2:].expand(N, N, nc, 2),
    )

    min_xy = torch.max(
        boxes[:, None, :, :2].expand(N, N, nc, 2),
        boxes[None, :, :, :2].expand(N, N, nc, 2),
    )

    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)

    # n, n, 151
    inters = inter[:, :, :, 0] * inter[:, :, :, 1]
    boxes_flat = boxes.view(-1, 4)
    areas_flat = (boxes_flat[:, 2] - boxes_flat[:, 0] + 1.0) * (
        boxes_flat[:, 3] - boxes_flat[:, 1] + 1.0
    )
    areas = areas_flat.view(boxes.size(0), boxes.size(1))
    union = -inters + areas[None] + areas[:, None]
    return inters / union
