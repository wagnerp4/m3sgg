"""
Utilities module for OED model.

This module provides utility functions for box operations and other helpers.
"""

import torch
import torch.nn.functional as F


def box_cxcywh_to_xyxy(x):
    """Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).

    :param x: Boxes in center format
    :type x: torch.Tensor
    :return: Boxes in corner format
    :rtype: torch.Tensor
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert boxes from corner format (x1, y1, x2, y2) to center format (cx, cy, w, h).

    :param x: Boxes in corner format
    :type x: torch.Tensor
    :return: Boxes in center format
    :rtype: torch.Tensor
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """Generalized Intersection over Union between two sets of boxes.

    :param boxes1: First set of boxes
    :type boxes1: torch.Tensor
    :param boxes2: Second set of boxes
    :type boxes2: torch.Tensor
    :return: Generalized IoU matrix
    :rtype: torch.Tensor
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union

    # Enclosing box
    lt_enclosing = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enclosing = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)
    area_enclosing = wh_enclosing[:, :, 0] * wh_enclosing[:, :, 1]

    # Generalized IoU
    giou = iou - (area_enclosing - union) / area_enclosing
    return giou


def box_area(boxes):
    """Compute the area of a set of bounding boxes.

    :param boxes: Bounding boxes in (x1, y1, x2, y2) format
    :type boxes: torch.Tensor
    :return: Areas of the boxes
    :rtype: torch.Tensor
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def nested_tensor_from_tensor_list(tensor_list):
    """Convert a list of tensors to a nested tensor.

    :param tensor_list: List of tensors
    :type tensor_list: list
    :return: Nested tensor
    :rtype: NestedTensor
    """
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


def _max_by_axis(the_list):
    """Find maximum dimensions across a list of shapes.

    :param the_list: List of shapes
    :type the_list: list
    :return: Maximum dimensions
    :rtype: list
    """
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor:
    """Nested tensor wrapper for efficient processing.

    :param object: Base object class
    :type object: class
    """

    def __init__(self, tensors, mask):
        """Initialize the nested tensor.

        :param tensors: Input tensors
        :type tensors: torch.Tensor
        :param mask: Mask tensor
        :type mask: torch.Tensor
        :return: None
        :rtype: None
        """
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        """Decompose the nested tensor.

        :return: Tuple of (tensors, mask)
        :rtype: tuple
        """
        return self.tensors, self.mask

    def __repr__(self):
        """String representation.

        :return: String representation
        :rtype: str
        """
        return str(self.tensors)
