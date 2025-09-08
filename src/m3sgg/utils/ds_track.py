# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import, division, print_function

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def box_xyxy_to_xywh(x):
    """Convert bounding box from xyxy format to xywh format.

    Converts bounding box coordinates from (x0, y0, x1, y1) format to
    (x, y, width, height) format.

    :param x: Bounding box tensor in xyxy format
    :type x: torch.Tensor
    :return: Bounding box tensor in xywh format
    :rtype: torch.Tensor
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [x0, y0, x1 - x0, y1 - y0]
    return torch.stack(b, dim=-1)


def get_sequence(entry, gt_annotation, shape, task="sgcls"):
    """Get sequence information for scene graph generation tasks.

    Processes detection results and ground truth annotations to prepare
    sequence data for different scene graph generation tasks.

    :param entry: Detection results containing bboxes and distributions
    :type entry: dict
    :param gt_annotation: Ground truth annotations
    :type gt_annotation: list
    :param shape: Image shape information
    :type shape: tuple
    :param task: Scene graph generation task type, defaults to "sgcls"
    :type task: str, optional
    :return: None (modifies entry in-place)
    :rtype: None
    """
    if task == "predcls":
        pass

    if task == "sgdet" or task == "sgcls":
        indices = [[]]
        # indices[0] store single-element sequence, to save memory
        pred_labels = torch.argmax(entry["distribution"], 1)
        for i in pred_labels.unique():
            index = torch.where(pred_labels == i)[0]
            if len(index) == 1:
                indices[0].append(index)
            else:
                indices.append(index)
        if len(indices[0]) > 0:
            indices[0] = torch.cat(indices[0])
        else:
            indices[0] = torch.tensor([])
        entry["indices"] = indices
        return
