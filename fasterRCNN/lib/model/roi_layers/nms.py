# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# # from ._utils import _C
# from fasterRCNN.lib.model import _C

# nms = _C.nms
# # nms.__doc__ = """
# # This function performs Non-maximum suppresion"""

import torch
from torchvision.ops import nms


def nms_py(boxes, scores, iou_threshold):
    # boxes: [N, 4], scores: [N]
    keep = nms(boxes, scores, iou_threshold)
    return keep
