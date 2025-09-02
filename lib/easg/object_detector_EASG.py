import copy
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fasterRCNN.lib.model.faster_rcnn.resnet import resnet
from fasterRCNN.lib.model.roi_layers import nms
from fasterRCNN.lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from lib.funcs import assign_relations


class detector(nn.Module):
    """Object detector module for EASG (Efficient and Accurate Scene Graph) generation.

    Implements object detection functionality specifically designed for EASG
    scene graph generation with video-based detection capabilities.

    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """

    def __init__(self, train, object_classes, use_SUPPLY, mode="edgecls"):
        """Initialize the EASG object detector.

        :param train: Whether in training mode
        :type train: bool
        :param object_classes: List of object class names
        :type object_classes: list
        :param use_SUPPLY: Whether to use SUPPLY relations
        :type use_SUPPLY: bool
        :param mode: Detection mode, defaults to "edgecls"
        :type mode: str, optional
        :return: None
        :rtype: None
        """
        super(detector, self).__init__()

        self.is_train = train
        self.use_SUPPLY = use_SUPPLY
        self.object_classes = object_classes
        self.mode = mode

        self.fasterRCNN = resnet(
            classes=self.object_classes, num_layers=101, class_agnostic=False
        )
        self.fasterRCNN.create_architecture()

        # TODO: Load checkpoint if available, otherwise start with random weights
        checkpoint_path = (
            "easg-generation/fasterRCNN/models/res101/easg/faster_rcnn_1_11_952.pth"
        )
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path)
                checkpoint_state_dict = checkpoint["model"]

                # Handle class number mismatch between checkpoint and current model
                current_state_dict = self.fasterRCNN.state_dict()

                # Filter out the classification and bbox prediction layers that have size mismatches
                filtered_checkpoint = {}
                for key, value in checkpoint_state_dict.items():
                    if key in current_state_dict:
                        if current_state_dict[key].shape == value.shape:
                            filtered_checkpoint[key] = value
                        else:
                            print(
                                f"[WARNING] Skipping {key}: checkpoint shape {value.shape} != current shape {current_state_dict[key].shape}"
                            )
                    else:
                        print(f"[WARNING] Skipping {key}: not found in current model")

                # Load the compatible parts of the checkpoint
                missing_keys, unexpected_keys = self.fasterRCNN.load_state_dict(
                    filtered_checkpoint, strict=False
                )
                print(
                    f"[OK] Loaded compatible parts of Faster R-CNN checkpoint from {checkpoint_path}"
                )
                if missing_keys:
                    print(
                        f"  Missing keys (will use random initialization): {missing_keys}"
                    )
                if unexpected_keys:
                    print(f"  Unexpected keys (ignored): {unexpected_keys}")

            except Exception as e:
                print(f"[WARNING] Could not load checkpoint {checkpoint_path}: {e}")
                print("Starting with random weights...")
        else:
            print(f"[WARNING] Checkpoint not found at {checkpoint_path}")
            print("Starting with random weights...")

        self.ROI_Align = copy.deepcopy(self.fasterRCNN.RCNN_roi_align)
        self.RCNN_Head = copy.deepcopy(self.fasterRCNN._head_to_tail)

    def forward(self, im_data, im_info, gt_grounding, im_all):
        # how many bboxes we have
        bbox_num = 0

        im_idx = []  # which frame are the relations belong to
        edge = []

        for i in gt_grounding:
            bbox_num += len(i)
        FINAL_BBOXES = torch.zeros([bbox_num, 5], dtype=torch.float32).cuda(0)
        FINAL_LABELS = torch.zeros([bbox_num], dtype=torch.int64).cuda(0)
        FINAL_LABELS_VERB = torch.zeros([bbox_num], dtype=torch.int64).cuda(0)

        bbox_idx = 0
        for i, j in enumerate(gt_grounding):
            for m in j:
                FINAL_BBOXES[bbox_idx, 1:] = torch.from_numpy(m["bbox"])
                FINAL_BBOXES[bbox_idx, 0] = i
                FINAL_LABELS[bbox_idx] = m["obj"]
                FINAL_LABELS_VERB[bbox_idx] = m["verb"]
                im_idx.append(i)
                edge.append(m["edge"])
                bbox_idx += 1
        im_idx = torch.tensor(im_idx, dtype=torch.float).cuda(0)

        counter = 0
        FINAL_BASE_FEATURES = torch.tensor([]).cuda(0)

        while counter < im_data.shape[0]:
            # compute 10 images in batch and  collect all frames data in the video
            if counter + 10 < im_data.shape[0]:
                inputs_data = im_data[counter : counter + 10]
            else:
                inputs_data = im_data[counter:]
            base_feat = self.fasterRCNN.RCNN_base(inputs_data)
            FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_feat), 0)
            counter += 10

        FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] * im_info[0, 2]
        FINAL_FEATURES = self.fasterRCNN.RCNN_roi_align(
            FINAL_BASE_FEATURES, FINAL_BBOXES
        )
        FINAL_FEATURES = self.fasterRCNN._head_to_tail(FINAL_FEATURES)

        if self.mode == "edgecls":
            FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]

            entry = {
                "boxes": FINAL_BBOXES,
                "labels": FINAL_LABELS,  # here is the groundtruth
                "labels_verb": FINAL_LABELS_VERB,
                "im_idx": im_idx,
                "features": FINAL_FEATURES,
                "edge": edge,
            }

            return entry
        else:
            if self.is_train:
                FINAL_DISTRIBUTIONS = torch.softmax(
                    self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1
                )
                PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)[1]
                PRED_LABELS = PRED_LABELS + 1

                FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]

                entry = {
                    "boxes": FINAL_BBOXES,
                    "labels": FINAL_LABELS,  # here is the groundtruth
                    "labels_verb": FINAL_LABELS_VERB,
                    "distribution": FINAL_DISTRIBUTIONS,
                    "pred_labels": PRED_LABELS,
                    "im_idx": im_idx,
                    "features": FINAL_FEATURES,
                    "edge": edge,
                }

                return entry
            else:
                FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]

                FINAL_DISTRIBUTIONS = torch.softmax(
                    self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1
                )
                PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)[1]
                PRED_LABELS = PRED_LABELS + 1

                entry = {
                    "boxes": FINAL_BBOXES,
                    "labels": FINAL_LABELS,  # here is the groundtruth
                    "labels_verb": FINAL_LABELS_VERB,
                    "distribution": FINAL_DISTRIBUTIONS,
                    "pred_labels": PRED_LABELS,
                    "im_idx": im_idx,
                    "features": FINAL_FEATURES,
                    "edge": edge,
                    "fmaps": FINAL_BASE_FEATURES,
                    "im_info": im_info[0, 2],
                }

                return entry
