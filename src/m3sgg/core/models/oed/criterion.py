"""
Criterion module for OED model.

This module implements the loss functions for training the OED model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SetCriterionOED(nn.Module):
    """Loss criterion for OED model.

    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """

    def __init__(
        self, num_obj_classes, num_queries, matcher, weight_dict, eos_coef, losses, conf
    ):
        """Initialize the criterion.

        :param num_obj_classes: Number of object classes
        :type num_obj_classes: int
        :param num_queries: Number of queries
        :type num_queries: int
        :param matcher: Hungarian matcher
        :type matcher: object
        :param weight_dict: Dictionary of loss weights
        :type weight_dict: dict
        :param eos_coef: End-of-sequence coefficient
        :type eos_coef: float
        :param losses: List of loss types
        :type losses: list
        :param conf: Configuration object
        :type conf: Config
        :return: None
        :rtype: None
        """
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_attn_classes = getattr(conf, "num_attn_classes", 3)
        self.num_spatial_classes = getattr(conf, "num_spatial_classes", 6)
        self.num_contacting_classes = getattr(conf, "num_contacting_classes", 17)
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        # Empty weight for object classification
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # Focal loss alpha
        self.alpha = conf.alpha

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        """Object classification loss.

        :param outputs: Model outputs
        :type outputs: dict
        :param targets: Ground truth targets
        :type targets: list
        :param indices: Matched indices
        :type indices: list
        :param num_interactions: Number of interactions
        :type num_interactions: int
        :param log: Whether to log errors, defaults to True
        :type log: bool, optional
        :return: Dictionary of losses
        :rtype: dict
        """
        assert "pred_obj_logits" in outputs
        src_logits = outputs["pred_obj_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["obj_labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_obj_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        obj_weights = self.empty_weight
        loss_obj_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, obj_weights
        )
        losses = {"loss_obj_ce": loss_obj_ce}

        if log:
            losses["obj_class_error"] = (
                100 - accuracy(src_logits[idx], target_classes_o)[0]
            )
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        """Object cardinality loss.

        :param outputs: Model outputs
        :type outputs: dict
        :param targets: Ground truth targets
        :type targets: list
        :param indices: Matched indices
        :type indices: list
        :param num_interactions: Number of interactions
        :type num_interactions: int
        :return: Dictionary of losses
        :rtype: dict
        """
        pred_logits = outputs["pred_obj_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["obj_labels"]) for v in targets], device=device
        )
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"obj_cardinality_error": card_err}
        return losses

    def loss_relation_labels(self, outputs, targets, indices, num_interactions):
        """Relation classification loss.

        :param outputs: Model outputs
        :type outputs: dict
        :param targets: Ground truth targets
        :type targets: list
        :param indices: Matched indices
        :type indices: list
        :param num_interactions: Number of interactions
        :type num_interactions: int
        :return: Dictionary of losses
        :rtype: dict
        """
        num_attn_rel, num_spatial_rel, num_contacting_rel = 3, 6, 17

        attn_logits = outputs["pred_attn_logits"].reshape(-1, num_attn_rel + 1)
        spatial_logits = outputs["pred_spatial_logits"].reshape(-1, num_spatial_rel)
        contacting_logits = outputs["pred_contacting_logits"].reshape(
            -1, num_contacting_rel
        )

        attn_probs = attn_logits.softmax(dim=-1)
        spatial_probs = spatial_logits.sigmoid()
        contacting_probs = contacting_logits.sigmoid()

        idx = self._get_src_permutation_idx(indices)
        idx = (idx[0].to(attn_logits.device), idx[1].to(attn_logits.device))

        target_attn_classes_o = torch.cat(
            [t["attn_labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_spatial_classes_o = torch.cat(
            [t["spatial_labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_contacting_classes_o = torch.cat(
            [t["contacting_labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        # Select matched queries for loss computation
        sel_idx = idx[0] * outputs["pred_attn_logits"].shape[1] + idx[1]
        attn_logits = attn_logits[sel_idx]
        spatial_probs = spatial_probs[sel_idx]
        contacting_probs = contacting_probs[sel_idx]

        target_attn_classes = target_attn_classes_o
        target_spatial_classes = target_spatial_classes_o
        target_contacting_classes = target_contacting_classes_o

        # Attention loss (cross-entropy)
        target_attn_labels = torch.where(target_attn_classes)[1]
        loss_attn_ce = F.cross_entropy(attn_logits, target_attn_labels)

        # Spatial and contacting losses (focal loss)
        loss_spatial_ce = self._neg_loss(
            spatial_probs, target_spatial_classes, alpha=self.alpha
        )
        loss_contacting_ce = self._neg_loss(
            contacting_probs, target_contacting_classes, alpha=self.alpha
        )

        losses = {
            "loss_attn_ce": loss_attn_ce,
            "loss_spatial_ce": loss_spatial_ce,
            "loss_contacting_ce": loss_contacting_ce,
        }

        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        """Bounding box loss.

        :param outputs: Model outputs
        :type outputs: dict
        :param targets: Ground truth targets
        :type targets: list
        :param indices: Matched indices
        :type indices: list
        :param num_interactions: Number of interactions
        :type num_interactions: int
        :return: Dictionary of losses
        :rtype: dict
        """
        assert "pred_sub_boxes" in outputs and "pred_obj_boxes" in outputs

        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs["pred_sub_boxes"][idx]
        src_obj_boxes = outputs["pred_obj_boxes"][idx]

        target_sub_boxes = torch.cat(
            [t["sub_boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        target_obj_boxes = torch.cat(
            [t["obj_boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses["loss_sub_bbox"] = src_sub_boxes.sum()
            losses["loss_obj_bbox"] = src_obj_boxes.sum()
            losses["loss_sub_giou"] = src_sub_boxes.sum()
            losses["loss_obj_giou"] = src_obj_boxes.sum()
        else:
            # L1 loss
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction="none")
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction="none")

            losses["loss_sub_bbox"] = loss_sub_bbox.sum() / num_interactions
            losses["loss_obj_bbox"] = (
                loss_obj_bbox * exist_obj_boxes.unsqueeze(1)
            ).sum() / (exist_obj_boxes.sum() + 1e-4)

            # GIoU loss
            from .utils import box_cxcywh_to_xyxy, generalized_box_iou

            loss_sub_giou = 1 - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_sub_boxes),
                    box_cxcywh_to_xyxy(target_sub_boxes),
                )
            )
            loss_obj_giou = 1 - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_obj_boxes),
                    box_cxcywh_to_xyxy(target_obj_boxes),
                )
            )

            losses["loss_sub_giou"] = loss_sub_giou.sum() / num_interactions
            losses["loss_obj_giou"] = (loss_obj_giou * exist_obj_boxes).sum() / (
                exist_obj_boxes.sum() + 1e-4
            )

        return losses

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        """Negative log likelihood loss.

        :param pred: Predictions
        :type pred: torch.Tensor
        :param gt: Ground truth
        :type gt: torch.Tensor
        :param weights: Loss weights, defaults to None
        :type weights: torch.Tensor, optional
        :param alpha: Focal loss alpha, defaults to 0.25
        :type alpha: float, optional
        :return: Loss value
        :rtype: torch.Tensor
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def _get_src_permutation_idx(self, indices):
        """Get source permutation indices.

        :param indices: Matched indices
        :type indices: list
        :return: Tuple of (batch_idx, src_idx)
        :rtype: tuple
        """
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """Get target permutation indices.

        :param indices: Matched indices
        :type indices: list
        :return: Tuple of (batch_idx, tgt_idx)
        :rtype: tuple
        """
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        """Get loss function.

        :param loss: Loss type
        :type loss: str
        :param outputs: Model outputs
        :type outputs: dict
        :param targets: Ground truth targets
        :type targets: list
        :param indices: Matched indices
        :type indices: list
        :param num: Number of interactions
        :type num: int
        :return: Loss dictionary
        :rtype: dict
        """
        loss_map = {
            "obj_labels": self.loss_obj_labels,
            "obj_cardinality": self.loss_obj_cardinality,
            "relation_labels": self.loss_relation_labels,
            "sub_obj_boxes": self.loss_sub_obj_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        """Forward pass for loss computation.

        :param outputs: Model outputs
        :type outputs: dict
        :param targets: Ground truth targets
        :type targets: list
        :return: Dictionary of losses
        :rtype: dict
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t["obj_labels"]) for t in targets)
        num_interactions = torch.as_tensor(
            [num_interactions],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )

        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_interactions)
            )

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "obj_labels":
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_interactions, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def accuracy(output, target, topk=(1,)):
    """Compute accuracy.

    :param output: Model output
    :type output: torch.Tensor
    :param target: Ground truth target
    :type target: torch.Tensor
    :param topk: Top-k accuracy, defaults to (1,)
    :type topk: tuple, optional
    :return: Tuple of accuracies
    :rtype: tuple
    """
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
