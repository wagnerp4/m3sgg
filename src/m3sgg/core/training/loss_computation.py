"""
Loss computation methods for different model types.

This module contains the model-specific loss computation logic that was
extracted from the monolithic train.py script.
"""

import torch
from typing import Dict, Any


class LossComputation:
    """Helper class for computing losses for different model types."""
    
    def __init__(self, config, dataset_train, ce_loss, bce_loss, mlm_loss, 
                 ce_loss_obj=None, ce_loss_rel=None, con_loss=None):
        """Initialize loss computation with necessary components.
        
        :param config: Configuration object
        :param dataset_train: Training dataset
        :param ce_loss: Cross entropy loss function
        :param bce_loss: Binary cross entropy loss function
        :param mlm_loss: Multi-label margin loss function
        :param ce_loss_obj: Object-specific cross entropy loss
        :param ce_loss_rel: Relation-specific cross entropy loss
        :param con_loss: Contrastive loss function
        """
        self.config = config
        self.dataset_train = dataset_train
        self.ce_loss = ce_loss
        self.bce_loss = bce_loss
        self.mlm_loss = mlm_loss
        self.ce_loss_obj = ce_loss_obj
        self.ce_loss_rel = ce_loss_rel
        self.con_loss = con_loss

    def compute_stket_losses(self, pred: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute losses for STKET model.
        
        :param pred: Model predictions
        :type pred: Dict[str, Any]
        :return: Dictionary of loss components
        :rtype: Dict[str, torch.Tensor]
        """
        attention_label = (
            torch.tensor(pred["attention_gt"], dtype=torch.long)
            .to(device=pred["attention_distribution"].device)
            .squeeze()
        )
        
        # Ensure attention_label is 1D for CrossEntropyLoss
        if attention_label.dim() > 1:
            attention_label = attention_label.flatten()
        
        if not self.config.bce_loss:
            # multi-label margin loss or adaptive loss
            spatial_label = -torch.ones(
                [len(pred["spatial_gt"]), 6], dtype=torch.long
            ).to(device=pred["attention_distribution"].device)
            contact_label = -torch.ones(
                [len(pred["contact_gt"]), 17], dtype=torch.long
            ).to(device=pred["attention_distribution"].device)
            
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, : len(pred["spatial_gt"][i])] = (
                    torch.tensor(pred["spatial_gt"][i])
                )
                contact_label[i, : len(pred["contact_gt"][i])] = (
                    torch.tensor(pred["contact_gt"][i])
                )
        else:
            # bce loss
            spatial_label = torch.zeros(
                [len(pred["spatial_gt"]), 6], dtype=torch.float32
            ).to(device=pred["attention_distribution"].device)
            contact_label = torch.zeros(
                [len(pred["contact_gt"]), 17], dtype=torch.float32
            ).to(device=pred["attention_distribution"].device)
            
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, pred["spatial_gt"][i]] = 1
                contact_label[i, pred["contact_gt"][i]] = 1

        losses = {}
        if self.config.mode == "sgcls" or self.config.mode == "sgdet":
            losses["object_loss"] = self.ce_loss(
                pred["distribution"], pred["labels"]
            )

        # Spatial encoder losses
        if self.config.enc_layer_num > 0:
            losses["spatial_attention_relation_loss"] = self.ce_loss(
                pred["spatial_attention_distribution"], attention_label
            )
            if not self.config.bce_loss:
                losses["spatial_spatial_relation_loss"] = self.mlm_loss(
                    pred["spatial_spatial_distribution"], spatial_label
                )
                losses["spatial_contact_relation_loss"] = self.mlm_loss(
                    pred["spatial_contact_distribution"], contact_label
                )
            else:
                losses["spatial_spatial_relation_loss"] = self.bce_loss(
                    pred["spatial_spatial_distribution"], spatial_label
                )
                losses["spatial_contact_relation_loss"] = self.bce_loss(
                    pred["spatial_contact_distribution"], contact_label
                )

        # Temporal decoder losses
        if self.config.dec_layer_num > 0:
            losses["temporal_attention_relation_loss"] = self.ce_loss(
                pred["temporal_attention_distribution"], attention_label
            )
            if not self.config.bce_loss:
                losses["temporal_spatial_relation_loss"] = self.mlm_loss(
                    pred["temporal_spatial_distribution"], spatial_label
                )
                losses["temporal_contact_relation_loss"] = self.mlm_loss(
                    pred["temporal_contact_distribution"], contact_label
                )
            else:
                losses["temporal_spatial_relation_loss"] = self.bce_loss(
                    pred["temporal_spatial_distribution"], spatial_label
                )
                losses["temporal_contact_relation_loss"] = self.bce_loss(
                    pred["temporal_contact_distribution"], contact_label
                )

        # Ensemble losses
        if (self.config.enc_layer_num > 0) and (self.config.dec_layer_num > 0):
            losses["ensemble_attention_relation_loss"] = self.ce_loss(
                pred["ensemble_attention_distribution"], attention_label
            )
            if not self.config.bce_loss:
                losses["ensemble_spatial_relation_loss"] = self.mlm_loss(
                    pred["ensemble_spatial_distribution"], spatial_label
                )
                losses["ensemble_contact_relation_loss"] = self.mlm_loss(
                    pred["ensemble_contact_distribution"], contact_label
                )
            else:
                losses["ensemble_spatial_relation_loss"] = self.bce_loss(
                    pred["ensemble_spatial_distribution"], spatial_label
                )
                losses["ensemble_contact_relation_loss"] = self.bce_loss(
                    pred["ensemble_contact_distribution"], contact_label
                )

        # Prior losses if enabled
        if self.config.use_spatial_prior and self.config.spatial_prior_loss:
            losses["spatial_prior_attention_relation_loss"] = self.ce_loss(
                pred["spatial_prior_attention_distribution"],
                attention_label,
            )
            if not self.config.bce_loss:
                losses["spatial_prior_spatial_relation_loss"] = self.mlm_loss(
                    pred["spatial_prior_spatial_distribution"],
                    spatial_label,
                )
                losses["spatial_prior_contact_relation_loss"] = self.mlm_loss(
                    pred["spatial_prior_contact_distribution"],
                    contact_label,
                )
            else:
                losses["spatial_prior_spatial_relation_loss"] = self.bce_loss(
                    pred["spatial_prior_spatial_distribution"],
                    spatial_label,
                )
                losses["spatial_prior_contact_relation_loss"] = self.bce_loss(
                    pred["spatial_prior_contact_distribution"],
                    contact_label,
                )

        if self.config.use_temporal_prior and self.config.temporal_prior_loss:
            losses["temporal_prior_attention_relation_loss"] = self.ce_loss(
                pred["temporal_prior_attention_distribution"],
                attention_label,
            )
            if not self.config.bce_loss:
                losses["temporal_prior_spatial_relation_loss"] = self.mlm_loss(
                    pred["temporal_prior_spatial_distribution"],
                    spatial_label,
                )
                losses["temporal_prior_contact_relation_loss"] = self.mlm_loss(
                    pred["temporal_prior_contact_distribution"],
                    contact_label,
                )
            else:
                losses["temporal_prior_spatial_relation_loss"] = self.bce_loss(
                    pred["temporal_prior_spatial_distribution"],
                    spatial_label,
                )
                losses["temporal_prior_contact_relation_loss"] = self.bce_loss(
                    pred["temporal_prior_contact_distribution"],
                    contact_label,
                )

        return losses

    def compute_tempura_losses(self, pred: Dict[str, Any], unc_vals: Any = None) -> Dict[str, torch.Tensor]:
        """Compute losses for Tempura model.
        
        :param pred: Model predictions
        :type pred: Dict[str, Any]
        :param unc_vals: Uncertainty values for loss weighting
        :type unc_vals: Any
        :return: Dictionary of loss components
        :rtype: Dict[str, torch.Tensor]
        """
        attention_distribution = pred["attention_distribution"]
        spatial_distribution = pred["spatial_distribution"]
        contact_distribution = pred["contacting_distribution"]

        if self.config.rel_head == "gmm":
            attention_distribution = torch.log(
                attention_distribution + 1e-12
            )

        if self.config.obj_head == "gmm" and self.config.mode != "predcls":
            pred["distribution"] = torch.log(pred["distribution"] + 1e-12)

        attention_label = (
            torch.tensor(pred["attention_gt"], dtype=torch.long)
            .to(device=attention_distribution.device)
            .squeeze()
        )
        
        # Ensure attention_label is 1D for CrossEntropyLoss
        if attention_label.dim() > 1:
            attention_label = attention_label.flatten()
        
        if self.config.mlm:
            # multi-label margin loss or adaptive loss
            spatial_label = -torch.ones(
                [len(pred["spatial_gt"]), 6], dtype=torch.long
            ).to(device=attention_distribution.device)
            contact_label = -torch.ones(
                [len(pred["contacting_gt"]), 17], dtype=torch.long
            ).to(device=attention_distribution.device)
            
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, : len(pred["spatial_gt"][i])] = (
                    torch.tensor(pred["spatial_gt"][i])
                )
                contact_label[i, : len(pred["contacting_gt"][i])] = (
                    torch.tensor(pred["contacting_gt"][i])
                )
        else:
            # bce loss
            spatial_label = torch.zeros(
                [len(pred["spatial_gt"]), 6], dtype=torch.float32
            ).to(device=attention_distribution.device)
            contact_label = torch.zeros(
                [len(pred["contacting_gt"]), 17], dtype=torch.float32
            ).to(device=attention_distribution.device)
            
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, pred["spatial_gt"][i]] = 1
                contact_label[i, pred["contacting_gt"][i]] = 1

        losses = {}
        if self.config.mode == "sgcls" or self.config.mode == "sgdet":
            losses["object_loss"] = self.ce_loss_obj(
                pred["distribution"], pred["labels"]
            )
            
            # Apply loss weighting if configured
            loss_weighting = self.config.obj_loss_weighting
            if loss_weighting is not None and unc_vals is not None:
                num = torch.exp(
                    unc_vals.obj_batch_unc[loss_weighting].sum(-1)
                )
                den = num.sum()
                weights = 1 + (num / den).to(device=pred["distribution"].device)
                losses["object_loss"] = weights * losses["object_loss"]
            
            losses["object_loss"] = losses["object_loss"].mean()
            
            if self.config.obj_con_loss and self.con_loss is not None:
                losses["object_contrastive_loss"] = (
                    self.config.lambda_con
                    * self.con_loss(pred["object_mem_features"], pred["labels"])
                )

        losses["attention_relation_loss"] = self.ce_loss_rel(
            attention_distribution, attention_label
        )
        
        if self.config.mlm:
            losses["spatial_relation_loss"] = self.mlm_loss(
                spatial_distribution, spatial_label
            )
            losses["contacting_relation_loss"] = self.mlm_loss(
                contact_distribution, contact_label
            )
        else:
            losses["spatial_relation_loss"] = self.bce_loss(
                spatial_distribution, spatial_label
            )
            losses["contacting_relation_loss"] = self.bce_loss(
                contact_distribution, contact_label
            )

        # Apply relation loss weighting if configured
        loss_weighting = self.config.rel_loss_weighting
        if loss_weighting is not None and unc_vals is not None:
            for rel in ["attention", "spatial", "contacting"]:
                num = torch.exp(
                    unc_vals.rel_batch_unc[rel][loss_weighting].sum(-1)
                )
                den = num.sum() + 1e-12
                weights = 1 + (num / den).to(device=pred["distribution"].device)

                if rel != "attention":
                    weights = weights.unsqueeze(-1).repeat(
                        1, losses[rel + "_relation_loss"].shape[-1]
                    )

                losses[rel + "_relation_loss"] = (
                    weights * losses[rel + "_relation_loss"]
                )

        for rel in ["attention", "spatial", "contacting"]:
            losses[rel + "_relation_loss"] = losses[
                rel + "_relation_loss"
            ].mean()

        return losses

    def compute_scenellm_losses(self, pred: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute losses for SceneLLM model.
        
        :param pred: Model predictions
        :type pred: Dict[str, Any]
        :return: Dictionary of loss components
        :rtype: Dict[str, torch.Tensor]
        """
        attention_distribution = pred["attention_distribution"]
        spatial_distribution = pred["spatial_distribution"]
        contact_distribution = pred["contact_distribution"]
        
        attention_label = (
            torch.tensor(pred["attention_gt"], dtype=torch.long)
            .to(device=attention_distribution.device)
            .squeeze()
        )
        
        # Ensure attention_label is 1D for CrossEntropyLoss
        if attention_label.dim() > 1:
            attention_label = attention_label.flatten()

        # Handle spatial and contact labels
        if not self.config.bce_loss:
            # multi-label margin loss
            spatial_label = -torch.ones(
                [len(pred["spatial_gt"]), 6], dtype=torch.long
            ).to(device=attention_distribution.device)
            contact_label = -torch.ones(
                [len(pred["contact_gt"]), 17], dtype=torch.long
            ).to(device=attention_distribution.device)
            
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, : len(pred["spatial_gt"][i])] = (
                    torch.tensor(pred["spatial_gt"][i])
                )
                contact_label[i, : len(pred["contact_gt"][i])] = (
                    torch.tensor(pred["contact_gt"][i])
                )
        else:
            # bce loss
            spatial_label = torch.zeros(
                [len(pred["spatial_gt"]), 6], dtype=torch.float32
            ).to(device=attention_distribution.device)
            contact_label = torch.zeros(
                [len(pred["contact_gt"]), 17], dtype=torch.float32
            ).to(device=attention_distribution.device)
            
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, pred["spatial_gt"][i]] = 1
                contact_label[i, pred["contact_gt"][i]] = 1

        losses = {}

        # VQ-VAE losses (only during VQ-VAE pretraining)
        if self.config.scenellm_training_stage == "vqvae":
            losses["vq_loss"] = pred["vq_loss"]
            losses["recon_loss"] = pred["recon_loss"]
            losses["embedding_loss"] = pred["embedding_loss"]
            losses["commitment_loss"] = pred["commitment_loss"]
        else:
            # SGG losses for stage 1 and stage 2
            if self.config.mode == "sgcls" or self.config.mode == "sgdet":
                losses["object_loss"] = self.config.alpha_obj * self.ce_loss(
                    pred["distribution"], pred["labels"]
                )

            losses["attention_relation_loss"] = self.config.alpha_rel * self.ce_loss(
                attention_distribution, attention_label
            )

            if not self.config.bce_loss:
                losses["spatial_relation_loss"] = self.config.alpha_rel * self.mlm_loss(
                    spatial_distribution, spatial_label
                )
                losses["contact_relation_loss"] = self.config.alpha_rel * self.mlm_loss(
                    contact_distribution, contact_label
                )
            else:
                losses["spatial_relation_loss"] = self.config.alpha_rel * self.bce_loss(
                    spatial_distribution, spatial_label
                )
                losses["contact_relation_loss"] = self.config.alpha_rel * self.bce_loss(
                    contact_distribution, contact_label
                )

        return losses

    def compute_oed_losses(self, pred: Dict[str, Any], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Compute losses for OED model.
        
        :param pred: Model predictions
        :type pred: Dict[str, Any]
        :param model: OED model instance
        :type model: torch.nn.Module
        :return: Dictionary of loss components
        :rtype: Dict[str, torch.Tensor]
        """
        # OED loss computation using the existing matcher
        if self.config.use_matcher:
            from lib.matcher import HungarianMatcher
            from lib.oed.criterion import SetCriterionOED

            # Initialize OED criterion if not already done
            if not hasattr(model, "criterion"):
                matcher = HungarianMatcher(0.5, 1, 1, 0.5)
                weight_dict = {
                    "loss_obj_ce": self.config.obj_loss_coef,
                    "loss_attn_ce": self.config.rel_loss_coef,
                    "loss_spatial_ce": self.config.rel_loss_coef,
                    "loss_contacting_ce": self.config.rel_loss_coef,
                    "loss_sub_bbox": self.config.bbox_loss_coef,
                    "loss_obj_bbox": self.config.bbox_loss_coef,
                    "loss_sub_giou": self.config.giou_loss_coef,
                    "loss_obj_giou": self.config.giou_loss_coef,
                }

                model.criterion = SetCriterionOED(
                    num_obj_classes=len(self.dataset_train.object_classes),
                    num_queries=self.config.num_queries,
                    matcher=matcher,
                    weight_dict=weight_dict,
                    eos_coef=self.config.oed_eos_coef,
                    losses=[
                        "obj_labels",
                        "relation_labels",
                        "sub_obj_boxes",
                    ],
                    conf=self.config,
                )

            # Convert predictions to OED format
            oed_pred = {
                "pred_obj_logits": pred["distribution"].unsqueeze(0),
                "pred_sub_boxes": torch.zeros(
                    1,
                    len(pred["labels"]),
                    4,
                    device=pred["distribution"].device,
                ),
                "pred_obj_boxes": torch.zeros(
                    1,
                    len(pred["labels"]),
                    4,
                    device=pred["distribution"].device,
                ),
                "pred_attn_logits": pred[
                    "attention_distribution"
                ].unsqueeze(0),
                "pred_spatial_logits": pred[
                    "spatial_distribution"
                ].unsqueeze(0),
                "pred_contacting_logits": pred[
                    "contact_distribution"
                ].unsqueeze(0),
            }

            # Convert targets to OED format
            oed_targets = [
                {
                    "obj_labels": pred["labels"],
                    "sub_boxes": torch.zeros(
                        len(pred["labels"]), 4, device=pred["labels"].device
                    ),
                    "obj_boxes": torch.zeros(
                        len(pred["labels"]), 4, device=pred["labels"].device
                    ),
                    "attn_labels": pred["attention_gt"],
                    "spatial_labels": pred["spatial_gt"],
                    "contacting_labels": pred["contact_gt"],
                }
            ]

            # Compute OED losses
            losses = model.criterion(oed_pred, oed_targets)
        else:
            # Fallback to standard loss computation
            losses = {}
            if self.config.mode == "sgcls" or self.config.mode == "sgdet":
                losses["object_loss"] = self.ce_loss(
                    pred["distribution"], pred["labels"]
                )

            # Handle attention_gt which might be a list or tensor
            attention_gt = pred["attention_gt"]
            if isinstance(attention_gt, list):
                attention_label = torch.tensor(
                    attention_gt,
                    dtype=torch.long,
                    device=pred["attention_distribution"].device,
                )
            else:
                attention_label = (
                    attention_gt.clone()
                    .detach()
                    .to(device=pred["attention_distribution"].device)
                    .squeeze()
                )

            # Ensure attention_label is 1D for CrossEntropyLoss
            if attention_label.dim() > 1:
                attention_label = attention_label.flatten()

            losses["attention_relation_loss"] = self.ce_loss(
                pred["attention_distribution"], attention_label
            )

            if not self.config.bce_loss:
                spatial_label = -torch.ones(
                    [len(pred["spatial_gt"]), 6], dtype=torch.long
                ).to(device=pred["attention_distribution"].device)
                contact_label = -torch.ones(
                    [len(pred["contact_gt"]), 17], dtype=torch.long
                ).to(device=pred["attention_distribution"].device)
                
                for i in range(len(pred["spatial_gt"])):
                    # Handle spatial_gt and contact_gt which might be lists or tensors
                    spatial_gt_item = pred["spatial_gt"][i]
                    contact_gt_item = pred["contact_gt"][i]

                    if isinstance(spatial_gt_item, list):
                        spatial_gt_tensor = torch.tensor(
                            spatial_gt_item,
                            dtype=torch.long,
                            device=pred["attention_distribution"].device,
                        )
                    else:
                        spatial_gt_tensor = spatial_gt_item.clone().detach()

                    if isinstance(contact_gt_item, list):
                        contact_gt_tensor = torch.tensor(
                            contact_gt_item,
                            dtype=torch.long,
                            device=pred["attention_distribution"].device,
                        )
                    else:
                        contact_gt_tensor = contact_gt_item.clone().detach()

                    spatial_label[i, : len(spatial_gt_tensor)] = (
                        spatial_gt_tensor
                    )
                    contact_label[i, : len(contact_gt_tensor)] = (
                        contact_gt_tensor
                    )

                losses["spatial_relation_loss"] = self.mlm_loss(
                    pred["spatial_distribution"], spatial_label
                )
                losses["contact_relation_loss"] = self.mlm_loss(
                    pred["contact_distribution"], contact_label
                )
            else:
                spatial_label = torch.zeros(
                    [len(pred["spatial_gt"]), 6], dtype=torch.float32
                ).to(device=pred["attention_distribution"].device)
                contact_label = torch.zeros(
                    [len(pred["contact_gt"]), 17], dtype=torch.float32
                ).to(device=pred["attention_distribution"].device)
                
                for i in range(len(pred["spatial_gt"])):
                    # Handle spatial_gt and contact_gt which might be lists or tensors
                    spatial_gt_item = pred["spatial_gt"][i]
                    contact_gt_item = pred["contact_gt"][i]

                    if isinstance(spatial_gt_item, list):
                        spatial_indices = spatial_gt_item
                    else:
                        spatial_indices = spatial_gt_item.tolist()

                    if isinstance(contact_gt_item, list):
                        contact_indices = contact_gt_item
                    else:
                        contact_indices = contact_gt_item.tolist()

                    spatial_label[i, spatial_indices] = 1
                    contact_label[i, contact_indices] = 1

                losses["spatial_relation_loss"] = self.bce_loss(
                    pred["spatial_distribution"], spatial_label
                )
                losses["contact_relation_loss"] = self.bce_loss(
                    pred["contact_distribution"], contact_label
                )

        return losses
