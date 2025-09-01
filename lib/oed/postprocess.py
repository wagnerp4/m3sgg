"""
Postprocessing module for OED model.

This module handles the post-processing of model outputs for evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PostProcessOED(nn.Module):
    """Post-processing for OED model outputs.
    
    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """
    
    def __init__(self, conf):
        """Initialize the post-processor.
        
        :param conf: Configuration object
        :type conf: Config
        :return: None
        :rtype: None
        """
        super().__init__()
        self.subject_category_id = getattr(conf, "subject_category_id", 1)
        self.use_matching = conf.oed_use_matching

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Post-process model outputs.
        
        :param outputs: Model outputs
        :type outputs: dict
        :param target_sizes: Target image sizes
        :type target_sizes: torch.Tensor
        :return: List of processed predictions
        :rtype: list
        """
        out_obj_logits = outputs["pred_obj_logits"]
        out_sub_boxes = outputs["pred_sub_boxes"]
        out_obj_boxes = outputs["pred_obj_boxes"]
        out_attn_logits = outputs["pred_attn_logits"]
        out_spatial_logits = outputs["pred_spatial_logits"]
        out_contacting_logits = outputs["pred_contacting_logits"]

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # Process object predictions
        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        # Process relation predictions
        attn_probs = out_attn_logits[..., :-1].softmax(-1)
        spatial_probs = out_spatial_logits.sigmoid()
        contacting_probs = out_contacting_logits.sigmoid()
        
        # Combine bounding boxes
        out_boxes = torch.cat([out_sub_boxes, out_obj_boxes], dim=1)

        results = []
        for index in range(len(target_sizes)):
            frame_pred = {}
            
            # Object predictions
            frame_pred["pred_scores"] = torch.cat([
                torch.ones(out_sub_boxes.shape[1]), 
                obj_scores[index].cpu()
            ]).numpy()
            
            frame_pred["pred_labels"] = torch.cat([
                torch.ones(out_sub_boxes.shape[1]), 
                obj_labels[index].cpu()
            ]).numpy()
            
            frame_pred["pred_boxes"] = out_boxes[index].cpu().numpy()
            
            # Pair indices
            frame_pred["pair_idx"] = torch.cat([
                torch.arange(out_sub_boxes.shape[1])[:, None],
                torch.arange(out_sub_boxes.shape[1], 2 * out_sub_boxes.shape[1])[:, None]
            ], dim=1).cpu().numpy()
            
            # Relation distributions
            frame_pred["attention_distribution"] = attn_probs[index].cpu().numpy()
            frame_pred["spatial_distribution"] = spatial_probs[index].cpu().numpy()
            frame_pred["contacting_distribution"] = contacting_probs[index].cpu().numpy()

            results.append(frame_pred)

        return results
