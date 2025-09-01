"""
OED Single-frame Model Implementation

This module implements the OED architecture for single-frame scene graph generation
as a baseline for comparison with the multi-frame variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



from .transformer import build_transformer
from .criterion import SetCriterionOED
from .postprocess import PostProcessOED


class OEDSingle(nn.Module):
    """OED Single-frame model for scene graph generation.
    
    Implements the one-stage end-to-end framework with cascaded decoders
    for single-frame processing.
    
    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """
    
    def __init__(self, conf, dataset):
        """Initialize the OED Single-frame model.
        
        :param conf: Configuration object containing model parameters
        :type conf: Config
        :param dataset: Dataset information for model setup
        :type dataset: object
        :return: None
        :rtype: None
        """
        super().__init__()
        
        # Store configuration and dataset
        self.conf = conf
        self.dataset = dataset
        
        # Model parameters
        self.num_queries = conf.num_queries
        self.hidden_dim = 256  # Default hidden dimension
        self.num_obj_classes = len(dataset.object_classes)
        self.num_attn_classes = conf.num_attn_classes
        self.num_spatial_classes = conf.num_spatial_classes
        self.num_contacting_classes = conf.num_contacting_classes
        
        # Build transformer (remove backbone dependency)
        # self.backbone = build_backbone(conf)  # Remove backbone dependency
        self.transformer = build_transformer(conf)
        
        # Query embedding
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        
        # Classification heads
        self.obj_class_embed = nn.Linear(self.hidden_dim, self.num_obj_classes + 1)
        self.attn_class_embed = nn.Linear(self.hidden_dim, self.num_attn_classes + 1)
        self.spatial_class_embed = nn.Linear(self.hidden_dim, self.num_spatial_classes)
        self.contacting_class_embed = nn.Linear(self.hidden_dim, self.num_contacting_classes)
        
        # Bounding box heads
        self.sub_bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        
        # Input projection - remove since we don't have backbone
        # self.input_proj = nn.Conv2d(self.backbone.num_channels, self.hidden_dim, kernel_size=1)
        
        # Hyperparameters
        self.aux_loss = getattr(conf, "aux_loss", False)
        self.dec_layers_hopd = conf.dec_layers_hopd
        self.dec_layers_interaction = conf.dec_layers_interaction
        self.use_matching = conf.oed_use_matching

    def forward(self, entry, targets=None):
        """Forward pass through the OED model.
        
        :param entry: Input data containing images and features from object detector
        :type entry: dict
        :param targets: Ground truth targets for training, defaults to None
        :type targets: dict, optional
        :return: Model predictions and outputs
        :rtype: dict
        """
        # Extract features from the object detector entry
        if "features" not in entry:
            raise ValueError("Entry must contain 'features' from object detector")
        
        # Use the features from the object detector
        features = entry["features"]  # Shape: (num_objects, feature_dim)
        
        # For now, create a simple transformer input
        # TODO: Implement proper feature processing for OED
        batch_size = 1  # Assume single batch for now
        num_objects = features.size(0)
        hidden_dim = self.hidden_dim
        
        # Create dummy inputs for transformer (this is a placeholder implementation)
        # In a real implementation, you would process the features properly
        src = features.unsqueeze(0)  # Add batch dimension: (1, num_objects, feature_dim)
        mask = torch.zeros(batch_size, num_objects, dtype=torch.bool, device=features.device)
        pos = torch.zeros(batch_size, num_objects, hidden_dim, device=features.device)
        
        # Process through transformer
        hopd_out, interaction_decoder_out, ins_attn_weight, rel_attn_weight = self.transformer(
            src, 
            mask, 
            self.query_embed.weight, 
            pos
        )[:4]
        
        # Generate outputs
        outputs_sub_coord = self.sub_bbox_embed(hopd_out).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hopd_out).sigmoid()
        outputs_obj_class = self.obj_class_embed(hopd_out)
        
        outputs_attn_class = self.attn_class_embed(interaction_decoder_out)
        outputs_spatial_class = self.spatial_class_embed(interaction_decoder_out)
        outputs_contacting_class = self.contacting_class_embed(interaction_decoder_out)
        
        # Extract ground truth from entry if available
        attention_gt = entry.get("attention_gt", [])
        spatial_gt = entry.get("spatial_gt", [])
        contact_gt = entry.get("contact_gt", [])
        labels = entry.get("labels", [])
        
        # Convert to tensors if they're lists
        if isinstance(attention_gt, list) and len(attention_gt) > 0:
            attention_gt = torch.tensor(attention_gt, dtype=torch.long, device=outputs_attn_class.device)
        if isinstance(spatial_gt, list) and len(spatial_gt) > 0:
            spatial_gt = [torch.tensor(sg, dtype=torch.long, device=outputs_spatial_class.device) for sg in spatial_gt]
        if isinstance(contact_gt, list) and len(contact_gt) > 0:
            contact_gt = [torch.tensor(cg, dtype=torch.long, device=outputs_contacting_class.device) for cg in contact_gt]
        if isinstance(labels, list) and len(labels) > 0:
            labels = torch.tensor(labels, dtype=torch.long, device=outputs_obj_class.device)
        
        # CRITICAL: Create pair indices that match the actual predictions  
        # In predcls mode, we need to ensure pair indices are valid for our predictions
        # The evaluator expects pair indices to reference valid objects in the prediction
        
        # Create pair indices that reference our actual predictions  
        # For simplicity, create pairs between first object (human) and all others
        num_pairs = self.num_queries
        pair_idx = torch.stack([
            torch.zeros(num_pairs, dtype=torch.long), 
            torch.arange(num_pairs, dtype=torch.long)
        ], dim=1).to(device=outputs_obj_class[-1].device)
        
        # Create image indices for all pairs
        if "im_idx" in entry and len(entry["im_idx"]) >= num_pairs:
            # Use the provided image indices, but truncate to our batch size
            im_idx = entry["im_idx"][:num_pairs]
        else:
            # Default to image 0 for all pairs
            im_idx = torch.zeros(num_pairs, dtype=torch.long, device=outputs_obj_class[-1].device)

        # Prepare output dictionary in the format expected by the training loop and evaluator
        out = {
            # Standard format expected by training loop
            "distribution": outputs_obj_class[-1],
            "attention_distribution": torch.softmax(outputs_attn_class[-1], dim=1),  # Evaluator expects softmax
            "spatial_distribution": torch.softmax(outputs_spatial_class[-1], dim=1),  # Evaluator expects softmax
            "contact_distribution": torch.softmax(outputs_contacting_class[-1], dim=1),  # Evaluator expects softmax
            
            # Ground truth labels
            "attention_gt": attention_gt,
            "spatial_gt": spatial_gt,
            "contact_gt": contact_gt,
            "labels": labels,
            
            # Critical fields required by Action Genome evaluator
            "pair_idx": pair_idx,
            "im_idx": im_idx,
            
            # OED format for compatibility
            "pred_obj_logits": outputs_obj_class[-1],
            "pred_sub_boxes": outputs_sub_coord[-1],
            "pred_obj_boxes": outputs_obj_coord[-1],
            "pred_attn_logits": outputs_attn_class[-1],
            "pred_spatial_logits": outputs_spatial_class[-1],
            "pred_contacting_logits": outputs_contacting_class[-1],
            "ins_attn_weight": ins_attn_weight,
            "rel_attn_weight": rel_attn_weight,
            
            # Additional fields for evaluation
            "pred_scores": torch.ones(self.num_queries, device=outputs_obj_class[-1].device),  # Default confidence scores
            "scores": torch.ones(self.num_queries, device=outputs_obj_class[-1].device),  # Object confidence scores
            "pred_labels": torch.argmax(outputs_obj_class[-1], dim=1),  # Predicted object labels
        }
        
        # CRITICAL: The evaluator expects individual object boxes in the correct format
        # The boxes field should contain individual object bounding boxes with batch indices
        # Format: [batch_idx, x1, y1, x2, y2] for each object
        if "boxes" in entry:
            # Use the original object boxes from the detector (this is the correct approach for predcls)
            out["boxes"] = entry["boxes"]  # Keep the full format with batch indices
        else:
            # Fallback: create boxes from OED predictions (should not happen in predcls mode)
            # Add batch index column to the predicted boxes
            batch_indices = torch.zeros(self.num_queries, 1, device=outputs_obj_class[-1].device)
            out["boxes"] = torch.cat([batch_indices, outputs_obj_coord[-1]], dim=1)
        
        # Add auxiliary outputs if enabled
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_obj_class, outputs_sub_coord, outputs_obj_coord,
                outputs_attn_class, outputs_spatial_class, outputs_contacting_class
            )
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_sub_coord, outputs_obj_coord, 
                      outputs_attn_class, outputs_spatial_class, outputs_contacting_class):
        """Set auxiliary outputs for loss computation.
        
        :param outputs_obj_class: Object classification outputs
        :type outputs_obj_class: torch.Tensor
        :param outputs_sub_coord: Subject bounding box outputs
        :type outputs_sub_coord: torch.Tensor
        :param outputs_obj_coord: Object bounding box outputs
        :type outputs_obj_coord: torch.Tensor
        :param outputs_attn_class: Attention classification outputs
        :type outputs_attn_class: torch.Tensor
        :param outputs_spatial_class: Spatial classification outputs
        :type outputs_spatial_class: torch.Tensor
        :param outputs_contacting_class: Contacting classification outputs
        :type outputs_spatial_class: torch.Tensor
        :return: List of auxiliary outputs
        :rtype: list
        """
        min_dec_layers_num = min(self.dec_layers_hopd, self.dec_layers_interaction)
        return [{
            "pred_obj_logits": a,
            "pred_sub_boxes": b,
            "pred_obj_boxes": c,
            "pred_attn_logits": d,
            "pred_spatial_logits": e,
            "pred_contacting_logits": f
        } for a, b, c, d, e, f in zip(
            outputs_obj_class[-min_dec_layers_num: -1],
            outputs_sub_coord[-min_dec_layers_num: -1],
            outputs_obj_coord[-min_dec_layers_num: -1],
            outputs_attn_class[-min_dec_layers_num: -1],
            outputs_spatial_class[-min_dec_layers_num: -1],
            outputs_contacting_class[-min_dec_layers_num: -1]
        )]


class MLP(nn.Module):
    """Multi-layer perceptron for bounding box prediction.
    
    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """Initialize the MLP.
        
        :param input_dim: Input dimension
        :type input_dim: int
        :param hidden_dim: Hidden dimension
        :type hidden_dim: int
        :param output_dim: Output dimension
        :type output_dim: int
        :param num_layers: Number of layers
        :type num_layers: int
        :return: None
        :rtype: None
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        """Forward pass through the MLP.
        
        :param x: Input tensor
        :type x: torch.Tensor
        :return: Output tensor
        :rtype: torch.Tensor
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_oed_single(conf, dataset):
    """Build OED single-frame model.
    
    :param conf: Configuration object
    :type conf: Config
    :param dataset: Dataset object
    :type dataset: object
    :return: Tuple of (model, criterion, postprocessors)
    :rtype: tuple
    """
    model = OEDSingle(conf, dataset)
    
    # Build criterion
    criterion = SetCriterionOED(
        num_obj_classes=len(dataset.object_classes),
        num_queries=conf.num_queries,
        matcher=None,  # Will be set during training
        weight_dict={},  # Will be set during training
        eos_coef=conf.oed_eos_coef,
        losses=["obj_labels", "relation_labels", "sub_obj_boxes"],
        conf=conf
    )
    
    # Build postprocessors
    postprocessors = {"oed": PostProcessOED(conf)}
    
    return model, criterion, postprocessors
