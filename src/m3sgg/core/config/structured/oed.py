"""OED model configuration classes.

This module provides structured configuration classes specifically
for the OED (Object-Event Detection) model.

:author: M3SGG Team
:version: 0.1.0
"""

from dataclasses import dataclass
from typing import Optional

from .base import BaseConfig, TrainingConfig, DataConfig, LoggingConfig, CheckpointConfig, EvaluationConfig, ModelConfig, LossConfig


@dataclass
class OEDConfig(BaseConfig):
    """Configuration for OED model.

    OED (Object-Event Detection) is a transformer-based model for
    video scene graph generation that uses object queries and
    attention mechanisms for detection.

    :param model_type: Model type identifier
    :type model_type: str
    :param num_queries: Number of query slots for OED
    :type num_queries: int
    :param dec_layers_hopd: Number of hopd decoding layers in OED transformer
    :type dec_layers_hopd: int
    :param dec_layers_interaction: Number of interaction decoding layers in OED transformer
    :type dec_layers_interaction: int
    :param num_attn_classes: Number of attention classes
    :type num_attn_classes: int
    :param num_spatial_classes: Number of spatial classes
    :type num_spatial_classes: int
    :param num_contacting_classes: Number of contacting classes
    :type num_contacting_classes: int
    :param alpha: Focal loss alpha for OED
    :type alpha: float
    :param oed_use_matching: Use obj/sub matching 2class loss in OED decoder
    :type oed_use_matching: bool
    :param bbox_loss_coef: L1 box coefficient
    :type bbox_loss_coef: float
    :param giou_loss_coef: GIoU box coefficient
    :type giou_loss_coef: float
    :param obj_loss_coef: Object classification coefficient
    :type obj_loss_coef: float
    :param rel_loss_coef: Relation classification coefficient
    :type rel_loss_coef: float
    :param oed_eos_coef: Relative classification weight of no-object class for OED
    :type oed_eos_coef: float
    :param interval1: Interval for training frame selection
    :type interval1: int
    :param interval2: Interval for test frame selection
    :type interval2: int
    :param num_ref_frames: Number of reference frames
    :type num_ref_frames: int
    :param oed_variant: OED variant (single/multi)
    :type oed_variant: str
    :param fuse_semantic_pos: Fuse semantic and positional embeddings
    :type fuse_semantic_pos: bool
    :param query_temporal_interaction: Enable query temporal interaction
    :type query_temporal_interaction: bool
    :param hidden_dim: Hidden dimension size
    :type hidden_dim: int
    :param num_heads: Number of attention heads
    :type num_heads: int
    :param num_layers: Number of transformer layers
    :type num_layers: int
    :param dropout: Dropout rate
    :type dropout: float
    :param use_bbox_encoding: Use bounding box encoding
    :type use_bbox_encoding: bool
    :param bbox_encoding_dim: Bounding box encoding dimension
    :type bbox_encoding_dim: int
    :param use_positional_encoding: Use positional encoding
    :type use_positional_encoding: bool
    :param positional_encoding_dim: Positional encoding dimension
    :type positional_encoding_dim: int
    :param use_temporal_encoding: Use temporal encoding
    :type use_temporal_encoding: bool
    :param temporal_encoding_dim: Temporal encoding dimension
    :type temporal_encoding_dim: int
    :param use_attention_weights: Use attention weights for visualization
    :type use_attention_weights: bool
    :param attention_dropout: Attention dropout rate
    :type attention_dropout: float
    :param ffn_dim: Feed-forward network dimension
    :type ffn_dim: int
    :param activation: Activation function
    :type activation: str
    :param norm_type: Normalization type
    :type norm_type: str
    :param use_bias: Use bias in linear layers
    :type use_bias: bool
    :param gradient_checkpointing: Use gradient checkpointing
    :type gradient_checkpointing: bool
    :param use_memory_efficient_attention: Use memory efficient attention
    :type use_memory_efficient_attention: bool
    :param use_auxiliary_loss: Use auxiliary loss
    :type use_auxiliary_loss: bool
    :param aux_loss_weight: Auxiliary loss weight
    :type aux_loss_weight: float
    :param use_contrastive_loss: Use contrastive loss
    :type use_contrastive_loss: bool
    :param contrastive_loss_weight: Contrastive loss weight
    :type contrastive_loss_weight: float
    :param contrastive_temperature: Contrastive loss temperature
    :type contrastive_temperature: float
    :param use_consistency_loss: Use consistency loss
    :type use_consistency_loss: bool
    :param consistency_loss_weight: Consistency loss weight
    :type consistency_loss_weight: float
    :param use_regularization_loss: Use regularization loss
    :type use_regularization_loss: bool
    :param regularization_loss_weight: Regularization loss weight
    :type regularization_loss_weight: float
    """

    # Model identification
    model_type: str = "oed"
    
    # Query parameters
    num_queries: int = 100
    
    # Decoder parameters
    dec_layers_hopd: int = 6
    dec_layers_interaction: int = 6
    
    # Class parameters
    num_attn_classes: int = 3
    num_spatial_classes: int = 6
    num_contacting_classes: int = 17
    
    # Loss coefficients
    alpha: float = 0.5
    oed_use_matching: bool = False
    bbox_loss_coef: float = 2.5
    giou_loss_coef: float = 1.0
    obj_loss_coef: float = 1.0
    rel_loss_coef: float = 2.0
    oed_eos_coef: float = 0.1
    
    # Frame parameters
    interval1: int = 4
    interval2: int = 4
    num_ref_frames: int = 2
    oed_variant: str = "multi"
    
    # Encoding parameters
    fuse_semantic_pos: bool = False
    query_temporal_interaction: bool = False
    
    # Architecture parameters
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    
    # Encoding parameters
    use_bbox_encoding: bool = True
    bbox_encoding_dim: int = 128
    use_positional_encoding: bool = True
    positional_encoding_dim: int = 128
    use_temporal_encoding: bool = True
    temporal_encoding_dim: int = 128
    
    # Attention parameters
    use_attention_weights: bool = False
    attention_dropout: float = 0.1
    
    # Feed-forward network
    ffn_dim: int = 1024
    activation: str = "relu"
    norm_type: str = "layer_norm"
    use_bias: bool = True
    
    # Optimization parameters
    gradient_checkpointing: bool = False
    use_memory_efficient_attention: bool = False
    
    # Additional loss parameters
    use_auxiliary_loss: bool = False
    aux_loss_weight: float = 0.1
    use_contrastive_loss: bool = False
    contrastive_loss_weight: float = 0.1
    contrastive_temperature: float = 0.07
    use_consistency_loss: bool = False
    consistency_loss_weight: float = 0.1
    use_regularization_loss: bool = False
    regularization_loss_weight: float = 0.01


@dataclass
class OEDTrainingConfig(TrainingConfig):
    """OED-specific training configuration.

    :param warmup_epochs: Number of warmup epochs
    :type warmup_epochs: int
    :param scheduler_type: Learning rate scheduler type
    :type scheduler_type: str
    :param scheduler_patience: Scheduler patience
    :type scheduler_patience: int
    :param scheduler_factor: Scheduler reduction factor
    :type scheduler_factor: float
    :param weight_decay: Weight decay for regularization
    :type weight_decay: float
    :param clip_grad_norm: Gradient clipping norm
    :type clip_grad_norm: float
    :param use_amp: Use automatic mixed precision
    :type use_amp: bool
    :param accumulation_steps: Gradient accumulation steps
    :type accumulation_steps: int
    :param use_ema: Use exponential moving average
    :type use_ema: bool
    :param ema_decay: EMA decay rate
    :type ema_decay: float
    :param use_swa: Use stochastic weight averaging
    :type use_swa: bool
    :param swa_lr: SWA learning rate
    :type swa_lr: float
    :param swa_epochs: SWA epochs
    :type swa_epochs: int
    :param use_curriculum_learning: Use curriculum learning
    :type use_curriculum_learning: bool
    :param curriculum_epochs: Number of epochs for curriculum
    :type curriculum_epochs: int
    :param use_auxiliary_loss: Use auxiliary loss
    :type use_auxiliary_loss: bool
    :param aux_loss_weight: Auxiliary loss weight
    :type aux_loss_weight: float
    :param use_contrastive_loss: Use contrastive loss
    :type use_contrastive_loss: bool
    :param contrastive_loss_weight: Contrastive loss weight
    :type contrastive_loss_weight: float
    :param contrastive_temperature: Contrastive loss temperature
    :type contrastive_temperature: float
    :param use_consistency_loss: Use consistency loss
    :type use_consistency_loss: bool
    :param consistency_loss_weight: Consistency loss weight
    :type consistency_loss_weight: float
    :param use_regularization_loss: Use regularization loss
    :type use_regularization_loss: bool
    :param regularization_loss_weight: Regularization loss weight
    :type regularization_loss_weight: float
    """

    warmup_epochs: int = 2
    scheduler_type: str = "reduce_on_plateau"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    use_amp: bool = False
    accumulation_steps: int = 1
    use_ema: bool = False
    ema_decay: float = 0.999
    use_swa: bool = False
    swa_lr: float = 1e-5
    swa_epochs: int = 5
    use_curriculum_learning: bool = False
    curriculum_epochs: int = 10
    use_auxiliary_loss: bool = False
    aux_loss_weight: float = 0.1
    use_contrastive_loss: bool = False
    contrastive_loss_weight: float = 0.1
    contrastive_temperature: float = 0.07
    use_consistency_loss: bool = False
    consistency_loss_weight: float = 0.1
    use_regularization_loss: bool = False
    regularization_loss_weight: float = 0.01


@dataclass
class OEDLossConfig(LossConfig):
    """OED-specific loss configuration.

    :param obj_loss_weight: Weight for object classification loss
    :type obj_loss_weight: float
    :param rel_loss_weight: Weight for relation classification loss
    :type rel_loss_weight: float
    :param bbox_loss_weight: Weight for bounding box regression loss
    :type bbox_loss_weight: float
    :param giou_loss_weight: Weight for GIoU loss
    :type giou_loss_weight: float
    :param use_focal_loss: Use focal loss for classification
    :type use_focal_loss: bool
    :param focal_alpha: Focal loss alpha parameter
    :type focal_alpha: float
    :param focal_gamma: Focal loss gamma parameter
    :type focal_gamma: float
    :param label_smoothing: Label smoothing factor
    :type label_smoothing: float
    :param use_class_weights: Use class weights for imbalanced data
    :type use_class_weights: bool
    :param obj_class_weights: Object class weights
    :type obj_class_weights: Optional[dict]
    :param rel_class_weights: Relation class weights
    :type rel_class_weights: Optional[dict]
    :param use_auxiliary_loss: Use auxiliary loss
    :type use_auxiliary_loss: bool
    :param aux_loss_weight: Auxiliary loss weight
    :type aux_loss_weight: float
    :param use_contrastive_loss: Use contrastive loss
    :type use_contrastive_loss: bool
    :param contrastive_loss_weight: Contrastive loss weight
    :type contrastive_loss_weight: float
    :param contrastive_temperature: Contrastive loss temperature
    :type contrastive_temperature: float
    :param use_consistency_loss: Use consistency loss
    :type use_consistency_loss: bool
    :param consistency_loss_weight: Consistency loss weight
    :type consistency_loss_weight: float
    :param use_regularization_loss: Use regularization loss
    :type use_regularization_loss: bool
    :param regularization_loss_weight: Regularization loss weight
    :type regularization_loss_weight: float
    """

    obj_loss_weight: float = 1.0
    rel_loss_weight: float = 2.0
    bbox_loss_weight: float = 2.5
    giou_loss_weight: float = 1.0
    
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    
    use_class_weights: bool = False
    obj_class_weights: Optional[dict] = None
    rel_class_weights: Optional[dict] = None
    
    use_auxiliary_loss: bool = False
    aux_loss_weight: float = 0.1
    use_contrastive_loss: bool = False
    contrastive_loss_weight: float = 0.1
    contrastive_temperature: float = 0.07
    use_consistency_loss: bool = False
    consistency_loss_weight: float = 0.1
    use_regularization_loss: bool = False
    regularization_loss_weight: float = 0.01


@dataclass
class OEDDataConfig(DataConfig):
    """OED-specific data configuration.

    :param max_objects: Maximum number of objects per frame
    :type max_objects: int
    :param max_relations: Maximum number of relations per frame
    :type max_relations: int
    :param max_frames: Maximum number of frames per video
    :type max_frames: int
    :param use_temporal_sampling: Use temporal sampling
    :type use_temporal_sampling: bool
    :param temporal_stride: Temporal stride for sampling
    :type temporal_stride: int
    :param use_augmentation: Use data augmentation
    :type use_augmentation: bool
    :param augmentation_prob: Probability of applying augmentation
    :type augmentation_prob: float
    :param use_negative_sampling: Use negative sampling for relations
    :type use_negative_sampling: bool
    :param negative_ratio: Ratio of negative samples
    :type negative_ratio: float
    :param use_hard_negative_mining: Use hard negative mining
    :type use_hard_negative_mining: bool
    :param hard_negative_ratio: Ratio of hard negative samples
    :type hard_negative_ratio: float
    :param use_contrastive_sampling: Use contrastive sampling
    :type use_contrastive_sampling: bool
    :param contrastive_ratio: Ratio of contrastive samples
    :type contrastive_ratio: float
    :param use_curriculum_learning: Use curriculum learning
    :type use_curriculum_learning: bool
    :param curriculum_epochs: Number of epochs for curriculum
    :type curriculum_epochs: int
    :param use_dynamic_sampling: Use dynamic sampling
    :type use_dynamic_sampling: bool
    :param dynamic_sampling_alpha: Dynamic sampling alpha
    :type dynamic_sampling_alpha: float
    :param use_balanced_sampling: Use balanced sampling
    :type use_balanced_sampling: bool
    :param balanced_sampling_alpha: Balanced sampling alpha
    :type balanced_sampling_alpha: float
    """

    max_objects: int = 50
    max_relations: int = 100
    max_frames: int = 30
    use_temporal_sampling: bool = True
    temporal_stride: int = 1
    use_augmentation: bool = False
    augmentation_prob: float = 0.5
    use_negative_sampling: bool = True
    negative_ratio: float = 0.5
    use_hard_negative_mining: bool = False
    hard_negative_ratio: float = 0.2
    use_contrastive_sampling: bool = False
    contrastive_ratio: float = 0.3
    use_curriculum_learning: bool = False
    curriculum_epochs: int = 10
    use_dynamic_sampling: bool = False
    dynamic_sampling_alpha: float = 0.5
    use_balanced_sampling: bool = False
    balanced_sampling_alpha: float = 0.5
