"""STTRAN model configuration classes.

This module provides structured configuration classes specifically
for the STTRAN (Spatial-Temporal Transformer) model.

:author: M3SGG Team
:version: 0.1.0
"""

from dataclasses import dataclass
from typing import Optional

from .base import BaseConfig, TrainingConfig, DataConfig, LoggingConfig, CheckpointConfig, EvaluationConfig, ModelConfig, LossConfig


@dataclass
class STTRANConfig(BaseConfig):
    """Configuration for STTRAN model.

    STTRAN (Spatial-Temporal Transformer) is a transformer-based model
    for video scene graph generation that processes spatial and temporal
    information through attention mechanisms.

    :param model_type: Model type identifier
    :type model_type: str
    :param lr: Learning rate (overridden for STTRAN)
    :type lr: float
    :param hidden_dim: Hidden dimension size
    :type hidden_dim: int
    :param num_heads: Number of attention heads
    :type num_heads: int
    :param num_layers: Number of transformer layers
    :type num_layers: int
    :param dropout: Dropout rate
    :type dropout: float
    :param use_spatial_encoding: Use spatial position encoding
    :type use_spatial_encoding: bool
    :param use_temporal_encoding: Use temporal position encoding
    :type use_temporal_encoding: bool
    :param max_seq_len: Maximum sequence length
    :type max_seq_len: int
    :param spatial_dim: Spatial dimension size
    :type spatial_dim: int
    :param temporal_dim: Temporal dimension size
    :type temporal_dim: int
    :param obj_feat_dim: Object feature dimension
    :type obj_feat_dim: int
    :param rel_feat_dim: Relation feature dimension
    :type rel_feat_dim: int
    :param num_obj_classes: Number of object classes
    :type num_obj_classes: int
    :param num_rel_classes: Number of relation classes
    :type num_rel_classes: int
    :param use_bbox_encoding: Use bounding box encoding
    :type use_bbox_encoding: bool
    :param bbox_encoding_dim: Bounding box encoding dimension
    :type bbox_encoding_dim: int
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
    """

    # Model identification
    model_type: str = "sttran"
    
    # Learning rate override for STTRAN
    lr: float = 2e-5
    
    # Architecture parameters
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    
    # Encoding parameters
    use_spatial_encoding: bool = True
    use_temporal_encoding: bool = True
    max_seq_len: int = 1000
    spatial_dim: int = 64
    temporal_dim: int = 64
    
    # Feature dimensions
    obj_feat_dim: int = 2048
    rel_feat_dim: int = 256
    
    # Class information
    num_obj_classes: int = 35
    num_rel_classes: int = 132
    
    # Bounding box encoding
    use_bbox_encoding: bool = True
    bbox_encoding_dim: int = 128
    
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


@dataclass
class STTRANTrainingConfig(TrainingConfig):
    """STTRAN-specific training configuration.

    :param warmup_epochs: Number of warmup epochs (STTRAN specific)
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
    """

    warmup_epochs: int = 2
    scheduler_type: str = "reduce_on_plateau"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    use_amp: bool = False
    accumulation_steps: int = 1


@dataclass
class STTRANLossConfig(LossConfig):
    """STTRAN-specific loss configuration.

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
    """

    obj_loss_weight: float = 1.0
    rel_loss_weight: float = 2.0
    bbox_loss_weight: float = 2.5
    giou_loss_weight: float = 1.0
    
    use_focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    
    use_class_weights: bool = False
    obj_class_weights: Optional[dict] = None
    rel_class_weights: Optional[dict] = None


@dataclass
class STTRANDataConfig(DataConfig):
    """STTRAN-specific data configuration.

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
