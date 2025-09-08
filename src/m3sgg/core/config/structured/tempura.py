"""Tempura model configuration classes.

This module provides structured configuration classes specifically
for the Tempura model, which includes memory mechanisms and GMM heads.

:author: M3SGG Team
:version: 0.1.0
"""

from dataclasses import dataclass
from typing import Optional

from .base import BaseConfig, TrainingConfig, DataConfig, LoggingConfig, CheckpointConfig, EvaluationConfig, ModelConfig, LossConfig


@dataclass
class TempuraConfig(BaseConfig):
    """Configuration for Tempura model.

    Tempura is a memory-enhanced model for video scene graph generation
    that uses Gaussian Mixture Model (GMM) heads and memory mechanisms
    for improved performance.

    :param model_type: Model type identifier
    :type model_type: str
    :param obj_head: Object classification head type
    :type obj_head: str
    :param rel_head: Relation classification head type
    :type rel_head: str
    :param K: Number of mixture models
    :type K: int
    :param rel_mem_compute: Relation memory computation type
    :type rel_mem_compute: Optional[str]
    :param obj_mem_compute: Object memory computation
    :type obj_mem_compute: bool
    :param take_obj_mem_feat: Take object memory features
    :type take_obj_mem_feat: bool
    :param obj_mem_weight_type: Object memory weight type
    :type obj_mem_weight_type: str
    :param rel_mem_weight_type: Relation memory weight type
    :type rel_mem_weight_type: str
    :param mem_feat_selection: Memory feature selection method
    :type mem_feat_selection: str
    :param mem_fusion: Memory fusion method
    :type mem_fusion: str
    :param mem_feat_lambda: Memory feature lambda
    :type mem_feat_lambda: Optional[float]
    :param pseudo_thresh: Pseudo label threshold
    :type pseudo_thresh: int
    :param obj_unc: Object uncertainty
    :type obj_unc: bool
    :param rel_unc: Relation uncertainty
    :type rel_unc: bool
    :param obj_loss_weighting: Object loss weighting
    :type obj_loss_weighting: Optional[str]
    :param rel_loss_weighting: Relation loss weighting
    :type rel_loss_weighting: Optional[str]
    :param mlm: Masked language modeling
    :type mlm: bool
    :param eos_coef: End-of-sequence coefficient
    :type eos_coef: float
    :param obj_con_loss: Object consistency loss
    :type obj_con_loss: Optional[str]
    :param lambda_con: Consistency loss coefficient
    :type lambda_con: float
    :param tracking: Enable tracking
    :type tracking: bool
    :param mem_size: Memory size
    :type mem_size: int
    :param mem_dim: Memory dimension
    :type mem_dim: int
    :param mem_update_rate: Memory update rate
    :type mem_update_rate: float
    :param mem_temperature: Memory temperature
    :type mem_temperature: float
    :param use_memory_attention: Use memory attention
    :type use_memory_attention: bool
    :param memory_dropout: Memory dropout rate
    :type memory_dropout: float
    :param gmm_components: Number of GMM components
    :type gmm_components: int
    :param gmm_covariance_type: GMM covariance type
    :type gmm_covariance_type: str
    :param gmm_reg_covar: GMM regularization covariance
    :type gmm_reg_covar: float
    :param use_prior: Use prior knowledge
    :type use_prior: bool
    :param prior_weight: Prior weight
    :type prior_weight: float
    """

    # Model identification
    model_type: str = "tempura"
    
    # GMM head parameters
    obj_head: str = "gmm"
    rel_head: str = "gmm"
    K: int = 4
    gmm_components: int = 4
    gmm_covariance_type: str = "full"
    gmm_reg_covar: float = 1e-6
    
    # Memory parameters
    rel_mem_compute: Optional[str] = None
    obj_mem_compute: bool = False
    take_obj_mem_feat: bool = False
    obj_mem_weight_type: str = "simple"
    rel_mem_weight_type: str = "simple"
    mem_feat_selection: str = "manual"
    mem_fusion: str = "early"
    mem_feat_lambda: Optional[float] = None
    mem_size: int = 1000
    mem_dim: int = 256
    mem_update_rate: float = 0.1
    mem_temperature: float = 1.0
    use_memory_attention: bool = True
    memory_dropout: float = 0.1
    
    # Uncertainty parameters
    pseudo_thresh: int = 7
    obj_unc: bool = False
    rel_unc: bool = False
    
    # Loss weighting
    obj_loss_weighting: Optional[str] = None
    rel_loss_weighting: Optional[str] = None
    
    # Additional parameters
    mlm: bool = False
    eos_coef: float = 1
    obj_con_loss: Optional[str] = None
    lambda_con: float = 1
    tracking: bool = True
    
    # Prior knowledge
    use_prior: bool = False
    prior_weight: float = 0.1


@dataclass
class TempuraTrainingConfig(TrainingConfig):
    """Tempura-specific training configuration.

    :param memory_warmup_epochs: Number of epochs for memory warmup
    :type memory_warmup_epochs: int
    :param memory_lr: Learning rate for memory parameters
    :type memory_lr: float
    :param gmm_lr: Learning rate for GMM parameters
    :type gmm_lr: float
    :param use_memory_scheduler: Use separate scheduler for memory
    :type use_memory_scheduler: bool
    :param memory_decay: Memory decay rate
    :type memory_decay: float
    :param use_curriculum_learning: Use curriculum learning
    :type use_curriculum_learning: bool
    :param curriculum_epochs: Number of epochs for curriculum
    :type curriculum_epochs: int
    :param use_memory_regularization: Use memory regularization
    :type use_memory_regularization: bool
    :param memory_reg_weight: Memory regularization weight
    :type memory_reg_weight: float
    """

    memory_warmup_epochs: int = 5
    memory_lr: float = 1e-4
    gmm_lr: float = 1e-3
    use_memory_scheduler: bool = True
    memory_decay: float = 0.99
    use_curriculum_learning: bool = False
    curriculum_epochs: int = 10
    use_memory_regularization: bool = True
    memory_reg_weight: float = 0.01


@dataclass
class TempuraLossConfig(LossConfig):
    """Tempura-specific loss configuration.

    :param obj_loss_weight: Weight for object classification loss
    :type obj_loss_weight: float
    :param rel_loss_weight: Weight for relation classification loss
    :type rel_loss_weight: float
    :param memory_loss_weight: Weight for memory loss
    :type memory_loss_weight: float
    :param gmm_loss_weight: Weight for GMM loss
    :type gmm_loss_weight: float
    :param consistency_loss_weight: Weight for consistency loss
    :type consistency_loss_weight: float
    :param uncertainty_loss_weight: Weight for uncertainty loss
    :type uncertainty_loss_weight: float
    :param use_memory_loss: Use memory loss
    :type use_memory_loss: bool
    :param use_gmm_loss: Use GMM loss
    :type use_gmm_loss: bool
    :param use_consistency_loss: Use consistency loss
    :type use_consistency_loss: bool
    :param use_uncertainty_loss: Use uncertainty loss
    :type use_uncertainty_loss: bool
    :param memory_loss_type: Type of memory loss
    :type memory_loss_type: str
    :param gmm_loss_type: Type of GMM loss
    :type gmm_loss_type: str
    :param consistency_loss_type: Type of consistency loss
    :type consistency_loss_type: str
    :param uncertainty_loss_type: Type of uncertainty loss
    :type uncertainty_loss_type: str
    """

    obj_loss_weight: float = 1.0
    rel_loss_weight: float = 1.0
    memory_loss_weight: float = 0.1
    gmm_loss_weight: float = 0.1
    consistency_loss_weight: float = 0.1
    uncertainty_loss_weight: float = 0.1
    
    use_memory_loss: bool = True
    use_gmm_loss: bool = True
    use_consistency_loss: bool = True
    use_uncertainty_loss: bool = True
    
    memory_loss_type: str = "mse"
    gmm_loss_type: str = "nll"
    consistency_loss_type: str = "mse"
    uncertainty_loss_type: str = "kl"


@dataclass
class TempuraDataConfig(DataConfig):
    """Tempura-specific data configuration.

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
    :param use_memory_sampling: Use memory sampling
    :type use_memory_sampling: bool
    :param memory_sampling_ratio: Memory sampling ratio
    :type memory_sampling_ratio: float
    :param use_negative_sampling: Use negative sampling
    :type use_negative_sampling: bool
    :param negative_ratio: Ratio of negative samples
    :type negative_ratio: float
    :param use_pseudo_labels: Use pseudo labels
    :type use_pseudo_labels: bool
    :param pseudo_label_threshold: Pseudo label threshold
    :type pseudo_label_threshold: float
    :param use_uncertainty_sampling: Use uncertainty sampling
    :type use_uncertainty_sampling: bool
    :param uncertainty_threshold: Uncertainty threshold
    :type uncertainty_threshold: float
    """

    max_objects: int = 50
    max_relations: int = 100
    max_frames: int = 30
    use_temporal_sampling: bool = True
    temporal_stride: int = 1
    use_memory_sampling: bool = True
    memory_sampling_ratio: float = 0.1
    use_negative_sampling: bool = True
    negative_ratio: float = 0.5
    use_pseudo_labels: bool = False
    pseudo_label_threshold: float = 0.7
    use_uncertainty_sampling: bool = False
    uncertainty_threshold: float = 0.5
