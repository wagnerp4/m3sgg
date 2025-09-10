"""SceneLLM model configuration classes.

This module provides structured configuration classes specifically
for the SceneLLM model, which combines VQ-VAE with language models.

:author: M3SGG Team
:version: 0.1.0
"""

from dataclasses import dataclass
from typing import Optional

from .base import (
    BaseConfig,
    TrainingConfig,
    DataConfig,
    LoggingConfig,
    CheckpointConfig,
    EvaluationConfig,
    ModelConfig,
    LossConfig,
)


@dataclass
class SceneLLMConfig(BaseConfig):
    """Configuration for SceneLLM model.

    SceneLLM combines VQ-VAE (Vector Quantized Variational AutoEncoder)
    with language models for scene graph generation and text summarization.

    :param model_type: Model type identifier
    :type model_type: str
    :param embed_dim: Embedding dimension for VQ-VAE
    :type embed_dim: int
    :param codebook_size: Size of VQ-VAE codebook
    :type codebook_size: int
    :param commitment_cost: Commitment cost for VQ-VAE
    :type commitment_cost: float
    :param llm_name: LLM model name
    :type llm_name: str
    :param lora_r: LoRA rank
    :type lora_r: int
    :param lora_alpha: LoRA alpha
    :type lora_alpha: int
    :param lora_dropout: LoRA dropout
    :type lora_dropout: float
    :param ot_step: Step size for optimal transport codebook update
    :type ot_step: int
    :param vqvae_epochs: Epochs for VQ-VAE pretraining
    :type vqvae_epochs: int
    :param stage1_iterations: Iterations for stage 1 training
    :type stage1_iterations: int
    :param stage2_iterations: Iterations for stage 2 training
    :type stage2_iterations: int
    :param alpha_obj: Weight for object loss in SceneLLM
    :type alpha_obj: float
    :param alpha_rel: Weight for relation loss in SceneLLM
    :type alpha_rel: float
    :param scenellm_training_stage: SceneLLM training stage
    :type scenellm_training_stage: str
    :param disable_checkpoint_saving: Disable checkpoint saving
    :type disable_checkpoint_saving: bool
    :param vqvae_hidden_dim: VQ-VAE hidden dimension
    :type vqvae_hidden_dim: int
    :param vqvae_num_layers: VQ-VAE number of layers
    :type vqvae_num_layers: int
    :param vqvae_num_resblocks: VQ-VAE number of residual blocks
    :type vqvae_num_resblocks: int
    :param vqvae_dropout: VQ-VAE dropout rate
    :type vqvae_dropout: float
    :param vqvae_use_attention: VQ-VAE use attention
    :type vqvae_use_attention: bool
    :param vqvae_attention_heads: VQ-VAE attention heads
    :type vqvae_attention_heads: int
    :param llm_max_length: LLM maximum sequence length
    :type llm_max_length: int
    :param llm_temperature: LLM temperature for generation
    :type llm_temperature: float
    :param llm_top_p: LLM top-p sampling
    :type llm_top_p: float
    :param llm_top_k: LLM top-k sampling
    :type llm_top_k: int
    :param llm_repetition_penalty: LLM repetition penalty
    :type llm_repetition_penalty: float
    :param llm_do_sample: LLM do sampling
    :type llm_do_sample: bool
    :param llm_pad_token_id: LLM pad token ID
    :type llm_pad_token_id: int
    :param llm_eos_token_id: LLM EOS token ID
    :type llm_eos_token_id: int
    :param llm_bos_token_id: LLM BOS token ID
    :type llm_bos_token_id: int
    :param use_peft: Use PEFT (Parameter Efficient Fine-Tuning)
    :type use_peft: bool
    :param peft_config: PEFT configuration
    :type peft_config: Optional[dict]
    :param use_gradient_checkpointing: Use gradient checkpointing
    :type use_gradient_checkpointing: bool
    :param use_flash_attention: Use flash attention
    :type use_flash_attention: bool
    :param use_8bit_optimizer: Use 8-bit optimizer
    :type use_8bit_optimizer: bool
    :param use_4bit_quantization: Use 4-bit quantization
    :type use_4bit_quantization: bool
    """

    # Model identification
    model_type: str = "scenellm"

    # VQ-VAE parameters
    embed_dim: int = 1024
    codebook_size: int = 8192
    commitment_cost: float = 0.25
    vqvae_hidden_dim: int = 512
    vqvae_num_layers: int = 4
    vqvae_num_resblocks: int = 2
    vqvae_dropout: float = 0.1
    vqvae_use_attention: bool = True
    vqvae_attention_heads: int = 8

    # LLM parameters
    llm_name: str = "google/gemma-2-2b"
    llm_max_length: int = 512
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_top_k: int = 50
    llm_repetition_penalty: float = 1.1
    llm_do_sample: bool = True
    llm_pad_token_id: int = 0
    llm_eos_token_id: int = 1
    llm_bos_token_id: int = 2

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training parameters
    ot_step: int = 512
    vqvae_epochs: int = 5
    stage1_iterations: int = 30000
    stage2_iterations: int = 50000
    alpha_obj: float = 1.0
    alpha_rel: float = 1.0
    scenellm_training_stage: str = "vqvae"
    disable_checkpoint_saving: bool = False

    # Optimization parameters
    use_peft: bool = True
    peft_config: Optional[dict] = None
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = False
    use_8bit_optimizer: bool = False
    use_4bit_quantization: bool = False


@dataclass
class SceneLLMTrainingConfig(TrainingConfig):
    """SceneLLM-specific training configuration.

    :param vqvae_lr: Learning rate for VQ-VAE
    :type vqvae_lr: float
    :param llm_lr: Learning rate for LLM
    :type llm_lr: float
    :param vqvae_weight_decay: Weight decay for VQ-VAE
    :type vqvae_weight_decay: float
    :param llm_weight_decay: Weight decay for LLM
    :type llm_weight_decay: float
    :param use_warmup: Use learning rate warmup
    :type use_warmup: bool
    :param warmup_steps: Number of warmup steps
    :type warmup_steps: int
    :param use_cosine_schedule: Use cosine learning rate schedule
    :type use_cosine_schedule: bool
    :param cosine_min_lr: Minimum learning rate for cosine schedule
    :type cosine_min_lr: float
    :param use_linear_schedule: Use linear learning rate schedule
    :type use_linear_schedule: bool
    :param linear_min_lr: Minimum learning rate for linear schedule
    :type linear_min_lr: float
    :param use_adafactor: Use Adafactor optimizer
    :type use_adafactor: bool
    :param adafactor_scale_parameter: Adafactor scale parameter
    :type adafactor_scale_parameter: bool
    :param adafactor_relative_step_size: Adafactor relative step size
    :type adafactor_relative_step_size: bool
    :param adafactor_warmup_init: Adafactor warmup init
    :type adafactor_warmup_init: bool
    :param use_dataloader_pin_memory: Use DataLoader pin memory
    :type use_dataloader_pin_memory: bool
    :param dataloader_num_workers: DataLoader number of workers
    :type dataloader_num_workers: int
    :param dataloader_prefetch_factor: DataLoader prefetch factor
    :type dataloader_prefetch_factor: int
    :param use_mixed_precision: Use mixed precision training
    :type use_mixed_precision: bool
    :param mixed_precision_backend: Mixed precision backend
    :type mixed_precision_backend: str
    :param mixed_precision_loss_scale: Mixed precision loss scale
    :type mixed_precision_loss_scale: str
    :param use_gradient_accumulation: Use gradient accumulation
    :type use_gradient_accumulation: bool
    :param gradient_accumulation_steps: Gradient accumulation steps
    :type gradient_accumulation_steps: int
    :param use_gradient_clipping: Use gradient clipping
    :type use_gradient_clipping: bool
    :param max_grad_norm: Maximum gradient norm
    :type max_grad_norm: float
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
    :param use_early_stopping: Use early stopping
    :type use_early_stopping: bool
    :param early_stopping_patience: Early stopping patience
    :type early_stopping_patience: int
    :param early_stopping_min_delta: Early stopping minimum delta
    :type early_stopping_min_delta: float
    :param early_stopping_monitor: Early stopping monitor metric
    :type early_stopping_monitor: str
    :param early_stopping_mode: Early stopping mode
    :type early_stopping_mode: str
    """

    vqvae_lr: float = 1e-4
    llm_lr: float = 2e-5
    vqvae_weight_decay: float = 1e-4
    llm_weight_decay: float = 0.01
    use_warmup: bool = True
    warmup_steps: int = 1000
    use_cosine_schedule: bool = True
    cosine_min_lr: float = 1e-7
    use_linear_schedule: bool = False
    linear_min_lr: float = 1e-7
    use_adafactor: bool = False
    adafactor_scale_parameter: bool = True
    adafactor_relative_step_size: bool = True
    adafactor_warmup_init: bool = False
    use_dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int = 2
    use_mixed_precision: bool = True
    mixed_precision_backend: str = "apex"
    mixed_precision_loss_scale: str = "dynamic"
    use_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    use_ema: bool = False
    ema_decay: float = 0.999
    use_swa: bool = False
    swa_lr: float = 1e-5
    swa_epochs: int = 5
    use_early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "min"


@dataclass
class SceneLLMLossConfig(LossConfig):
    """SceneLLM-specific loss configuration.

    :param vqvae_loss_weight: Weight for VQ-VAE loss
    :type vqvae_loss_weight: float
    :param llm_loss_weight: Weight for LLM loss
    :type llm_loss_weight: float
    :param obj_loss_weight: Weight for object loss
    :type obj_loss_weight: float
    :param rel_loss_weight: Weight for relation loss
    :type rel_loss_weight: float
    :param commitment_loss_weight: Weight for commitment loss
    :type commitment_loss_weight: float
    :param perceptual_loss_weight: Weight for perceptual loss
    :type perceptual_loss_weight: float
    :param use_commitment_loss: Use commitment loss
    :type use_commitment_loss: bool
    :param use_perceptual_loss: Use perceptual loss
    :type use_perceptual_loss: bool
    :param use_kl_loss: Use KL divergence loss
    :type use_kl_loss: bool
    :param kl_loss_weight: Weight for KL divergence loss
    :type kl_loss_weight: float
    :param use_contrastive_loss: Use contrastive loss
    :type use_contrastive_loss: bool
    :param contrastive_loss_weight: Weight for contrastive loss
    :type contrastive_loss_weight: float
    :param contrastive_temperature: Contrastive loss temperature
    :type contrastive_temperature: float
    :param use_consistency_loss: Use consistency loss
    :type use_consistency_loss: bool
    :param consistency_loss_weight: Weight for consistency loss
    :type consistency_loss_weight: float
    :param use_regularization_loss: Use regularization loss
    :type use_regularization_loss: bool
    :param regularization_loss_weight: Weight for regularization loss
    :type regularization_loss_weight: float
    :param use_auxiliary_loss: Use auxiliary loss
    :type use_auxiliary_loss: bool
    :param auxiliary_loss_weight: Weight for auxiliary loss
    :type auxiliary_loss_weight: float
    """

    vqvae_loss_weight: float = 1.0
    llm_loss_weight: float = 1.0
    obj_loss_weight: float = 1.0
    rel_loss_weight: float = 1.0
    commitment_loss_weight: float = 0.25
    perceptual_loss_weight: float = 0.1
    use_commitment_loss: bool = True
    use_perceptual_loss: bool = False
    use_kl_loss: bool = False
    kl_loss_weight: float = 0.1
    use_contrastive_loss: bool = False
    contrastive_loss_weight: float = 0.1
    contrastive_temperature: float = 0.07
    use_consistency_loss: bool = False
    consistency_loss_weight: float = 0.1
    use_regularization_loss: bool = False
    regularization_loss_weight: float = 0.01
    use_auxiliary_loss: bool = False
    auxiliary_loss_weight: float = 0.1


@dataclass
class SceneLLMDataConfig(DataConfig):
    """SceneLLM-specific data configuration.

    :param max_objects: Maximum number of objects per frame
    :type max_objects: int
    :param max_relations: Maximum number of relations per frame
    :type max_relations: int
    :param max_frames: Maximum number of frames per video
    :type max_frames: int
    :param max_text_length: Maximum text length
    :type max_text_length: int
    :param use_text_augmentation: Use text augmentation
    :type use_text_augmentation: bool
    :param text_augmentation_prob: Text augmentation probability
    :type text_augmentation_prob: float
    :param use_image_augmentation: Use image augmentation
    :type use_image_augmentation: bool
    :param image_augmentation_prob: Image augmentation probability
    :type image_augmentation_prob: float
    :param use_temporal_augmentation: Use temporal augmentation
    :type use_temporal_augmentation: bool
    :param temporal_augmentation_prob: Temporal augmentation probability
    :type temporal_augmentation_prob: float
    :param use_negative_sampling: Use negative sampling
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
    max_text_length: int = 512
    use_text_augmentation: bool = False
    text_augmentation_prob: float = 0.5
    use_image_augmentation: bool = False
    image_augmentation_prob: float = 0.5
    use_temporal_augmentation: bool = False
    temporal_augmentation_prob: float = 0.5
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
