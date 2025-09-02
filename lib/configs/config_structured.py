"""Structured configuration classes for VidSgg models using OmegaConf.

This module provides type-safe configuration classes for different model architectures
including STTRAN, STKET, Tempura, EASG, SceneLLM, and OED. These classes are designed
to work with OmegaConf for enhanced configuration management, validation, and interpolation.

:author: VidSgg Team
:version: 0.1.0
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    """Base configuration class with common parameters for all models.

    :param mode: Training mode (predcls/sgcls/sgdet)
    :type mode: str
    :param save_path: Path to save model outputs
    :type save_path: str
    :param model_path: Path to model weights
    :type model_path: str
    :param dataset: Dataset name (action_genome/EASG)
    :type dataset: str
    :param data_path: Path to dataset
    :type data_path: str
    :param datasize: Dataset size (mini/large)
    :type datasize: str
    :param ckpt: Checkpoint path
    :type ckpt: Optional[str]
    :param optimizer: Optimizer type (adamw/adam/sgd)
    :type optimizer: str
    :param lr: Learning rate
    :type lr: float
    :param nepoch: Number of epochs
    :type nepoch: float
    :param enc_layer: Number of encoder layers
    :type enc_layer: int
    :param dec_layer: Number of decoder layers
    :type dec_layer: int
    :param bce_loss: Use BCE loss instead of multi-label margin loss
    :type bce_loss: bool
    :param device: Torch device string (e.g., cuda:0, cpu)
    :type device: str
    :param seed: Global random seed
    :type seed: int
    """

    mode: str = "predcls"
    save_path: str = "output"
    model_path: str = "weights/predcls.tar"
    dataset: str = "action_genome"
    data_path: str = "data/action_genome"
    datasize: str = "large"
    ckpt: Optional[str] = None
    optimizer: str = "adamw"
    lr: float = 1e-5
    nepoch: float = 10
    enc_layer: int = 1
    dec_layer: int = 3
    bce_loss: bool = False
    device: str = "cuda:0"
    seed: int = 42
    num_workers: int = 0


@dataclass
class STTRANConfig(BaseConfig):
    """Configuration for STTRAN model.

    :param model_type: Model type identifier
    :type model_type: str
    """

    model_type: str = "sttran"


@dataclass
class STKETConfig(BaseConfig):
    """Configuration for STKET model.

    :param model_type: Model type identifier
    :type model_type: str
    :param enc_layer_num: Number of encoder layers
    :type enc_layer_num: int
    :param dec_layer_num: Number of decoder layers
    :type dec_layer_num: int
    :param N_layer: Number of layers
    :type N_layer: int
    :param pred_contact_threshold: Contact prediction threshold
    :type pred_contact_threshold: float
    :param window_size: Window size for temporal processing
    :type window_size: int
    :param use_spatial_prior: Use spatial prior
    :type use_spatial_prior: bool
    :param use_temporal_prior: Use temporal prior
    :type use_temporal_prior: bool
    :param spatial_prior_loss: Use spatial prior loss
    :type spatial_prior_loss: bool
    :param temporal_prior_loss: Use temporal prior loss
    :type temporal_prior_loss: bool
    :param eval: Evaluation mode
    :type eval: bool
    """

    model_type: str = "stket"
    enc_layer_num: int = 1
    dec_layer_num: int = 3
    N_layer: int = 1
    pred_contact_threshold: float = 0.5
    window_size: int = 3
    use_spatial_prior: bool = False
    use_temporal_prior: bool = False
    spatial_prior_loss: bool = False
    temporal_prior_loss: bool = False
    eval: bool = False


@dataclass
class TempuraConfig(BaseConfig):
    """Configuration for Tempura model.

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
    """

    model_type: str = "tempura"
    obj_head: str = "gmm"
    rel_head: str = "gmm"
    K: int = 4
    rel_mem_compute: Optional[str] = None
    obj_mem_compute: bool = False
    take_obj_mem_feat: bool = False
    obj_mem_weight_type: str = "simple"
    rel_mem_weight_type: str = "simple"
    mem_feat_selection: str = "manual"
    mem_fusion: str = "early"
    mem_feat_lambda: Optional[float] = None
    pseudo_thresh: int = 7
    obj_unc: bool = False
    rel_unc: bool = False
    obj_loss_weighting: Optional[str] = None
    rel_loss_weighting: Optional[str] = None
    mlm: bool = False
    eos_coef: float = 1
    obj_con_loss: Optional[str] = None
    lambda_con: float = 1
    tracking: bool = True


@dataclass
class EASGConfig(BaseConfig):
    """Configuration for EASG model.

    :param model_type: Model type identifier
    :type model_type: str
    """

    model_type: str = "EASG"


@dataclass
class SceneLLMConfig(BaseConfig):
    """Configuration for SceneLLM model.

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
    """

    model_type: str = "scenellm"
    embed_dim: int = 1024
    codebook_size: int = 8192
    commitment_cost: float = 0.25
    llm_name: str = "google/gemma-2-2b"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    ot_step: int = 512
    vqvae_epochs: int = 5
    stage1_iterations: int = 30000
    stage2_iterations: int = 50000
    alpha_obj: float = 1.0
    alpha_rel: float = 1.0
    scenellm_training_stage: str = "vqvae"
    disable_checkpoint_saving: bool = False


@dataclass
class OEDConfig(BaseConfig):
    """Configuration for OED model.

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
    """

    model_type: str = "oed"
    num_queries: int = 100
    dec_layers_hopd: int = 6
    dec_layers_interaction: int = 6
    num_attn_classes: int = 3
    num_spatial_classes: int = 6
    num_contacting_classes: int = 17
    alpha: float = 0.5
    oed_use_matching: bool = False
    bbox_loss_coef: float = 2.5
    giou_loss_coef: float = 1.0
    obj_loss_coef: float = 1.0
    rel_loss_coef: float = 2.0
    oed_eos_coef: float = 0.1
    interval1: int = 4
    interval2: int = 4
    num_ref_frames: int = 2
    oed_variant: str = "multi"
    fuse_semantic_pos: bool = False
    query_temporal_interaction: bool = False


# Model configuration registry
MODEL_CONFIGS = {
    "sttran": STTRANConfig,
    "stket": STKETConfig,
    "tempura": TempuraConfig,
    "EASG": EASGConfig,
    "scenellm": SceneLLMConfig,
    "oed": OEDConfig,
}


def get_config_class(model_type: str):
    """Get the appropriate configuration class for a model type.

    :param model_type: The model type identifier
    :type model_type: str
    :return: The configuration class for the model type
    :rtype: type
    :raises ValueError: If the model type is not supported
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_type]
