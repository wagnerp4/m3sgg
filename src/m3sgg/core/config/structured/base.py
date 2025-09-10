"""Base structured configuration classes for M3SGG models.

This module provides the base configuration classes and common structures
used across all model types in the M3SGG framework.

:author: M3SGG Team
:version: 0.1.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class BaseConfig:
    """Base configuration class with common parameters for all models.

    This class defines the common configuration parameters that are shared
    across all model types in the M3SGG framework.

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
    :param fraction: Fraction of dataset to use (1=all, 2=half, etc.)
    :type fraction: int
    :param ckpt: Checkpoint path
    :type ckpt: Optional[str]
    :param optimizer: Optimizer type (adamw/adam/sgd)
    :type optimizer: str
    :param lr: Learning rate
    :type lr: float
    :param nepoch: Number of epochs
    :type nepoch: float
    :param niter: Number of iterations for iterative training
    :type niter: Optional[int]
    :param eval_frequency: Evaluation frequency for iterative training
    :type eval_frequency: int
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
    :param num_workers: Number of DataLoader workers
    :type num_workers: int
    :param model_type: Model type identifier
    :type model_type: str
    :param use_matcher: Use Hungarian matcher (for DSG-DETR)
    :type use_matcher: bool
    :param eval: Evaluation mode
    :type eval: bool
    """

    # Core training parameters
    mode: str = "predcls"
    save_path: str = "output"
    model_path: str = "weights/predcls.tar"
    dataset: str = "action_genome"
    data_path: str = "data/action_genome"
    datasize: str = "large"
    fraction: int = 1
    ckpt: Optional[str] = None
    optimizer: str = "adamw"
    lr: float = 1e-5
    nepoch: float = 10
    niter: Optional[int] = None
    eval_frequency: int = 50

    # Model architecture parameters
    enc_layer: int = 1
    dec_layer: int = 3
    bce_loss: bool = False

    # System parameters
    device: str = "cuda:0"
    seed: int = 42
    num_workers: int = 0

    # Model identification
    model_type: str = "sttran"
    use_matcher: bool = False
    eval: bool = False


@dataclass
class TrainingConfig:
    """Training-specific configuration parameters.

    :param batch_size: Training batch size
    :type batch_size: int
    :param val_batch_size: Validation batch size
    :type val_batch_size: int
    :param max_grad_norm: Maximum gradient norm for clipping
    :type max_grad_norm: float
    :param warmup_epochs: Number of warmup epochs
    :type warmup_epochs: int
    :param early_stopping_patience: Early stopping patience
    :type early_stopping_patience: int
    :param save_frequency: Model saving frequency (epochs)
    :type save_frequency: int
    :param log_frequency: Logging frequency (iterations)
    :type log_frequency: int
    :param val_frequency: Validation frequency (epochs)
    :type val_frequency: int
    """

    batch_size: int = 1
    val_batch_size: int = 1
    max_grad_norm: float = 1.0
    warmup_epochs: int = 0
    early_stopping_patience: int = 10
    save_frequency: int = 1
    log_frequency: int = 10
    val_frequency: int = 1


@dataclass
class DataConfig:
    """Data-specific configuration parameters.

    :param cache_dir: Directory for caching data
    :type cache_dir: str
    :param pin_memory: Pin memory for DataLoader
    :type pin_memory: bool
    :param shuffle: Shuffle training data
    :type shuffle: bool
    :param drop_last: Drop last incomplete batch
    :type drop_last: bool
    :param prefetch_factor: DataLoader prefetch factor
    :type prefetch_factor: int
    :param persistent_workers: Use persistent workers
    :type persistent_workers: bool
    """

    cache_dir: str = "data/cache"
    pin_memory: bool = False
    shuffle: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration parameters.

    :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    :type log_level: str
    :param log_file: Log file path
    :type log_file: Optional[str]
    :param tensorboard_dir: TensorBoard log directory
    :type tensorboard_dir: Optional[str]
    :param wandb_project: Weights & Biases project name
    :type wandb_project: Optional[str]
    :param wandb_entity: Weights & Biases entity name
    :type wandb_entity: Optional[str]
    :param log_gradients: Log gradient norms
    :type log_gradients: bool
    :param log_weights: Log weight histograms
    :type log_weights: bool
    """

    log_level: str = "INFO"
    log_file: Optional[str] = None
    tensorboard_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    log_gradients: bool = False
    log_weights: bool = False


@dataclass
class CheckpointConfig:
    """Checkpoint configuration parameters.

    :param save_dir: Directory to save checkpoints
    :type save_dir: str
    :param save_best: Save best model only
    :type save_best: bool
    :param save_last: Save last model
    :type save_last: bool
    :param save_frequency: Save frequency (epochs)
    :type save_frequency: int
    :param max_checkpoints: Maximum number of checkpoints to keep
    :type max_checkpoints: int
    :param checkpoint_metric: Metric to use for best model selection
    :type checkpoint_metric: str
    :param checkpoint_mode: Mode for metric comparison (min/max)
    :type checkpoint_mode: str
    """

    save_dir: str = "checkpoints"
    save_best: bool = True
    save_last: bool = True
    save_frequency: int = 1
    max_checkpoints: int = 5
    checkpoint_metric: str = "recall@20"
    checkpoint_mode: str = "max"


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters.

    :param iou_threshold: IoU threshold for evaluation
    :type iou_threshold: float
    :param constraint: Constraint type for evaluation
    :type constraint: str
    :param save_predictions: Save predictions to file
    :type save_predictions: bool
    :param predictions_dir: Directory to save predictions
    :type predictions_dir: str
    :param eval_metrics: List of metrics to compute
    :type eval_metrics: List[str]
    :param eval_frequency: Evaluation frequency (epochs)
    :type eval_frequency: int
    """

    iou_threshold: float = 0.5
    constraint: str = "with"
    save_predictions: bool = True
    predictions_dir: str = "predictions"
    eval_metrics: List[str] = field(
        default_factory=lambda: ["recall@20", "mean_recall@20"]
    )
    eval_frequency: int = 1


@dataclass
class ModelConfig:
    """Model-specific configuration parameters.

    :param hidden_dim: Hidden dimension size
    :type hidden_dim: int
    :param num_heads: Number of attention heads
    :type num_heads: int
    :param num_layers: Number of transformer layers
    :type num_layers: int
    :param dropout: Dropout rate
    :type dropout: float
    :param activation: Activation function
    :type activation: str
    :param norm_type: Normalization type
    :type norm_type: str
    :param use_bias: Use bias in linear layers
    :type use_bias: bool
    """

    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    activation: str = "relu"
    norm_type: str = "layer_norm"
    use_bias: bool = True


# Common loss configuration
@dataclass
class LossConfig:
    """Loss function configuration parameters.

    :param loss_weights: Weights for different loss components
    :type loss_weights: Dict[str, float]
    :param label_smoothing: Label smoothing factor
    :type label_smoothing: float
    :param focal_alpha: Focal loss alpha parameter
    :type focal_alpha: float
    :param focal_gamma: Focal loss gamma parameter
    :type focal_gamma: float
    :param class_weights: Class weights for imbalanced datasets
    :type class_weights: Optional[Dict[str, float]]
    """

    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "obj_loss": 1.0,
            "rel_loss": 1.0,
            "bbox_loss": 1.0,
            "giou_loss": 1.0,
        }
    )
    label_smoothing: float = 0.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    class_weights: Optional[Dict[str, float]] = None


# Model configuration registry
MODEL_CONFIGS = {
    "sttran": "STTRANConfig",
    "stket": "STKETConfig",
    "tempura": "TempuraConfig",
    "EASG": "EASGConfig",
    "scenellm": "SceneLLMConfig",
    "oed": "OEDConfig",
    "vlm": "VLMConfig",
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

    # Import the specific config class
    if model_type == "sttran":
        from .sttran import STTRANConfig

        return STTRANConfig
    elif model_type == "stket":
        from .stket import STKETConfig

        return STKETConfig
    elif model_type == "tempura":
        from .tempura import TempuraConfig

        return TempuraConfig
    elif model_type == "EASG":
        from .easg import EASGConfig

        return EASGConfig
    elif model_type == "scenellm":
        from .scenellm import SceneLLMConfig

        return SceneLLMConfig
    elif model_type == "oed":
        from .oed import OEDConfig

        return OEDConfig
    elif model_type == "vlm":
        from .vlm import VLMConfig

        return VLMConfig
