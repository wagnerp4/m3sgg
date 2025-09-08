"""Structured configuration classes for M3SGG models.

This module provides structured configuration classes for all model types
in the M3SGG framework, including base configurations and model-specific
configurations.

:author: M3SGG Team
:version: 0.1.0
"""

# Import from the existing structured directory
from ..structured.base import (
    BaseConfig,
    TrainingConfig,
    DataConfig,
    LoggingConfig,
    CheckpointConfig,
    EvaluationConfig,
    ModelConfig,
    LossConfig,
    get_config_class,
    MODEL_CONFIGS,
)

# Import model-specific configurations
from ..structured.sttran import STTRANConfig
from ..structured.stket import STKETConfig
from ..structured.tempura import TempuraConfig
from ..structured.easg import EASGConfig
from ..structured.scenellm import SceneLLMConfig
from ..structured.oed import OEDConfig
from ..structured.vlm import VLMConfig

# Make all configurations available
__all__ = [
    "BaseConfig",
    "TrainingConfig", 
    "DataConfig",
    "LoggingConfig",
    "CheckpointConfig",
    "EvaluationConfig",
    "ModelConfig",
    "LossConfig",
    "STTRANConfig",
    "STKETConfig",
    "TempuraConfig",
    "EASGConfig",
    "SceneLLMConfig",
    "OEDConfig",
    "VLMConfig",
    "get_config_class",
    "MODEL_CONFIGS",
]
