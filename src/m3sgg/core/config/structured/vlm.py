"""VLM model configuration classes.

This module provides structured configuration classes specifically
for VLM (Vision-Language Model) based scene graph generation.

:author: M3SGG Team
:version: 0.1.0
"""

from dataclasses import dataclass
from typing import Optional

from .base import BaseConfig


@dataclass
class VLMConfig(BaseConfig):
    """Configuration for VLM model.

    :param model_type: Model type identifier
    :type model_type: str
    :param vlm_model_name: HuggingFace model name for VLM
    :type vlm_model_name: str
    :param vlm_use_chain_of_thought: Use chain-of-thought reasoning
    :type vlm_use_chain_of_thought: bool
    :param vlm_use_tree_of_thought: Use tree-of-thought reasoning
    :type vlm_use_tree_of_thought: bool
    :param vlm_confidence_threshold: Confidence threshold for detection
    :type vlm_confidence_threshold: float
    :param vlm_temperature: Temperature for text generation
    :type vlm_temperature: float
    :param vlm_top_p: Top-p sampling for generation
    :type vlm_top_p: float
    :param vlm_max_new_tokens: Maximum new tokens for generation
    :type vlm_max_new_tokens: int
    """

    model_type: str = "vlm"
    vlm_model_name: str = "apple/FastVLM-0.5B"
    vlm_use_chain_of_thought: bool = True
    vlm_use_tree_of_thought: bool = False
    vlm_confidence_threshold: float = 0.5
    vlm_temperature: float = 0.7
    vlm_top_p: float = 0.9
    vlm_max_new_tokens: int = 512
