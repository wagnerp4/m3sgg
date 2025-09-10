"""Configuration management for M3SGG.

This module provides configuration classes and utilities for managing
model parameters, training settings, and experiment configurations.
"""

# Import config modules
from . import config_omegaconf
from . import config_structured
from . import config_utils

# Make modules available
__all__ = [
    "config_omegaconf",
    "config_structured",
    "config_utils",
]
