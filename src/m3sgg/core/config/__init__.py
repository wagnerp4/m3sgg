"""Configuration management for M3SGG.

This module provides configuration classes and utilities for managing
model parameters, training settings, and experiment configurations.
"""

# Import config modules
from . import config
from . import unified
from . import modern
from . import legacy

# Make modules available
__all__ = [
    "config",
    "unified", 
    "modern",
    "legacy",
]
