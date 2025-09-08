"""Core functionality for M3SGG framework.

This module contains the core components including models, detectors, 
configuration management, and evaluation tools.
"""

# Import core submodules
from . import config
from . import models
from . import detectors
from . import evaluation

# Make submodules available
__all__ = [
    "config",
    "models", 
    "detectors",
    "evaluation",
]
