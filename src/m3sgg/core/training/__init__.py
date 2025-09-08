"""Training module for M3SGG.

This module contains the core training functionality including the Trainer class,
evaluation components, and factory classes for scene graph generation models.
The module provides a modular architecture for training various model types
with consistent interfaces and comprehensive logging.

Key Components:
- Trainer: Main training loop and orchestration
- ModelFactory: Creates different model types based on configuration
- LossFactory: Sets up loss functions for different model types
- MemorySetup: Handles memory computation for TEMPURA models
- Evaluator: Evaluation metrics and validation

The factory pattern is used throughout to provide clean separation of concerns
and easy extensibility for new model types and configurations.
"""

from .trainer import Trainer
from .evaluation import Evaluator
from .model_factory import ModelFactory
from .loss_factory import LossFactory
from .memory_setup import MemorySetup

__all__ = [
    "Trainer",
    "Evaluator",
    "ModelFactory",
    "LossFactory",
    "MemorySetup",
]
