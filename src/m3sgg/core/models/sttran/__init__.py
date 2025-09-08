"""
STTran (Spatio-Temporal Transformer) model implementation.

This module provides the STTran model for video scene graph generation.
"""

from .sttran import ObjectClassifier, STTran

__all__ = [
    "ObjectClassifier",
    "STTran",
]
