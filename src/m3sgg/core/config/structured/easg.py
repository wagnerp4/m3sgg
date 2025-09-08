"""EASG model configuration classes.

This module provides structured configuration classes specifically
for the EASG model.

:author: M3SGG Team
:version: 0.1.0
"""

from dataclasses import dataclass
from .base import BaseConfig


@dataclass
class EASGConfig(BaseConfig):
    """Configuration for EASG model.

    :param model_type: Model type identifier
    :type model_type: str
    """

    model_type: str = "EASG"
