"""M3SGG CLI package.

This package provides command-line interface functionality for the M3SGG framework.

:author: M3SGG Team
:version: 0.1.0
"""

from .main import main
from .train import train_command

__all__ = ["main", "train_command"]
