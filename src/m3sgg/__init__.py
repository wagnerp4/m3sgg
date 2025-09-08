"""M3SGG: Modular, multi-modal Scene Graph Generation Framework.

A comprehensive framework for training and evaluating video scene graph generation models
based on transformer-based deep learning architectures.
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

__version__ = "0.1.0"
__author__ = "M3SGG Team"

from . import core
from . import datasets
from . import utils
from . import language
__all__ = [
    "core",
    "datasets", 
    "utils",
    "language",
]