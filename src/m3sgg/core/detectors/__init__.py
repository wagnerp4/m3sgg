"""Object detector implementations for M3SGG.

This module contains object detector implementations including FasterRCNN
and EASG detectors for extracting object features from video frames.
"""

# Import detector modules
from . import faster_rcnn
from . import easg

# Make modules available
__all__ = [
    "faster_rcnn",
    "easg",
]
