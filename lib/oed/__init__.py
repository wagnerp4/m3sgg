"""
OED: One-stage End-to-End Dynamic Scene Graph Generation

This module implements the OED architecture for dynamic scene graph generation
using a DETR-style transformer with cascaded decoders and temporal context aggregation.
The OED models now use the existing object detector instead of their own backbone.
"""

from .oed_multi import OEDMulti
from .oed_single import OEDSingle

__all__ = ["OEDMulti", "OEDSingle"]
