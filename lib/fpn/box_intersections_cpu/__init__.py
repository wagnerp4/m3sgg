"""
Bounding box operations for CPU computation.
"""

from .bbox import bbox_intersections, bbox_overlaps

__all__ = ["bbox_overlaps", "bbox_intersections"]
