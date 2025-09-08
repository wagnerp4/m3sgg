"""STKET model configuration classes.

This module provides structured configuration classes specifically
for the STKET model.

:author: M3SGG Team
:version: 0.1.0
"""

from dataclasses import dataclass
from .base import BaseConfig


@dataclass
class STKETConfig(BaseConfig):
    """Configuration for STKET model.

    :param model_type: Model type identifier
    :type model_type: str
    :param enc_layer_num: Number of encoder layers
    :type enc_layer_num: int
    :param dec_layer_num: Number of decoder layers
    :type dec_layer_num: int
    :param N_layer: Number of layers
    :type N_layer: int
    :param pred_contact_threshold: Contact prediction threshold
    :type pred_contact_threshold: float
    :param window_size: Window size for temporal processing
    :type window_size: int
    :param use_spatial_prior: Use spatial prior
    :type use_spatial_prior: bool
    :param use_temporal_prior: Use temporal prior
    :type use_temporal_prior: bool
    :param spatial_prior_loss: Use spatial prior loss
    :type spatial_prior_loss: bool
    :param temporal_prior_loss: Use temporal prior loss
    :type temporal_prior_loss: bool
    :param eval: Evaluation mode
    :type eval: bool
    """

    model_type: str = "stket"
    enc_layer_num: int = 1
    dec_layer_num: int = 3
    N_layer: int = 1
    pred_contact_threshold: float = 0.5
    window_size: int = 3
    use_spatial_prior: bool = False
    use_temporal_prior: bool = False
    spatial_prior_loss: bool = False
    temporal_prior_loss: bool = False
    eval: bool = False
