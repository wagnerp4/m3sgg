"""
Loss factory for creating different loss functions based on model type and configuration.

This module contains the loss function setup logic that was extracted from
the monolithic training.py script to improve modularity and maintainability.
"""

import logging
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from m3sgg.utils.infoNCE import EucNormLoss, SupConLoss


class LossFactory:
    """Factory class for creating loss functions based on configuration.
    
    This factory provides a unified interface for creating different loss functions
    based on model type and configuration. It handles loss function setup logic
    that was extracted from the monolithic training script to improve modularity
    and maintainability. Supports basic losses for all models and model-specific
    losses for TEMPURA models.
    
    :param config: Configuration object containing loss parameters
    :type config: Config
    :param model: Model instance for extracting class information
    :type model: torch.nn.Module
    :param device: Device to place loss functions on
    :type device: torch.device
    :param logger: Optional logger instance
    :type logger: Optional[logging.Logger]
    """
    
    def __init__(self, config, model, device, logger: Optional[logging.Logger] = None):
        """Initialize the loss factory.
        
        :param config: Configuration object containing loss parameters
        :type config: Config
        :param model: Model instance for extracting class information
        :type model: torch.nn.Module
        :param device: Device to place loss functions on
        :type device: torch.device
        :param logger: Optional logger instance
        :type logger: Optional[logging.Logger]
        """
        self.config = config
        self.model = model
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
    
    def create_losses(self) -> Dict[str, Union[nn.Module, None]]:
        """Create loss functions based on the configuration.
        
        :return: Dictionary containing loss functions
        :rtype: Dict[str, Union[nn.Module, None]]
        """
        losses = {}
        
        # Basic losses for all models
        if self.config.bce_loss:
            losses["ce_loss"] = nn.CrossEntropyLoss()
            losses["bce_loss"] = nn.BCELoss()
        else:
            losses["ce_loss"] = nn.CrossEntropyLoss()
            losses["mlm_loss"] = nn.MultiLabelMarginLoss()
        
        # Model-specific losses
        if self.config.model_type == "tempura":
            tempura_losses = self._create_tempura_losses()
            losses.update(tempura_losses)
        
        return losses
    
    def _create_tempura_losses(self) -> Dict[str, Union[nn.Module, None]]:
        """Create loss functions specific to TEMPURA model.
        
        :return: Dictionary containing TEMPURA-specific loss functions
        :rtype: Dict[str, Union[nn.Module, None]]
        """
        losses = {}
        
        # Object classification loss
        weights = torch.ones(len(self.model.obj_classes))
        weights[0] = self.config.eos_coef
        
        if self.config.obj_head != "gmm":
            losses["ce_loss_obj"] = nn.CrossEntropyLoss(
                weight=weights.to(device=self.device), reduction="none"
            )
        else:
            losses["ce_loss_obj"] = nn.NLLLoss(
                weight=weights.to(device=self.device), reduction="none"
            )
        
        # Relation classification loss
        if self.config.rel_head != "gmm":
            losses["ce_loss_rel"] = nn.CrossEntropyLoss(reduction="none")
        else:
            losses["ce_loss_rel"] = nn.NLLLoss(reduction="none")
        
        # Multi-label or binary cross-entropy loss
        if self.config.mlm:
            losses["mlm_loss"] = nn.MultiLabelMarginLoss(reduction="none")
        else:
            losses["bce_loss"] = nn.BCELoss(reduction="none")
        
        # Contrastive loss for object features
        if self.config.obj_con_loss == "euc_con":
            losses["con_loss"] = EucNormLoss()
            losses["con_loss"].train()
        elif self.config.obj_con_loss == "info_nce":
            losses["con_loss"] = SupConLoss(temperature=0.1)
            losses["con_loss"].train()
        else:
            losses["con_loss"] = None
        
        self.logger.info("Created TEMPURA-specific loss functions")
        return losses
