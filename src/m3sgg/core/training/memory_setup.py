"""
Memory setup for TEMPURA models.

This module contains the memory computation logic that was extracted from
the monolithic training.py script to improve modularity and maintainability.
"""

import logging
from typing import Dict, Optional, Tuple

import torch

from m3sgg.utils.memory import memory_computation
from m3sgg.utils.uncertainty import uncertainty_values


class MemorySetup:
    """Class for setting up memory computation for TEMPURA models.
    
    This class handles memory computation logic that was extracted from the
    monolithic training script to improve modularity and maintainability.
    It manages uncertainty values computation, memory tensor creation, and
    memory assignment to TEMPURA models for improved performance.
    
    :param config: Configuration object containing memory parameters
    :type config: Config
    :param model: Model instance for extracting class information
    :type model: torch.nn.Module
    :param device: Device to place memory tensors on
    :type device: torch.device
    :param logger: Optional logger instance
    :type logger: Optional[logging.Logger]
    """
    
    def __init__(self, config, model, device, logger: Optional[logging.Logger] = None):
        """Initialize the memory setup.
        
        :param config: Configuration object containing memory parameters
        :type config: Config
        :param model: Model instance for extracting class information
        :type model: torch.nn.Module
        :param device: Device to place memory tensors on
        :type device: torch.device
        :param logger: Optional logger instance
        :type logger: Optional[logging.Logger]
        """
        self.config = config
        self.model = model
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
    
    def setup_memory(self) -> bool:
        """Set up memory computation for TEMPURA model if required.
        
        :return: True if memory was set up, False otherwise
        :rtype: bool
        """
        if not (self.config.model_type == "tempura" and 
                (self.config.rel_mem_compute or self.config.obj_mem_compute)):
            return False
        
        self.logger.info("Computing memory for TEMPURA model")
        
        # Initialize uncertainty values for memory computation
        unc_vals = uncertainty_values(
            obj_classes=len(self.model.obj_classes),
            attention_class_num=self.model.attention_class_num,
            spatial_class_num=self.model.spatial_class_num,
            contact_class_num=self.model.contact_class_num,
        )
        
        # Prepare relationship class numbers
        rel_class_num = {
            "attention": self.model.attention_class_num,
            "spatial": self.model.spatial_class_num,
            "contacting": self.model.contact_class_num,
        }
        
        # Determine object feature dimension based on tracking configuration
        if self.config.tracking:
            obj_feature_dim = 2048 + 200 + 128
        else:
            obj_feature_dim = 1024
        
        # Compute memory
        rel_memory, obj_memory = memory_computation(
            unc_vals,
            self.config.save_path,
            rel_class_num,
            len(self.model.obj_classes),
            obj_feature_dim=obj_feature_dim,
            rel_feature_dim=1936,
            obj_weight_type=self.config.obj_mem_weight_type,
            rel_weight_type=self.config.rel_mem_weight_type,
            obj_mem=self.config.obj_mem_compute,
            obj_unc=self.config.obj_unc,
            include_bg_mem=False,
        )
        
        # Set memory in model
        self.model.object_classifier.obj_memory = obj_memory.to(self.device)
        self.model.rel_memory = {k: rel_memory[k].to(self.device) for k in rel_memory.keys()}
        
        self.logger.info("Memory computation completed and set in model")
        return True
    
    def get_memory_info(self) -> Dict[str, int]:
        """Get information about the computed memory.
        
        :return: Dictionary containing memory information
        :rtype: Dict[str, int]
        """
        if not hasattr(self.model, "rel_memory") or not hasattr(self.model.object_classifier, "obj_memory"):
            return {}
        
        info = {
            "obj_memory_shape": list(self.model.object_classifier.obj_memory.shape),
            "rel_memory_keys": list(self.model.rel_memory.keys()),
        }
        
        for key, memory in self.model.rel_memory.items():
            info[f"rel_memory_{key}_shape"] = list(memory.shape)
        
        return info
