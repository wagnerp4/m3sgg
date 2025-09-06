"""Checkpoint utility functions for VidSgg training.

This module provides utilities for safe checkpoint saving with disk space validation
and checkpoint configuration based on available storage.

:author: VidSgg Team
:version: 0.1.0
"""

import logging
import os
import shutil
from typing import Tuple, Optional, Dict, Any

import torch


def check_disk_space_and_configure_checkpointing(save_path: str, logger: logging.Logger, conf) -> Tuple[bool, str]:
    """
    Check available disk space and configure checkpoint saving strategy.
    
    :param save_path: Path where checkpoints will be saved
    :type save_path: str
    :param logger: Logger instance for output
    :type logger: logging.Logger
    :param conf: Configuration object
    :type conf: Config
    :return: Tuple of (checkpoint_enabled, checkpoint_strategy)
    :rtype: tuple
    """
    # Get disk usage information
    total, used, free = shutil.disk_usage(save_path)
    
    # Convert to GB for readability
    total_gb = total / (1024**3)
    used_gb = used / (1024**3)
    free_gb = free / (1024**3)
    
    logger.info(f"Disk space check - Total: {total_gb:.1f}GB, Used: {used_gb:.1f}GB, Free: {free_gb:.1f}GB")
    
    # Estimate checkpoint size (typically 100-200MB for TEMPURA models)
    estimated_checkpoint_size_mb = 200
    estimated_checkpoint_size_gb = estimated_checkpoint_size_mb / 1024
    
    # Safety margin: require at least 2x the estimated checkpoint size
    required_space_gb = estimated_checkpoint_size_gb * 2
    
    if free_gb < required_space_gb:
        logger.warning("Insufficient disk space for checkpoint saving!")
        logger.warning(f"Required: {required_space_gb:.2f}GB, Available: {free_gb:.1f}GB")
        logger.warning("Disabling checkpoint saving to prevent corruption.")
        
        # Disable checkpoint saving
        conf.disable_checkpoint_saving = True
        return False, "disabled"
    
    elif free_gb < required_space_gb * 2:
        logger.warning("Low disk space detected!")
        logger.warning(f"Available: {free_gb:.1f}GB, Recommended: {required_space_gb * 2:.2f}GB")
        logger.warning("Using conservative checkpoint strategy - only saving best model.")
        
        # Conservative strategy: only save best model, not every epoch
        return True, "conservative"
    
    else:
        logger.info(f"Sufficient disk space available: {free_gb:.1f}GB")
        return True, "full"


def safe_save_checkpoint(
    model: torch.nn.Module, 
    checkpoint_path: str, 
    model_type: str, 
    dataset: str, 
    additional_metadata: Optional[Dict[str, Any]] = None, 
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Safely save checkpoint with disk space validation.
    
    :param model: Model to save
    :type model: torch.nn.Module
    :param checkpoint_path: Path to save checkpoint
    :type checkpoint_path: str
    :param model_type: Type of model being saved
    :type model_type: str
    :param dataset: Dataset name
    :type dataset: str
    :param additional_metadata: Additional metadata to save
    :type additional_metadata: dict, optional
    :param logger: Logger instance for output
    :type logger: logging.Logger, optional
    :return: True if save was successful, False otherwise
    :rtype: bool
    """
    try:
        # Check available space before saving
        total, used, free = shutil.disk_usage(os.path.dirname(checkpoint_path))
        free_gb = free / (1024**3)
        required_space_gb = 0.5  # 500MB safety margin
        
        if free_gb < required_space_gb:
            if logger:
                logger.error(f"Insufficient disk space to save checkpoint: {free_gb:.2f}GB available, {required_space_gb:.2f}GB required")
            return False
        
        # Import the save function
        from lib.model_detector import save_checkpoint_with_metadata
        
        # Save checkpoint
        save_checkpoint_with_metadata(
            model,
            checkpoint_path,
            model_type,
            dataset,
            additional_metadata=additional_metadata
        )
        
        if logger:
            logger.info(f"Checkpoint saved successfully to: {checkpoint_path}")
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to save checkpoint: {str(e)}")
        return False


def validate_checkpoint_file(checkpoint_path: str, logger: Optional[logging.Logger] = None) -> bool:
    """
    Validate a checkpoint file before loading.
    
    :param checkpoint_path: Path to checkpoint file
    :type checkpoint_path: str
    :param logger: Logger instance for output
    :type logger: logging.Logger, optional
    :return: True if checkpoint is valid, False otherwise
    :rtype: bool
    """
    try:
        # Check if file exists
        if not os.path.exists(checkpoint_path):
            if logger:
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return False
        
        # Check file size (should be reasonable for a model checkpoint)
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        if file_size_mb < 10:  # Less than 10MB is suspicious
            if logger:
                logger.warning(f"Checkpoint file is unusually small: {file_size_mb:.1f}MB")
        
        # Try to load the checkpoint to validate structure
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        # Validate checkpoint structure
        if "state_dict" not in ckpt:
            if logger:
                logger.error("Invalid checkpoint format - missing 'state_dict' key")
            return False
        
        if logger:
            logger.info(f"Checkpoint validation successful: {checkpoint_path} ({file_size_mb:.1f}MB)")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Checkpoint validation failed for {checkpoint_path}: {str(e)}")
        return False
