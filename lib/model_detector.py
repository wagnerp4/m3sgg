"""Model type detection utility for VidSgg checkpoints.

This module provides functionality to automatically detect the model type
from a checkpoint's state_dict without requiring explicit model specification.

:author: VidSgg Team
:version: 0.1.0
"""

import time
import torch
from typing import Optional, Dict, Any


def detect_model_type_from_checkpoint(checkpoint_path: str) -> Optional[str]:
    """Detect the model type from a checkpoint file by analyzing state_dict keys.

    This function examines the layer names in the checkpoint's state_dict to
    determine which model architecture was used. Each model has unique layer
    names that serve as fingerprints for identification.

    :param checkpoint_path: Path to the checkpoint file
    :type checkpoint_path: str
    :return: Detected model type or None if detection fails
    :rtype: Optional[str]
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Check for stored metadata first (preferred method)
        if "model_metadata" in checkpoint:
            metadata = checkpoint["model_metadata"]
            if "model_type" in metadata:
                return metadata["model_type"]
        
        # Fallback to layer-based detection
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        # Get all layer names
        layer_names = set(state_dict.keys())
        
        # Model detection based on unique layer signatures
        model_type = _detect_model_from_layer_names(layer_names)
        
        return model_type
        
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None


def _detect_model_from_layer_names(layer_names: set) -> Optional[str]:
    """Detect model type from a set of layer names.

    :param layer_names: Set of layer names from state_dict
    :type layer_names: set
    :return: Detected model type or None
    :rtype: Optional[str]
    """
    
    # SceneLLM detection - has unique VQ-VAE and LLM components
    scenellm_signatures = {
        "quantiser.codebook",
        "llm.model",
        "sia.",
        "llm_input_projection",
        "feature_projection",
        "roi_feature_projection"
    }
    if any(sig in str(layer_names) for sig in scenellm_signatures):
        return "scenellm"
    
    # OED detection - has transformer and query components (more specific)
    oed_signatures = {
        "query_embed.weight",
        "transformer.layers",
        "obj_class_embed.weight",
        "attn_class_embed.weight", 
        "spatial_class_embed.weight",
        "contacting_class_embed.weight",
        "sub_bbox_embed.layers",
        "obj_bbox_embed.layers"
    }
    if any(sig in str(layer_names) for sig in oed_signatures):
        return "oed"
    
    # TEMPURA detection - has GMM components and memory features
    tempura_signatures = {
        "object_classifier.obj_memory",
        "rel_memory",
        "obj_mem_compute",
        "rel_mem_compute",
        "mem_fusion",
        ".heads.mu_",
        ".heads.pi_",
        ".heads.var_",
        "_rel_compress.heads"
    }
    if any(sig in str(layer_names) for sig in tempura_signatures):
        return "tempura"
    
    # STKET detection - has spatial/temporal encoder/decoder structure
    stket_signatures = {
        "spatial_encoder",
        "temporal_decoder", 
        "spatial_attention_distribution",
        "temporal_attention_distribution",
        "ensemble_attention_distribution",
        "spatial_prior_",
        "temporal_prior_"
    }
    if any(sig in str(layer_names) for sig in stket_signatures):
        return "stket"
    
    # EASG detection - has verb-specific components
    easg_signatures = {
        "distribution_verb",
        "labels_verb",
        "edge_distribution",
        "features_verb"
    }
    if any(sig in str(layer_names) for sig in easg_signatures):
        return "sttran_easg"  # EASG uses STTran architecture
    
    # STTran detection - has glocal_transformer and specific components
    sttran_signatures = {
        "glocal_transformer",
        "union_func1.weight",
        "subj_fc.weight",
        "obj_fc.weight", 
        "vr_fc.weight",
        "obj_embed.weight",
        "obj_embed2.weight"
    }
    if any(sig in str(layer_names) for sig in sttran_signatures):
        return "sttran"
    
    # If no specific signatures found, return None
    return None


def get_model_class_from_type(model_type: str):
    """Get the model class from the detected model type.

    :param model_type: Detected model type string
    :type model_type: str
    :return: Model class or None if not found
    :rtype: class or None
    """
    model_class_mapping = {
        "sttran": "STTran",
        "stket": "STKET", 
        "tempura": "TEMPURA",
        "scenellm": "SceneLLM",
        "oed": "OEDMulti",  # Default to multi-frame OED
        "sttran_easg": "STTran_EASG"
    }
    
    return model_class_mapping.get(model_type)


def detect_dataset_from_checkpoint(checkpoint_path: str) -> Optional[str]:
    """Detect the dataset type from a checkpoint file.

    :param checkpoint_path: Path to the checkpoint file
    :type checkpoint_path: str
    :return: Detected dataset type or None
    :rtype: Optional[str]
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        layer_names = set(state_dict.keys())
        
        # EASG has verb-specific components
        if any("verb" in name or "edge_distribution" in name for name in layer_names):
            return "EASG"
        
        # Action Genome has attention/spatial/contact distributions
        if any("attention_distribution" in name or "spatial_distribution" in name for name in layer_names):
            return "action_genome"
            
        return None
        
    except Exception as e:
        print(f"Error detecting dataset from checkpoint {checkpoint_path}: {e}")
        return None


def get_model_info_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Get comprehensive model information from a checkpoint.

    :param checkpoint_path: Path to the checkpoint file
    :type checkpoint_path: str
    :return: Dictionary containing model information
    :rtype: Dict[str, Any]
    """
    info = {
        "model_type": None,
        "dataset": None,
        "model_class": None,
        "checkpoint_path": checkpoint_path
    }
    
    # Detect model type
    model_type = detect_model_type_from_checkpoint(checkpoint_path)
    info["model_type"] = model_type
    
    # Detect dataset
    dataset = detect_dataset_from_checkpoint(checkpoint_path)
    info["dataset"] = dataset
    
    # Get model class
    if model_type:
        info["model_class"] = get_model_class_from_type(model_type)
    
    return info


def save_checkpoint_with_metadata(
    model, 
    save_path: str, 
    model_type: str, 
    dataset: str = None,
    additional_metadata: Dict[str, Any] = None
) -> None:
    """Save model checkpoint with metadata for future identification.

    :param model: The model to save
    :type model: nn.Module
    :param save_path: Path where to save the checkpoint
    :type save_path: str
    :param model_type: Type of the model (e.g., 'sttran', 'tempura', 'scenellm')
    :type model_type: str
    :param dataset: Dataset used for training (e.g., 'action_genome', 'EASG')
    :type dataset: str, optional
    :param additional_metadata: Additional metadata to store
    :type additional_metadata: Dict[str, Any], optional
    """
    metadata = {
        "model_type": model_type,
        "dataset": dataset,
        "timestamp": time.time(),
        "pytorch_version": torch.__version__,
    }
    
    if additional_metadata:
        metadata.update(additional_metadata)
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_metadata": metadata
    }
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint with metadata to: {save_path}")
    print(f"Model type: {model_type}, Dataset: {dataset}")
