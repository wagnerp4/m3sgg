# Credit to the authors of the original code: https://doi.org/10.1016/j.patcog.2025.111992.

"""
SceneLLM main module with imports from distributed components.

This module provides access to all SceneLLM components through a unified interface.
The implementation has been distributed across multiple files for better organization:

- vqvae.py: VQ-VAE quantizer implementation
- sia.py: Spatial Information Aggregator and hierarchical graph functions
- ot.py: Optimal Transport codebook updater
- llm.py: SceneLLM LoRA implementation
- network.py: Main SceneLLM model and SGG decoder

TODO: Compare different clustering methods
TODO: Improve prompt template for LLM
TODO: Add better LLM
TODO: Improve GCN architecture
TODO: Use Cross Entropy instead of MSE
"""

# Import all components from separate modules
from .llm import TRANSFORMERS_AVAILABLE, SceneLLMLoRA
from .network import SceneLLM, SGGDecoder
from .ot import OT_AVAILABLE, OTCodebookUpdater
from .sia import SIA, build_hierarchical_graph
from .vqvae import VQVAEQuantizer

# Re-export all classes for backward compatibility
__all__ = [
    "SceneLLM",
    "VQVAEQuantizer",
    "SIA",
    "OTCodebookUpdater",
    "SceneLLMLoRA",
    "SGGDecoder",
    "build_hierarchical_graph",
    "OT_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
]
