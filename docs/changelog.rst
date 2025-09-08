Changelog
=========

All notable changes to M3SGG will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~

* Comprehensive Sphinx documentation
* GitHub Pages deployment workflow
* API documentation for all modules
* Training and evaluation guides
* Installation and usage instructions

Changed
~~~~~~~

* Improved project structure organization
* Enhanced configuration management
* Updated dependency specifications

[0.1.0] - 2024-12-01
--------------------

Added
~~~~~

**Models**

* STTran (Spatial-Temporal Transformer) baseline implementation
* Tempura model with uncertainty quantification
* SceneLLM integration with language models
* STKET (Spatial-Temporal Knowledge-Enhanced Transformer)
* DSG-DETR (Dynamic Scene Graph DETR)
* OED (Object-Entity Disentanglement) model

**Datasets**

* Action Genome dataset support
* EASG (Enhanced Action Scene Graph) dataset integration
* Visual Genome dataset compatibility
* Comprehensive data preprocessing pipelines
* Efficient data loading and caching mechanisms

**Training Infrastructure**

* Multi-mode training support (PredCLS, SGCLS, SGDET)
* Distributed training capabilities
* Mixed precision training
* Comprehensive logging and monitoring
* Automatic checkpointing and model saving

**Evaluation Framework**

* Standard SGG evaluation metrics (Recall@K, mRecall@K)
* Cross-dataset evaluation capabilities
* Uncertainty calibration metrics
* Robustness testing utilities
* Comprehensive result analysis tools

**GUI Application**

* Interactive video scene graph visualization
* Real-time model inference
* Multi-model comparison interface
* Result export and visualization tools

**Utilities and Tools**

* Object detection integration (FasterRCNN)
* Feature extraction pipelines
* Data augmentation and preprocessing
* Performance profiling tools
* Extensive test suite

Fixed
~~~~~

* Memory management in large dataset loading
* Temporal consistency in video processing
* Cross-platform compatibility issues
* GPU memory optimization

Security
~~~~~~~~

* Input validation for all data loaders
* Safe model loading and checkpoint handling
* Secure temporary file handling

[0.0.1] - 2024-01-01
--------------------

Added
~~~~~

* Initial project structure
* Basic STTran implementation
* Action Genome dataset loader
* Simple training script
* Basic evaluation metrics

---

**Legend:**

* ``Added`` for new features
* ``Changed`` for changes in existing functionality
* ``Deprecated`` for soon-to-be removed features
* ``Removed`` for now removed features
* ``Fixed`` for any bug fixes
* ``Security`` for security-related changes
