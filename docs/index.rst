M3SGG Documentation
===================

Welcome to the documentation for M3SGG (Modular, multi-modal Scene Graph Generation), a modular framework for video scene graph generation and analysis.

Overview
--------

M3SGG builds on established SGG research and extends it with modular components, dataset support, and training/evaluation tooling. It supports multiple approaches and provides utilities for training, evaluation, and analysis of video scene graphs.

Key Features
------------

* **Multiple SGG Models**: STTran, DSG-DETR, STKET, Tempura, SceneLLM, OED, VLM
* **Dataset Support**: Action Genome, EASG, and Visual Genome datasets
* **Language Integration**: Summarization and language modeling capabilities
* **GUI Application**: Interactive demo application for visualization and testing
* **Comprehensive Evaluation**: Multiple evaluation modes (PredCLS, SGCLS, SGDET)

Quick Start
-----------

To get started quickly, see the :doc:`installation` guide and then check out the :doc:`usage` examples.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   
   installation
   usage
   datasets
   models
   training
   evaluation

.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   
   api

.. toctree::
   :maxdepth: 1
   :caption: Additional Information:
   
   contributing
   changelog
   license

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
