DLHM VidSGG Documentation
=========================

Welcome to the documentation for DLHM VidSGG (Dynamic Scene Graph Generation for Videos), a comprehensive framework for video scene graph generation and analysis.

Overview
--------

DLHM VidSGG is built upon the Spatial-Temporal Transformer for Dynamic Scene Graph Generation framework, extending it with new models, datasets, and advanced processing functionality. The project supports multiple scene graph generation approaches and provides tools for training, evaluation, and analysis of video scene graphs.

Key Features
------------

* **Multiple SGG Models**: STTran, DSG-DETR, STKET, Tempura, SceneLLM, and OED
* **Dataset Support**: Action Genome, EASG, and Visual Genome datasets
* **NLP Integration**: T5/Pegasus summarization and language modeling capabilities
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
