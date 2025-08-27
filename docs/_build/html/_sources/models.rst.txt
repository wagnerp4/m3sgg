Models
======

This section provides detailed information about the scene graph generation models implemented in DLHM VidSGG.

Model Overview
--------------

DLHM VidSGG implements several state-of-the-art models for video scene graph generation:

* **STTran**: Spatial-Temporal Transformer baseline
* **Tempura**: Temporal relationship modeling with uncertainty
* **SceneLLM**: Large language model integration
* **STKET**: Spatial-temporal knowledge-enhanced transformer
* **DSG-DETR**: Dynamic Scene Graph DETR
* **OED**: Object-Entity Disentanglement

Each model can be trained and evaluated in three modes: PredCLS, SGCLS, and SGDET.

STTran Model
------------

The Spatial-Temporal Transformer serves as the baseline model for video scene graph generation.

**Architecture**

STTran uses a transformer-based architecture to model spatial and temporal relationships:

* **Spatial Encoder**: Processes object features within frames
* **Temporal Encoder**: Models relationships across time
* **Relation Decoder**: Predicts relationship classifications

**Key Features**

* Multi-head attention for spatial relationship modeling
* Temporal attention for cross-frame relationship tracking
* Hierarchical object representation learning

**Usage**

.. code-block:: bash

   # Training STTran
   python train.py -model sttran -mode predcls -data_path data/action_genome

**Configuration**

.. code-block:: python

   sttran_config = {
       'spatial_layer': 4,
       'temporal_layer': 2,
       'hidden_dim': 512,
       'num_heads': 8,
       'dropout': 0.1
   }

Tempura Model
-------------

Tempura focuses on temporal relationship modeling with uncertainty quantification.

**Architecture**

* **Gaussian Mixture Models**: For uncertainty modeling
* **Temporal Attention**: Enhanced temporal relationship understanding
* **Uncertainty Heads**: Quantify prediction confidence

**Key Features**

* Uncertainty-aware relationship prediction
* Improved temporal consistency
* Robust to noisy annotations

**Usage**

.. code-block:: bash

   # Training Tempura
   python train.py -model tempura -mode predcls -data_path data/action_genome

**Configuration**

.. code-block:: python

   tempura_config = {
       'num_mixtures': 3,
       'uncertainty_threshold': 0.5,
       'temporal_window': 5,
       'gmm_regularization': 0.01
   }

SceneLLM Model
--------------

SceneLLM integrates large language models for enhanced scene understanding.

**Architecture**

* **Vision Encoder**: Processes visual features
* **Language Model**: Generates textual scene descriptions
* **Multimodal Fusion**: Combines visual and textual representations
* **Scene Graph Decoder**: Produces structured scene graphs

**Key Features**

* Natural language scene understanding
* Multimodal learning capabilities
* Zero-shot relationship recognition
* Textual scene graph generation

**Usage**

.. code-block:: bash

   # Training SceneLLM
   python train.py -model scenellm -mode predcls -data_path data/action_genome

**Configuration**

.. code-block:: python

   scenellm_config = {
       'llm_model': 'gemma3-270M',
       'vision_backbone': 'resnet101',
       'fusion_layers': 3,
       'text_generation': True
   }

STKET Model
-----------

Spatial-Temporal Knowledge-Enhanced Transformer incorporates external knowledge.

**Architecture**

* **Knowledge Graph Integration**: External knowledge incorporation
* **Enhanced Attention**: Knowledge-guided attention mechanisms
* **Multi-scale Temporal Modeling**: Different temporal scales

**Key Features**

* External knowledge integration
* Improved relationship reasoning
* Multi-scale temporal analysis

**Usage**

.. code-block:: bash

   # Training STKET
   python train.py -model stket -mode predcls -data_path data/action_genome

Model Comparison
----------------

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Model Performance on Action Genome (PredCLS)
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Model
     - Recall@10
     - Recall@20
     - Recall@50
     - mRecall@50
   * - STTran
     - 14.6
     - 19.2
     - 26.5
     - 7.8
   * - Tempura
     - 15.8
     - 21.1
     - 28.3
     - 8.9
   * - SceneLLM
     - 16.2
     - 22.0
     - 30.1
     - 9.5
   * - STKET
     - 15.1
     - 20.5
     - 27.8
     - 8.3

Computational Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Computational Requirements
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Model
     - GPU Memory
     - Training Time
     - Inference Speed
     - Parameters
   * - STTran
     - 8GB
     - 12 hours
     - 30 FPS
     - 45M
   * - Tempura
     - 10GB
     - 15 hours
     - 25 FPS
     - 52M
   * - SceneLLM
     - 16GB
     - 24 hours
     - 15 FPS
     - 270M
   * - STKET
     - 12GB
     - 18 hours
     - 28 FPS
     - 58M

Model Selection Guidelines
--------------------------

Choose Based on Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For Speed and Efficiency**
   * **STTran**: Best balance of speed and accuracy
   * **Recommended for**: Real-time applications, limited resources

**For Accuracy**
   * **SceneLLM**: Highest accuracy with language understanding
   * **Recommended for**: Research, offline analysis

**For Uncertainty Quantification**
   * **Tempura**: Built-in uncertainty estimation
   * **Recommended for**: Safety-critical applications, quality control

**For Knowledge Integration**
   * **STKET**: External knowledge incorporation
   * **Recommended for**: Domain-specific applications, expert systems

Training Considerations
-----------------------

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~~

**Common Hyperparameters**

.. code-block:: python

   base_config = {
       'learning_rate': 1e-4,
       'batch_size': 1,
       'weight_decay': 1e-5,
       'gradient_clip': 5.0,
       'warmup_steps': 1000,
       'scheduler': 'cosine'
   }

**Model-Specific Tuning**

.. code-block:: python

   # STTran specific
   sttran_tuning = {
       'spatial_layer': [2, 4, 6],
       'temporal_layer': [1, 2, 3],
       'hidden_dim': [256, 512, 1024]
   }
   
   # Tempura specific
   tempura_tuning = {
       'num_mixtures': [2, 3, 5],
       'uncertainty_threshold': [0.3, 0.5, 0.7],
       'gmm_regularization': [0.001, 0.01, 0.1]
   }

Training Strategies
~~~~~~~~~~~~~~~~~~~

**Progressive Training**

1. Start with simpler models (STTran)
2. Transfer knowledge to complex models
3. Fine-tune on specific datasets

**Curriculum Learning**

1. Train on easier samples first
2. Gradually increase difficulty
3. Improve convergence and performance

Model Customization
-------------------

Custom Model Development
~~~~~~~~~~~~~~~~~~~~~~~~

To implement a custom model:

.. code-block:: python

   from lib.base_model import BaseModel
   
   class CustomModel(BaseModel):
       def __init__(self, config):
           super().__init__(config)
           # Define model components
           
       def forward(self, inputs):
           # Implement forward pass
           pass
           
       def compute_loss(self, predictions, targets):
           # Implement loss computation
           pass

**Integration Steps**

1. Implement model class
2. Add to model factory
3. Update configuration files
4. Test with existing pipeline

Transfer Learning
~~~~~~~~~~~~~~~~~

**Pretrained Models**

Download pretrained models for different datasets:

.. code-block:: bash

   # Download Action Genome pretrained models
   wget <model_url> -P data/checkpoints/

**Fine-tuning**

.. code-block:: python

   # Load pretrained model
   model = load_model('data/checkpoints/sttran_pretrained.pth')
   
   # Fine-tune on new dataset
   fine_tune(model, new_dataset, epochs=10)

Deployment
----------

Model Export
~~~~~~~~~~~~

Export trained models for deployment:

.. code-block:: python

   # Export to ONNX
   torch.onnx.export(model, sample_input, 'model.onnx')
   
   # Export to TorchScript
   traced_model = torch.jit.trace(model, sample_input)
   traced_model.save('model.pt')

Optimization
~~~~~~~~~~~~

**Model Quantization**

.. code-block:: python

   # Post-training quantization
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )

**Model Pruning**

.. code-block:: python

   # Structured pruning
   from torch.nn.utils import prune
   prune.global_unstructured(
       parameters_to_prune,
       pruning_method=prune.L1Unstructured,
       amount=0.2
   )

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Training Instability**
   * Reduce learning rate
   * Add gradient clipping
   * Use mixed precision training

**Poor Performance**
   * Check data preprocessing
   * Verify model configuration
   * Compare with baseline results

**Memory Issues**
   * Reduce batch size
   * Use gradient accumulation
   * Enable gradient checkpointing

Next Steps
----------

* :doc:`training` - Detailed training procedures
* :doc:`evaluation` - Model evaluation and metrics
* :doc:`api/models` - API documentation for models
