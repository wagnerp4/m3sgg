Training Guide
==============

This guide provides comprehensive information about training models in DLHM VidSGG.

Training Overview
-----------------

DLHM VidSGG supports training various scene graph generation models on multiple datasets with different evaluation modes.

**Supported Training Modes**

* **PredCLS**: Predicate Classification - predict relationships given object boxes and labels
* **SGCLS**: Scene Graph Classification - predict object labels and relationships given boxes  
* **SGDET**: Scene Graph Detection - end-to-end detection and relationship prediction

Basic Training
--------------

Simple Training Command
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python train.py -mode predcls -datasize large -data_path data/action_genome -model sttran

This command trains STTran model on Action Genome dataset in PredCLS mode.

Complete Training Command
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python train.py \
     -mode predcls \
     -datasize large \
     -data_path data/action_genome \
     -model sttran \
     -lr 1e-4 \
     -batch_size 1 \
     -epochs 100 \
     -save_path output/sttran_predcls

Training Parameters
-------------------

Core Parameters
~~~~~~~~~~~~~~~

.. list-table:: Essential Training Parameters
   :widths: 25 20 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``-mode``
     - predcls
     - Training mode: predcls, sgcls, sgdet
   * - ``-model``
     - sttran
     - Model type: sttran, tempura, scenellm, stket
   * - ``-data_path``
     - Required
     - Path to dataset directory
   * - ``-datasize``
     - large
     - Dataset size: small, large
   * - ``-lr``
     - 1e-4
     - Learning rate
   * - ``-batch_size``
     - 1
     - Batch size for training
   * - ``-epochs``
     - 100
     - Number of training epochs

Advanced Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table:: Advanced Training Parameters
   :widths: 25 20 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``-weight_decay``
     - 1e-5
     - L2 regularization weight
   * - ``-clip_grad``
     - 5.0
     - Gradient clipping threshold
   * - ``-warmup_steps``
     - 1000
     - Learning rate warmup steps
   * - ``-scheduler``
     - step
     - LR scheduler: step, cosine, plateau
   * - ``-save_freq``
     - 10
     - Model checkpoint save frequency
   * - ``-eval_freq``
     - 5
     - Evaluation frequency during training

Model-Specific Training
-----------------------

STTran Training
~~~~~~~~~~~~~~~

**Standard Configuration**

.. code-block:: bash

   python train.py \
     -mode predcls \
     -model sttran \
     -data_path data/action_genome \
     -lr 1e-4 \
     -spatial_layer 4 \
     -temporal_layer 2 \
     -hidden_dim 512

**Optimized Configuration**

.. code-block:: bash

   python train.py \
     -mode predcls \
     -model sttran \
     -data_path data/action_genome \
     -lr 5e-5 \
     -batch_size 2 \
     -warmup_steps 2000 \
     -scheduler cosine

Tempura Training
~~~~~~~~~~~~~~~~

**Basic Configuration**

.. code-block:: bash

   python train.py \
     -mode predcls \
     -model tempura \
     -data_path data/action_genome \
     -lr 1e-4 \
     -num_mixtures 3 \
     -uncertainty_threshold 0.5

**Advanced Configuration**

.. code-block:: bash

   python train.py \
     -mode predcls \
     -model tempura \
     -data_path data/action_genome \
     -lr 8e-5 \
     -gmm_regularization 0.01 \
     -temporal_window 5

SceneLLM Training
~~~~~~~~~~~~~~~~~

**Basic Configuration**

.. code-block:: bash

   python train.py \
     -mode predcls \
     -model scenellm \
     -data_path data/action_genome \
     -lr 5e-5 \
     -batch_size 1

**With Language Model Fine-tuning**

.. code-block:: bash

   python train.py \
     -mode predcls \
     -model scenellm \
     -data_path data/action_genome \
     -lr 1e-5 \
     -llm_lr 1e-6 \
     -freeze_llm_epochs 10

Training Strategies
-------------------

Progressive Training
~~~~~~~~~~~~~~~~~~~~

Train models progressively from easier to harder modes:

.. code-block:: bash

   # Step 1: Train PredCLS (easiest)
   python train.py -mode predcls -model sttran -epochs 50
   
   # Step 2: Fine-tune for SGCLS
   python train.py -mode sgcls -model sttran -resume_from checkpoint_predcls.pth -epochs 25
   
   # Step 3: Fine-tune for SGDET (hardest)
   python train.py -mode sgdet -model sttran -resume_from checkpoint_sgcls.pth -epochs 25

Multi-Dataset Training
~~~~~~~~~~~~~~~~~~~~~~

Train on multiple datasets for better generalization:

.. code-block:: bash

   # Train on Action Genome
   python train.py -mode predcls -data_path data/action_genome -epochs 80
   
   # Fine-tune on EASG
   python train.py -mode predcls -data_path data/EASG -resume_from ag_checkpoint.pth -epochs 20

Curriculum Learning
~~~~~~~~~~~~~~~~~~~

Implement curriculum learning for better convergence:

.. code-block:: python

   # Example curriculum learning script
   for epoch in range(epochs):
       if epoch < 20:
           # Easy samples first
           dataloader = get_easy_samples()
       elif epoch < 60:
           # Medium difficulty
           dataloader = get_medium_samples()
       else:
           # Full dataset
           dataloader = get_full_dataset()
       
       train_epoch(model, dataloader)

Monitoring Training
-------------------

Training Logs
~~~~~~~~~~~~~

Monitor training progress through log files:

.. code-block:: text

   output/action_genome/sttran_predcls_20241201_143022/logfile.txt

**Log Content Example**

.. code-block:: text

   Epoch 1/100 - Loss: 2.45 - LR: 1e-4 - Time: 120s
   Epoch 2/100 - Loss: 2.32 - LR: 1e-4 - Time: 118s
   Epoch 5/100 - Eval - Recall@10: 8.2 - Recall@20: 12.1
   ...

Visualization
~~~~~~~~~~~~~

Use tensorboard for visual monitoring:

.. code-block:: bash

   # Launch tensorboard
   tensorboard --logdir output/

**Tracked Metrics**

* Training and validation loss
* Learning rate schedules
* Gradient norms
* Model weights histograms
* Evaluation metrics

Early Stopping
~~~~~~~~~~~~~~

Implement early stopping to prevent overfitting:

.. code-block:: python

   early_stopping = EarlyStopping(
       patience=10,
       min_delta=0.001,
       monitor='val_recall@20'
   )

Optimization Techniques
-----------------------

Mixed Precision Training
~~~~~~~~~~~~~~~~~~~~~~~~

Use automatic mixed precision for faster training:

.. code-block:: bash

   python train.py \
     -mode predcls \
     -model sttran \
     -use_amp True \
     -opt_level O1

Gradient Accumulation
~~~~~~~~~~~~~~~~~~~~~

Simulate larger batch sizes with gradient accumulation:

.. code-block:: bash

   python train.py \
     -mode predcls \
     -model sttran \
     -batch_size 1 \
     -accumulate_grad_batches 4  # Effective batch size: 4

Data Parallel Training
~~~~~~~~~~~~~~~~~~~~~~

Use multiple GPUs for faster training:

.. code-block:: bash

   # Single node, multiple GPUs
   python -m torch.distributed.launch --nproc_per_node=4 train.py \
     -mode predcls \
     -model sttran \
     -distributed True

Hyperparameter Tuning
----------------------

Grid Search
~~~~~~~~~~~

Systematic hyperparameter exploration:

.. code-block:: bash

   # Grid search script
   for lr in 1e-5 1e-4 5e-4; do
     for batch_size in 1 2 4; do
       python train.py -lr $lr -batch_size $batch_size
     done
   done

Random Search
~~~~~~~~~~~~~

More efficient hyperparameter exploration:

.. code-block:: python

   import random
   
   # Random hyperparameter sampling
   lr = random.uniform(1e-5, 1e-3)
   weight_decay = random.uniform(1e-6, 1e-4)
   hidden_dim = random.choice([256, 512, 1024])

Bayesian Optimization
~~~~~~~~~~~~~~~~~~~~~

Use Optuna for advanced hyperparameter optimization:

.. code-block:: python

   import optuna
   
   def objective(trial):
       lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
       batch_size = trial.suggest_categorical('batch_size', [1, 2, 4])
       
       # Train model with suggested hyperparameters
       score = train_model(lr=lr, batch_size=batch_size)
       return score
   
   study = optuna.create_study()
   study.optimize(objective, n_trials=100)

Checkpointing
-------------

Automatic Checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~

Models are automatically saved during training:

.. code-block:: text

   output/action_genome/sttran_predcls_20241201_143022/
   ├── checkpoint_epoch_10.tar
   ├── checkpoint_epoch_20.tar
   ├── checkpoint_best.tar
   └── checkpoint_final.tar

Manual Checkpointing
~~~~~~~~~~~~~~~~~~~~

Save checkpoints at specific points:

.. code-block:: python

   # Save checkpoint
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'loss': loss,
       'config': config
   }, f'checkpoint_epoch_{epoch}.tar')

Resume Training
~~~~~~~~~~~~~~~

Resume from saved checkpoints:

.. code-block:: bash

   python train.py \
     -mode predcls \
     -model sttran \
     -resume_from output/checkpoint_epoch_50.tar

Troubleshooting
---------------

Common Training Issues
~~~~~~~~~~~~~~~~~~~~~~

**Loss Not Decreasing**

* Check learning rate (try lower values: 1e-5, 5e-5)
* Verify data loading and preprocessing
* Check model configuration
* Monitor gradient norms

**Training Instability**

* Add gradient clipping: ``-clip_grad 5.0``
* Use learning rate warmup: ``-warmup_steps 1000``
* Reduce learning rate
* Check for NaN values in loss

**Memory Issues**

* Reduce batch size: ``-batch_size 1``
* Use gradient accumulation
* Enable gradient checkpointing
* Clear cache regularly

**Slow Training**

* Use mixed precision training
* Increase number of data loading workers
* Optimize data preprocessing
* Use faster storage (SSD)

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

**GPU Utilization**

.. code-block:: bash

   # Monitor GPU usage
   nvidia-smi -l 1

**Memory Profiling**

.. code-block:: python

   # Profile memory usage
   import torch.profiler
   
   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CUDA],
       record_shapes=True
   ) as prof:
       train_step()
   
   print(prof.key_averages().table())

Best Practices
--------------

Training Workflow
~~~~~~~~~~~~~~~~~

1. **Data Preparation**: Verify dataset integrity and preprocessing
2. **Baseline Training**: Start with known good configurations
3. **Hyperparameter Tuning**: Systematically optimize parameters
4. **Model Selection**: Choose best performing checkpoint
5. **Final Evaluation**: Evaluate on test set

Reproducibility
~~~~~~~~~~~~~~~

Ensure reproducible results:

.. code-block:: python

   # Set random seeds
   torch.manual_seed(42)
   np.random.seed(42)
   random.seed(42)
   
   # Use deterministic algorithms
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

Documentation
~~~~~~~~~~~~~

Document training experiments:

.. code-block:: text

   Training Log - STTran PredCLS
   =============================
   Date: 2024-01-15
   Model: STTran
   Dataset: Action Genome (large)
   Mode: PredCLS
   
   Hyperparameters:
   - Learning Rate: 1e-4
   - Batch Size: 2
   - Epochs: 100
   
   Results:
   - Best Recall@20: 19.8%
   - Training Time: 12 hours
   - Final Loss: 0.85

Next Steps
----------

* :doc:`evaluation` - Learn about model evaluation and metrics
* :doc:`usage` - Understanding basic usage patterns  
* :doc:`models` - Deep dive into model architectures
