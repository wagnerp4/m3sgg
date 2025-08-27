Evaluation Guide
================

This guide covers evaluation metrics, procedures, and analysis for DLHM VidSGG models.

Evaluation Overview
-------------------

DLHM VidSGG provides comprehensive evaluation capabilities for video scene graph generation models across different modes and datasets.

**Evaluation Modes**

* **PredCLS**: Evaluate relationship prediction given ground truth objects
* **SGCLS**: Evaluate both object classification and relationship prediction
* **SGDET**: Evaluate end-to-end object detection and relationship prediction

Basic Evaluation
----------------

Simple Evaluation Command
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python test.py -m predcls -datasize large -data_path data/action_genome -model_path output/model.pth

This evaluates a trained model on the Action Genome test set.

Complete Evaluation Command
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python test.py \
     -m predcls \
     -datasize large \
     -data_path data/action_genome \
     -model_path output/sttran_predcls/checkpoint_best.tar \
     -save_results output/evaluation_results.json

Evaluation Metrics
------------------

Recall Metrics
~~~~~~~~~~~~~~

The primary evaluation metrics for scene graph generation:

**Recall@K**
   Percentage of ground truth relationships that appear in top-K predictions

**Mean Recall@K (mRecall@K)**
   Average recall across all relationship categories

.. list-table:: Standard Evaluation Metrics
   :widths: 25 75
   :header-rows: 1

   * - Metric
     - Description
   * - Recall@10
     - Recall considering top 10 predictions per frame
   * - Recall@20  
     - Recall considering top 20 predictions per frame
   * - Recall@50
     - Recall considering top 50 predictions per frame
   * - mRecall@10
     - Mean recall across relationship categories (top 10)
   * - mRecall@20
     - Mean recall across relationship categories (top 20)
   * - mRecall@50
     - Mean recall across relationship categories (top 50)

Zero-Shot Metrics
~~~~~~~~~~~~~~~~~

For unseen relationship combinations:

* **Zero-Shot Recall@K**: Performance on novel object-relationship-object triplets
* **Compositional Recall**: Performance on new compositions of seen elements

Per-Category Analysis
~~~~~~~~~~~~~~~~~~~~~

Detailed analysis for each relationship category:

.. code-block:: python

   # Example per-category results
   per_category_results = {
       'holding': {'recall@20': 25.3, 'precision': 18.7},
       'sitting_on': {'recall@20': 31.2, 'precision': 22.1},
       'looking_at': {'recall@20': 15.8, 'precision': 12.4}
   }

Evaluation Procedures
---------------------

Standard Evaluation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Evaluate all models on test set
   for model in sttran tempura scenellm stket; do
     python test.py \
       -m predcls \
       -model_path output/${model}_predcls/checkpoint_best.tar \
       -save_results results/${model}_predcls_results.json
   done

Cross-Dataset Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate model generalization across datasets:

.. code-block:: bash

   # Train on Action Genome, test on EASG
   python test.py \
     -m predcls \
     -data_path data/EASG \
     -model_path output/action_genome_model.pth \
     -save_results cross_dataset_results.json

Temporal Evaluation
~~~~~~~~~~~~~~~~~~~

Analyze performance across different temporal windows:

.. code-block:: bash

   # Evaluate with different temporal window sizes
   for window in 1 3 5 10; do
     python test.py \
       -m predcls \
       -temporal_window $window \
       -model_path output/model.pth
   done

Mode-Specific Evaluation
------------------------

PredCLS Evaluation
~~~~~~~~~~~~~~~~~~

**Input**: Ground truth object bounding boxes and labels
**Task**: Predict relationships between objects

.. code-block:: bash

   python test.py -m predcls -model_path output/sttran_predcls.pth

**Key Metrics**:
* Relationship prediction accuracy
* Per-category relationship recall
* Temporal consistency

SGCLS Evaluation  
~~~~~~~~~~~~~~~~

**Input**: Ground truth object bounding boxes
**Task**: Predict object labels and relationships

.. code-block:: bash

   python test.py -m sgcls -model_path output/sttran_sgcls.pth

**Key Metrics**:
* Object classification accuracy
* Relationship prediction given predicted objects
* Joint object-relationship accuracy

SGDET Evaluation
~~~~~~~~~~~~~~~~

**Input**: Raw video frames
**Task**: Detect objects and predict relationships end-to-end

.. code-block:: bash

   python test.py -m sgdet -model_path output/sttran_sgdet.pth

**Key Metrics**:
* Object detection mAP
* Relationship prediction accuracy
* End-to-end scene graph quality

Advanced Evaluation
-------------------

Uncertainty Evaluation
~~~~~~~~~~~~~~~~~~~~~~

For models with uncertainty estimation (e.g., Tempura):

.. code-block:: python

   # Evaluate uncertainty calibration
   python evaluate_uncertainty.py \
     -model_path output/tempura_model.pth \
     -calibration_method temperature_scaling

**Uncertainty Metrics**:
* Calibration error (ECE)
* Reliability diagrams
* Uncertainty-accuracy correlation

Robustness Evaluation
~~~~~~~~~~~~~~~~~~~~~

Test model robustness to various perturbations:

.. code-block:: bash

   # Evaluate with noise
   python test_robustness.py \
     -model_path output/model.pth \
     -noise_level 0.1 \
     -noise_type gaussian

**Robustness Tests**:
* Gaussian noise in input frames
* Occlusions and crops
* Temporal jittering
* Lighting changes

Efficiency Evaluation
~~~~~~~~~~~~~~~~~~~~~

Measure computational efficiency:

.. code-block:: python

   # Profile model inference
   python profile_model.py \
     -model_path output/model.pth \
     -batch_size 1 \
     -num_iterations 100

**Efficiency Metrics**:
* Inference time per frame
* GPU memory usage
* FLOPs count
* Model parameters

Evaluation Analysis
-------------------

Statistical Significance
~~~~~~~~~~~~~~~~~~~~~~~~

Test statistical significance of results:

.. code-block:: python

   from scipy import stats
   
   # Compare two models
   model1_scores = [19.2, 18.8, 19.5, ...]
   model2_scores = [20.1, 19.7, 20.3, ...]
   
   t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
   print(f"P-value: {p_value}")

Error Analysis
~~~~~~~~~~~~~~

Analyze common failure modes:

.. code-block:: python

   # Analyze prediction errors
   python analyze_errors.py \
     -predictions output/predictions.json \
     -ground_truth data/test_annotations.json \
     -save_analysis error_analysis.html

**Analysis Categories**:
* Frequent false positives
* Common missed relationships
* Object detection failures
* Temporal inconsistencies

Visualization
~~~~~~~~~~~~~

Generate evaluation visualizations:

.. code-block:: python

   # Create evaluation plots
   python visualize_results.py \
     -results_dir output/evaluation_results/ \
     -output_dir plots/

**Visualization Types**:
* Recall curves
* Precision-recall plots
* Confusion matrices
* Per-category performance bars

Benchmark Comparison
--------------------

Standard Benchmarks
~~~~~~~~~~~~~~~~~~~

Compare against established benchmarks:

.. list-table:: Action Genome Benchmark Results
   :widths: 20 15 15 15 15 20
   :header-rows: 1

   * - Model
     - R@10
     - R@20
     - R@50
     - mR@50
     - Year
   * - IMP
     - 8.9
     - 12.1
     - 17.8
     - 4.2
     - 2017
   * - KERN
     - 9.2
     - 12.7
     - 18.4
     - 4.8
     - 2019
   * - STTran
     - 14.6
     - 19.2
     - 26.5
     - 7.8
     - 2021
   * - Tempura
     - 15.8
     - 21.1
     - 28.3
     - 8.9
     - 2022

Leaderboard Submission
~~~~~~~~~~~~~~~~~~~~~~

Prepare results for benchmark submission:

.. code-block:: python

   # Format results for submission
   python format_submission.py \
     -predictions output/test_predictions.json \
     -output submission.zip

Custom Evaluation
-----------------

Domain-Specific Metrics
~~~~~~~~~~~~~~~~~~~~~~~

Implement custom metrics for specific domains:

.. code-block:: python

   def custom_metric(predictions, ground_truth):
       # Custom evaluation logic
       score = compute_domain_specific_score(predictions, ground_truth)
       return score

Temporal Metrics
~~~~~~~~~~~~~~~~

Evaluate temporal consistency:

.. code-block:: python

   def temporal_consistency(predictions):
       # Measure consistency across time
       consistency_score = 0
       for t in range(1, len(predictions)):
           consistency_score += similarity(predictions[t], predictions[t-1])
       return consistency_score / (len(predictions) - 1)

Quality Assessment
~~~~~~~~~~~~~~~~~~

Assess overall scene graph quality:

.. code-block:: python

   def scene_graph_quality(prediction, ground_truth):
       # Graph-level similarity metrics
       node_similarity = compute_node_similarity(prediction, ground_truth)
       edge_similarity = compute_edge_similarity(prediction, ground_truth)
       structure_similarity = compute_structure_similarity(prediction, ground_truth)
       
       return (node_similarity + edge_similarity + structure_similarity) / 3

Evaluation Best Practices
--------------------------

Reproducibility
~~~~~~~~~~~~~~~

Ensure reproducible evaluation results:

.. code-block:: python

   # Set random seeds for consistent evaluation
   torch.manual_seed(42)
   np.random.seed(42)
   
   # Use consistent evaluation protocols
   eval_config = {
       'batch_size': 1,
       'num_workers': 0,  # For reproducibility
       'deterministic': True
   }

Multiple Runs
~~~~~~~~~~~~~

Perform multiple evaluation runs:

.. code-block:: bash

   # Run evaluation multiple times with different seeds
   for seed in 42 123 456 789 999; do
     python test.py \
       -m predcls \
       -model_path output/model.pth \
       -seed $seed \
       -save_results results/run_${seed}.json
   done

Statistical Reporting
~~~~~~~~~~~~~~~~~~~~~

Report results with confidence intervals:

.. code-block:: python

   import numpy as np
   from scipy import stats
   
   # Calculate mean and confidence interval
   scores = [19.2, 18.8, 19.5, 19.1, 19.3]
   mean_score = np.mean(scores)
   std_error = stats.sem(scores)
   ci = stats.t.interval(0.95, len(scores)-1, loc=mean_score, scale=std_error)
   
   print(f"Recall@20: {mean_score:.1f} ± {std_error:.1f} (95% CI: {ci[0]:.1f}-{ci[1]:.1f})")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Low Evaluation Scores**
   * Verify ground truth data format
   * Check evaluation metric implementation
   * Compare preprocessing with training

**Inconsistent Results**
   * Set random seeds for reproducibility
   * Use same data splits as training
   * Verify model loading correctly

**Memory Issues During Evaluation**
   * Reduce batch size to 1
   * Process samples sequentially
   * Clear cache between batches

Performance Debugging
~~~~~~~~~~~~~~~~~~~~~

**Slow Evaluation**
   * Profile bottlenecks in evaluation code
   * Optimize data loading pipeline
   * Use GPU for faster inference

**Unexpected Results**
   * Visualize predictions vs ground truth
   * Check for data leakage or preprocessing errors
   * Validate against simple baselines

Evaluation Reports
------------------

Automated Reports
~~~~~~~~~~~~~~~~~

Generate comprehensive evaluation reports:

.. code-block:: python

   # Generate evaluation report
   python generate_report.py \
     -results output/evaluation_results.json \
     -template templates/evaluation_report.html \
     -output reports/model_evaluation.html

Report Contents
~~~~~~~~~~~~~~~

Standard evaluation reports include:

* **Model Information**: Architecture, parameters, training details
* **Dataset Statistics**: Test set size, class distribution
* **Quantitative Results**: All evaluation metrics with confidence intervals
* **Qualitative Analysis**: Visualization of predictions and failures
* **Comparison**: Performance relative to baselines and state-of-the-art

Next Steps
----------

* :doc:`training` - Return to training with evaluation insights
* :doc:`models` - Understand model architectures and their evaluation characteristics
* :doc:`api/lib` - API documentation for evaluation functions
