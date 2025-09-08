# AG200 Benchmark Scripts

This directory contains scripts for running comprehensive benchmarks on the Action Genome 200 dataset with three different models: STTran, DSG-DETR, and Tempura.

## Overview

The benchmark trains each model 3 times on the ag200 dataset in all three modes (`predcls`, `sgcls`, `sgdet`) for 10 epochs each, providing a total of 27 training runs. This supports analysis of model performance and computation metrics across modes.

## Files

- `ag200_benchmark.sh` - Bash script for Linux/macOS
- `ag200_benchmark.ps1` - PowerShell script for Windows
- `plot_ag200_benchmark.py` - Python script for plotting results
- `README_AG200_Benchmark.md` - This documentation

## Prerequisites

1. **Dataset**: Ensure the ag200 dataset is available at `data/action_genome200`
2. **Python Environment**: All required packages should be installed
3. **GPU**: NVIDIA GPU recommended for training (CPU fallback available)

## Usage

### Running the Benchmark

#### On Linux/macOS:
```bash
# Make script executable
chmod +x scripts/training/runs/run_benchmark/ag200_benchmark.sh

# Run the benchmark
./scripts/training/runs/run_benchmark/ag200_benchmark.sh
```

#### On Windows:
```powershell
# Run the PowerShell script
.\scripts\training\runs\run_benchmark\ag200_benchmark.ps1
```

You can customize parameters (epochs, paths) via the script arguments.

### Plotting Results

The PowerShell script invokes plotting automatically at the end. To run manually:

```powershell
python scripts/training/runs/run_benchmark/plot_ag200_benchmark.py --results_dir data/benchmark/ag200 --output_dir data/benchmark/ag200/plots
```

## Output Structure

```
data/benchmark/ag200/
├── predcls/
│   ├── sttran_run1/
│   │   ├── logfile.txt
│   │   ├── metrics_summary.txt
│   │   └── model_best.tar
│   ├── ...
├── sgcls/
│   ├── dsg-detr_run1/
│   ├── ...
├── sgdet/
│   ├── tempura_run1/
│   ├── ...
├── logs/
│   ├── sttran_predcls_run1.log
│   ├── dsg-detr_sgdet_run3.log
│   └── ...
└── plots/
    ├── ag200_combined_metrics.pdf
    ├── sttran_sgdet_train_vs_val_loss.pdf
    └── compute_performance_tradeoff.pdf
```

## Metrics Tracked

### Performance Metrics
- **R@20, R@50, R@100**: Recall at different K values
- **mR@20, mR@50, mR@100**: Mean recall at different K values
- **Training Loss**: Per-epoch average
- **Validation Loss**: Per-epoch average

### Computation Metrics
- **Training Time**: Total duration per run
- **Memory Usage**: Memory delta per run
- **Disk Usage**: Storage change (included in logs)
- **GPU Utilization**: Logged if `nvidia-smi` is available

## Model Configurations

### STTran
- Learning Rate: 1e-4
- Encoder Layers: 1
- Decoder Layers: 3
- Batch Size: 1
- Optimizer: AdamW

### DSG-DETR
- Learning Rate: 1e-4
- Encoder Layers: 1
- Decoder Layers: 3
- Batch Size: 1
- Optimizer: AdamW
- Hungarian Matcher: Enabled

### Tempura
- Learning Rate: 1e-4
- Encoder Layers: 1
- Decoder Layers: 3
- Batch Size: 1
- Optimizer: AdamW
- Object Head: GMM
- Relation Head: GMM
- K (mixtures): 3
- Memory Computation: Disabled

## Expected Runtime

- **Per Model Run**: hardware-dependent
- **Total Benchmark**: 27 runs (3 models × 3 modes × 3 runs)

## Troubleshooting

1. **Dataset Not Found**
   - Ensure `data/action_genome200` exists
   - Check dataset path in script parameters

2. **CUDA Out of Memory**
   - Reduce batch size in script
   - Use CPU training (set device to "cpu")

3. **Training Fails**
   - Check log files under `data/benchmark/ag200/logs`
   - Verify Python environment and dependencies

4. **Plotting Issues**
   - Install matplotlib: `pip install matplotlib`
   - Check that results directory contains run folders with `logfile.txt`

## Analysis

- **Model Performance**: Compare R@K and mR@K across models and modes
- **Training Stability**: Analyze loss curves and convergence
- **Computational Efficiency**: Compare duration and memory delta vs final R@20
- **Statistical Significance**: 3 runs per model and mode for robustness

## Citation

If you use these benchmark scripts in your research, please cite the original papers:

- STTran: https://arxiv.org/abs/2007.15607
- DSG-DETR: https://arxiv.org/abs/2207.09760
- Tempura: https://arxiv.org/abs/2108.11928
