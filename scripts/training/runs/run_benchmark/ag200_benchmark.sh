#!/bin/bash

# AG200 Benchmark Script for Scene Graph Detection
# Trains STTran, DSG-DETR, and Tempura models on ag200 dataset in sgdet mode
# Logs metrics per iteration for plotting R@20/50/100 and mR@20/50/100 vs iteration

set -e  # Exit on any error

# Configuration
DATASET="action_genome200"
MODE="sgdet"
EPOCHS=10
DATA_PATH="data/action_genome200"
OUTPUT_BASE="output/ag200_benchmark"
LOG_DIR="logs/ag200_benchmark"

# Models to train
MODELS=("sttran" "dsg-detr" "tempura")

# Create directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$LOG_DIR"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to get system info
get_system_info() {
    echo "=== System Information ==="
    echo "Hostname: $(hostname)"
    echo "OS: $(uname -a)"
    echo "Python: $(python --version)"
    echo "CUDA: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'NVIDIA GPU not available')"
    echo "Available Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    echo "Available Disk: $(df -h . | tail -1 | awk '{print $4}')"
    echo "========================="
}

# Function to train a model with comprehensive logging
train_model() {
    local model_name=$1
    local model_type=$2
    local run_id=$3
    
    log_message "Starting training for $model_name (Run $run_id)..."
    
    # Create model-specific output directory
    local model_output_dir="$OUTPUT_BASE/${model_name}_run${run_id}"
    local model_log_file="$LOG_DIR/${model_name}_run${run_id}.log"
    
    mkdir -p "$model_output_dir"
    
    # Build training command
    local train_cmd="python scripts/training/train.py"
    train_cmd="$train_cmd -mode $MODE"
    train_cmd="$train_cmd -dataset $DATASET"
    train_cmd="$train_cmd -data_path $DATA_PATH"
    train_cmd="$train_cmd -model $model_type"
    train_cmd="$train_cmd -nepoch $EPOCHS"
    train_cmd="$train_cmd -save_path $model_output_dir"
    train_cmd="$train_cmd -lr 1e-4"
    train_cmd="$train_cmd -batch_size 1"
    train_cmd="$train_cmd -enc_layer 1"
    train_cmd="$train_cmd -dec_layer 3"
    train_cmd="$train_cmd -num_workers 4"
    train_cmd="$train_cmd -seed 42"
    
    # Add model-specific parameters
    case $model_type in
        "dsg-detr")
            train_cmd="$train_cmd -use_matcher True"
            ;;
        "tempura")
            train_cmd="$train_cmd -obj_head gmm"
            train_cmd="$train_cmd -rel_head gmm"
            train_cmd="$train_cmd -K 3"
            train_cmd="$train_cmd -obj_mem_compute False"
            train_cmd="$train_cmd -rel_mem_compute False"
            ;;
    esac
    
    log_message "Training command: $train_cmd"
    echo "Training command: $train_cmd" >> "$model_log_file"
    
    # Record start time and system resources
    local start_time=$(date +%s)
    local start_memory=$(free -m | grep '^Mem:' | awk '{print $3}')
    local start_disk=$(df -m . | tail -1 | awk '{print $4}')
    
    echo "=== Training Start ===" >> "$model_log_file"
    echo "Start time: $(date)" >> "$model_log_file"
    echo "Start memory usage: ${start_memory}MB" >> "$model_log_file"
    echo "Start disk free: ${start_disk}MB" >> "$model_log_file"
    echo "GPU info: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo 'NVIDIA GPU not available')" >> "$model_log_file"
    echo "=====================" >> "$model_log_file"
    
    # Run training with comprehensive logging
    if $train_cmd 2>&1 | tee -a "$model_log_file"; then
        local end_time=$(date +%s)
        local end_memory=$(free -m | grep '^Mem:' | awk '{print $3}')
        local end_disk=$(df -m . | tail -1 | awk '{print $4}')
        local duration=$((end_time - start_time))
        
        echo "=== Training Complete ===" >> "$model_log_file"
        echo "End time: $(date)" >> "$model_log_file"
        echo "Duration: ${duration} seconds ($(($duration / 60)) minutes)" >> "$model_log_file"
        echo "End memory usage: ${end_memory}MB" >> "$model_log_file"
        echo "End disk free: ${end_disk}MB" >> "$model_log_file"
        echo "Memory delta: $((end_memory - start_memory))MB" >> "$model_log_file"
        echo "Disk delta: $((start_disk - end_disk))MB" >> "$model_log_file"
        echo "GPU info: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo 'NVIDIA GPU not available')" >> "$model_log_file"
        echo "========================" >> "$model_log_file"
        
        log_message "$model_name (Run $run_id) training completed successfully!"
        log_message "Duration: ${duration} seconds ($(($duration / 60)) minutes)"
        
        # Extract key metrics from log file
        extract_metrics "$model_log_file" "$model_output_dir" "$model_name" "$run_id"
        
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        echo "=== Training Failed ===" >> "$model_log_file"
        echo "End time: $(date)" >> "$model_log_file"
        echo "Duration: ${duration} seconds ($(($duration / 60)) minutes)" >> "$model_log_file"
        echo "======================" >> "$model_log_file"
        
        log_message "ERROR: $model_name (Run $run_id) training failed!"
        log_message "Check log file: $model_log_file"
        return 1
    fi
    
    echo "----------------------------------------"
}

# Function to extract metrics from log files
extract_metrics() {
    local log_file=$1
    local output_dir=$2
    local model_name=$3
    local run_id=$4
    
    local metrics_file="$output_dir/metrics_summary.txt"
    
    echo "=== Metrics Summary for $model_name (Run $run_id) ===" > "$metrics_file"
    echo "Extracted from: $log_file" >> "$metrics_file"
    echo "Extraction time: $(date)" >> "$metrics_file"
    echo "" >> "$metrics_file"
    
    # Extract final metrics
    echo "=== Final Epoch Metrics ===" >> "$metrics_file"
    grep -E "(R@10|R@20|R@50|R@100|mR@10|mR@20|mR@50|mR@100)" "$log_file" | tail -10 >> "$metrics_file" || echo "No final metrics found" >> "$metrics_file"
    echo "" >> "$metrics_file"
    
    # Extract loss progression
    echo "=== Loss Progression ===" >> "$metrics_file"
    grep -E "(Epoch.*avg_train_loss|Epoch.*avg_val_loss)" "$log_file" >> "$metrics_file" || echo "No loss progression found" >> "$metrics_file"
    echo "" >> "$metrics_file"
    
    # Extract timing information
    echo "=== Timing Information ===" >> "$metrics_file"
    grep -E "(s/batch|m/epoch|Duration)" "$log_file" >> "$metrics_file" || echo "No timing information found" >> "$metrics_file"
    echo "" >> "$metrics_file"
    
    # Extract system resource usage
    echo "=== System Resource Usage ===" >> "$metrics_file"
    grep -E "(memory|disk|GPU)" "$log_file" >> "$metrics_file" || echo "No resource usage found" >> "$metrics_file"
    
    log_message "Metrics extracted to: $metrics_file"
}

# Function to create summary report
create_summary_report() {
    local summary_file="$OUTPUT_BASE/benchmark_summary.txt"
    
    echo "=== AG200 Benchmark Summary ===" > "$summary_file"
    echo "Generated: $(date)" >> "$summary_file"
    echo "Dataset: $DATASET" >> "$summary_file"
    echo "Mode: $MODE" >> "$summary_file"
    echo "Epochs per model: $EPOCHS" >> "$summary_file"
    echo "Total runs: $(( ${#MODELS[@]} * 3 ))" >> "$summary_file"
    echo "" >> "$summary_file"
    
    # Collect results from all runs
    for model in "${MODELS[@]}"; do
        echo "=== $model Results ===" >> "$summary_file"
        for run in 1 2 3; do
            local metrics_file="$OUTPUT_BASE/${model}_run${run}/metrics_summary.txt"
            if [ -f "$metrics_file" ]; then
                echo "--- Run $run ---" >> "$summary_file"
                grep -A 5 "Final Epoch Metrics" "$metrics_file" >> "$summary_file" || echo "No metrics available" >> "$summary_file"
                echo "" >> "$summary_file"
            else
                echo "--- Run $run ---" >> "$summary_file"
                echo "Metrics file not found: $metrics_file" >> "$summary_file"
                echo "" >> "$summary_file"
            fi
        done
        echo "" >> "$summary_file"
    done
    
    log_message "Summary report created: $summary_file"
}

# Main execution
main() {
    log_message "Starting AG200 Benchmark for Scene Graph Detection"
    log_message "=================================================="
    
    # Display system information
    get_system_info
    
    # Check prerequisites
    if [ ! -f "scripts/training/train.py" ]; then
        log_message "ERROR: Training script not found!"
        log_message "Please run this script from the project root directory."
        exit 1
    fi
    
    if [ ! -d "$DATA_PATH" ]; then
        log_message "WARNING: Dataset directory not found: $DATA_PATH"
        log_message "Please ensure the ag200 dataset is properly set up."
    fi
    
    # Record overall start time
    local overall_start=$(date +%s)
    
    # Train each model 3 times
    local run_count=0
    for model in "${MODELS[@]}"; do
        for run in 1 2 3; do
            run_count=$((run_count + 1))
            log_message "Starting run $run_count of $(( ${#MODELS[@]} * 3 )): $model (Run $run)"
            
            if ! train_model "$model" "$model" "$run"; then
                log_message "Training failed for $model (Run $run). Continuing with next run..."
            fi
        done
    done
    
    # Calculate total time
    local overall_end=$(date +%s)
    local total_duration=$((overall_end - overall_start))
    
    # Create summary report
    create_summary_report
    
    log_message "=================================================="
    log_message "AG200 Benchmark completed!"
    log_message "Total duration: $total_duration seconds ($(($total_duration / 60)) minutes)"
    log_message "Results saved to: $OUTPUT_BASE"
    log_message "Logs saved to: $LOG_DIR"
    log_message "=================================================="
}

# Run main function
main "$@"
