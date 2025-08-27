#!/bin/bash

# Batch training script for DLHM_VidSGG models
# This script trains DSG-DETR, STTran, and TEMPURA models sequentially

echo "Starting batch training for DLHM_VidSGG models..."
echo "================================================"

# Set error handling
set -e  # Exit on any error

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to train a model
train_model() {
    local model_name=$1
    local model_param=$2
    
    log_message "Starting training for $model_name..."
    echo "Command: python train.py -mode predcls -dataset action_genome -datasize large -save_path output -data_path data/action_genome -model $model_param"
    
    # Run the training command
    python train.py -mode predcls -dataset action_genome -datasize large -save_path output -data_path data/action_genome -model $model_param
    
    if [ $? -eq 0 ]; then
        log_message "$model_name training completed successfully!"
    else
        log_message "ERROR: $model_name training failed!"
        exit 1
    fi
    
    echo "----------------------------------------"
}

# Check if we're in the correct directory
if [ ! -f "train.py" ]; then
    echo "ERROR: train.py not found in current directory!"
    echo "Please run this script from the DLHM_VidSGG directory."
    exit 1
fi

# Check if data directory exists
if [ ! -d "data/action_genome" ]; then
    echo "WARNING: data/action_genome directory not found!"
    echo "Please ensure your dataset is properly set up."
fi

# Create output directory if it doesn't exist
mkdir -p output

# Train models sequentially
log_message "Training DSG-DETR..."
train_model "DSG-DETR" "dsg-detr"

log_message "Training STTran..."
train_model "STTran" "sttran"

log_message "Training TEMPURA..."
train_model "TEMPURA" "tempura"

log_message "All models trained successfully!"
echo "================================================"
log_message "Training completed for all models:"
echo "- DSG-DETR"
echo "- STTran" 
echo "- TEMPURA"
echo ""
echo "Check the output/ directory for results."