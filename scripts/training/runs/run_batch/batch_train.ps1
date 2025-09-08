# Batch training script for DLHM_VidSGG models (PowerShell version)
# This script trains DSG-DETR, STTran, and TEMPURA models sequentially

Write-Host "Starting batch training for DLHM_VidSGG models..." -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Function to log with timestamp
function Write-LogMessage {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor Cyan
}

# Function to train a model
function Train-Model {
    param(
        [string]$ModelName,
        [string]$ModelParam
    )
    
    Write-LogMessage "Starting training for $ModelName..."
    Write-Host "Command: python scripts/training/training.py -mode predcls -dataset action_genome -datasize large -save_path output -data_path data/action_genome9000 -model $ModelParam" -ForegroundColor Yellow
    
    # TRAINING COMMAND
    $result = python scripts/training/training.py -mode predcls -dataset action_genome -datasize large -save_path output -data_path data/action_genome -model $ModelParam -nepoch 1
    
    if ($LASTEXITCODE -eq 0) {
        Write-LogMessage "$ModelName training completed successfully!" -ForegroundColor Green
    } else {
        Write-LogMessage "ERROR: $ModelName training failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "----------------------------------------" -ForegroundColor Gray
}

# Check if we're in the correct directory
if (-not (Test-Path "train.py")) {
    Write-Host "ERROR: train.py not found in current directory!" -ForegroundColor Red
    Write-Host "Please run this script from the DLHM_VidSGG directory." -ForegroundColor Red
    exit 1
}

# Check if data directory exists
if (-not (Test-Path "data/action_genome")) {
    Write-Host "WARNING: data/action_genome directory not found!" -ForegroundColor Yellow
    Write-Host "Please ensure your dataset is properly set up." -ForegroundColor Yellow
}

# Create output directory if it doesn't exist
if (-not (Test-Path "output")) {
    New-Item -ItemType Directory -Path "output" -Force | Out-Null
}

# Train models sequentially
Write-LogMessage "Training DSG-DETR..."
Train-Model -ModelName "DSG-DETR" -ModelParam "dsg-detr"

Write-LogMessage "Training STTran..."
Train-Model -ModelName "STTran" -ModelParam "sttran"

Write-LogMessage "Training TEMPURA..."
Train-Model -ModelName "TEMPURA" -ModelParam "tempura"

Write-LogMessage "Training STKET..."
Train-Model -ModelName "STKET" -ModelParam "stket"

Write-LogMessage "All models trained successfully!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-LogMessage "Training completed for all models:"
Write-Host "- DSG-DETR" -ForegroundColor White
Write-Host "- STTran" -ForegroundColor White
Write-Host "- TEMPURA" -ForegroundColor White
Write-Host ""
Write-Host "Check the output/ directory for results." -ForegroundColor White 