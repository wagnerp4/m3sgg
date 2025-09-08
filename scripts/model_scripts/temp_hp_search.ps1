# TEMPURA Hyperparameter Search Training Script (PowerShell version)
# This script performs hyperparameter search for the TEMPURA model

Write-Host "Starting TEMPURA hyperparameter search..." -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

# Function to log with timestamp
function Write-LogMessage {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor Cyan
}

# Function to train TEMPURA with specific hyperparameters

function Train-TEMPURA {
    param(
        [hashtable]$HyperParams,
        [int]$RunNumber
    )
    
    Write-LogMessage "Starting TEMPURA training run #$RunNumber..."
    
    # Create run-specific output directory
    $run_output_dir = "output/tempura_hp_search/run_$RunNumber"
    if (-not (Test-Path $run_output_dir)) {
        New-Item -ItemType Directory -Path $run_output_dir -Force | Out-Null
    }
    
    # Save hyperparameters for this run
    $hp_file = "$run_output_dir/hyperparameters.json"
    $HyperParams | ConvertTo-Json -Depth 10 | Out-File -FilePath $hp_file -Encoding UTF8
    
    # Build the command with hyperparameters
    $cmd = "python scripts/training/training.py -mode predcls -dataset action_genome -datasize large -save_path $run_output_dir -data_path data/action_genome -model tempura -nepoch 5"
    
    # Add hyperparameters to command
    foreach ($key in $HyperParams.Keys) {
        $value = $HyperParams[$key]
        if ($value -eq $true) {
            $cmd += " -$key"
        } elseif ($value -ne $null -and $value -ne "") {
            $cmd += " -$key $value"
        }
    }
    
    Write-Host "Command: $cmd" -ForegroundColor Yellow
    
    # Run the training command
    $result = Invoke-Expression $cmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-LogMessage "TEMPURA training run #$RunNumber completed successfully!" -ForegroundColor Green
        return $true
    } else {
        Write-LogMessage "ERROR: TEMPURA training run #$RunNumber failed!" -ForegroundColor Red
        return $false
    }
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

# Create output directory for hyperparameter search
$hp_search_dir = "output/tempura_hp_search"
if (-not (Test-Path $hp_search_dir)) {
    New-Item -ItemType Directory -Path $hp_search_dir -Force | Out-Null
}

# Generate hyperparameter combinations using Python script
Write-LogMessage "Generating hyperparameter combinations..."
$python_script = "scripts/tempura_hp_generator.py"

if (-not (Test-Path $python_script)) {
    Write-Host "ERROR: $python_script not found!" -ForegroundColor Red
    Write-Host "Please ensure the hyperparameter generator script exists." -ForegroundColor Red
    exit 1
}

# Run the Python script to generate hyperparameter combinations
Write-LogMessage "Running hyperparameter generator..."
$hp_combinations = python $python_script 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to generate hyperparameter combinations!" -ForegroundColor Red
    exit 1
}

# Parse the hyperparameter combinations (assuming JSON output)
try {
    $hp_list = $hp_combinations | ConvertFrom-Json
    Write-LogMessage "Generated $($hp_list.Count) hyperparameter combinations"
} catch {
    Write-Host "ERROR: Failed to parse hyperparameter combinations!" -ForegroundColor Red
    Write-Host "Python script output: $hp_combinations" -ForegroundColor Red
    Write-Host "Error details: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Train with each hyperparameter combination
$successful_runs = 0
$total_runs = $hp_list.Count

Write-LogMessage "Starting hyperparameter search with $total_runs combinations..."

for ($i = 0; $i -lt $hp_list.Count; $i++) {
    $hp_combo_obj = $hp_list[$i]
    $run_number = $i + 1
    
    # Convert PSCustomObject to Hashtable
    $hp_combo = @{}
    $hp_combo_obj.PSObject.Properties | ForEach-Object {
        $hp_combo[$_.Name] = $_.Value
    }
    
    Write-Host "----------------------------------------" -ForegroundColor Gray
    Write-LogMessage "Training run $run_number/$total_runs"
    Write-Host "Hyperparameters: $($hp_combo | ConvertTo-Json -Compress)" -ForegroundColor White
    
    $success = Train-TEMPURA -HyperParams $hp_combo -RunNumber $run_number
    
    if ($success) {
        $successful_runs++
    }
    
    Write-Host "----------------------------------------" -ForegroundColor Gray
}

# Summary
Write-LogMessage "Hyperparameter search completed!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-LogMessage "Results Summary:"
Write-Host "- Total runs: $total_runs" -ForegroundColor White
Write-Host "- Successful runs: $successful_runs" -ForegroundColor White
Write-Host "- Failed runs: $($total_runs - $successful_runs)" -ForegroundColor White
Write-Host ""

# Analyze results if any runs were successful
if ($successful_runs -gt 0) {
    Write-LogMessage "Analyzing results..."
    $analysis_script = "scripts/tempura_results_analyzer.py"
    
    if (Test-Path $analysis_script) {
        Write-Host "Running results analysis..." -ForegroundColor Yellow
        python $analysis_script --results_dir "output/tempura_hp_search" --output_dir "output/tempura_analysis"
        
        if ($LASTEXITCODE -eq 0) {
            Write-LogMessage "Results analysis completed successfully!" -ForegroundColor Green
            Write-Host "Check output/tempura_analysis/ for detailed analysis and visualizations." -ForegroundColor White
        } else {
            Write-LogMessage "Warning: Results analysis failed!" -ForegroundColor Yellow
        }
    } else {
        Write-LogMessage "Warning: Results analyzer script not found at $analysis_script" -ForegroundColor Yellow
    }
} else {
    Write-LogMessage "No successful runs to analyze." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Check the output/tempura_hp_search/ directory for individual run results." -ForegroundColor White
Write-Host "Best model checkpoints are saved in each run's directory." -ForegroundColor White 