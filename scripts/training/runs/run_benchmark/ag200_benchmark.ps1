# AG200 Benchmark Script for Scene Graph Detection (PowerShell version)
# Trains STTran, DSG-DETR, and Tempura models on ag200 dataset across predcls, sgcls, sgdet
# Logs metrics per epoch/iteration for plotting R@20/50/100 and mR@20/50/100 vs iteration

param(
    [string]$Dataset = "action_genome",
    [int]$Epochs = 10,
    [int]$Fraction = 50,
    [string]$DataPath = "data/action_genome200",
    [string]$OutputBase = "data/benchmark/ag200",
    [string]$LogDir = "data/benchmark/ag200/logs"
)

# Models and modes to train
$Models = @("sttran", "dsg-detr", "tempura")
$Modes = @("predcls", "sgcls", "sgdet")

# Create directories
if (-not (Test-Path $OutputBase)) { New-Item -ItemType Directory -Path $OutputBase -Force | Out-Null }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

function Write-LogMessage {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor Cyan
}

function Get-SystemInfo {
    Write-Host "=== System Information ===" -ForegroundColor Yellow
    Write-Host "Hostname: $env:COMPUTERNAME"
    try { Write-Host "OS: $((Get-CimInstance -ClassName Win32_OperatingSystem).Caption)" } catch { Write-Host "OS: Unknown" }
    Write-Host "Python: $(python --version)"
    try {
        $gpuInfo = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>$null
        if ($gpuInfo) { Write-Host "CUDA: $gpuInfo" } else { Write-Host "CUDA: NVIDIA GPU not available" }
    } catch { Write-Host "CUDA: NVIDIA GPU not available" }
    try {
        $memory = Get-CimInstance -ClassName Win32_ComputerSystem
        $totalMemoryGB = [math]::Round($memory.TotalPhysicalMemory / 1GB, 2)
        Write-Host "Total Memory: ${totalMemoryGB}GB"
        $disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='C:'"
        $freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 2)
        Write-Host "Available Disk: ${freeSpaceGB}GB"
    } catch { }
    Write-Host "=========================" -ForegroundColor Yellow
}

function Start-ModelTraining {
    param(
        [string]$ModelName,
        [string]$ModelType,
        [string]$Mode
    )

    Write-LogMessage "Starting training for $ModelName on mode=$Mode..."

    $ModeDir = Join-Path $OutputBase $Mode
    if (-not (Test-Path $ModeDir)) { New-Item -ItemType Directory -Path $ModeDir -Force | Out-Null }

    $ModelOutputDir = Join-Path $ModeDir "${ModelName}_run1"
    $ModelLogFile = Join-Path $LogDir "${ModelName}_${Mode}_run1.log"

    if (-not (Test-Path $ModelOutputDir)) { New-Item -ItemType Directory -Path $ModelOutputDir -Force | Out-Null }

    $TrainCmd = "python scripts/training/training.py"
    $TrainCmd += " -mode $Mode"
    $TrainCmd += " -dataset $Dataset"
    $TrainCmd += " -data_path $DataPath"
    $TrainCmd += " -model $ModelType"
    $TrainCmd += " -nepoch $Epochs"
    $TrainCmd += " -fraction $Fraction"
    $TrainCmd += " -save_path `"$ModelOutputDir`""
    $TrainCmd += " -lr 1e-4"
    $TrainCmd += " -enc_layer 1"
    $TrainCmd += " -dec_layer 3"
    $TrainCmd += " -num_workers 4"
    $TrainCmd += " -seed 42"

    switch ($ModelType) {
        "dsg-detr" { $TrainCmd += " -use_matcher True" }
        "tempura" {
            $TrainCmd += " -obj_head gmm"
            $TrainCmd += " -rel_head gmm"
            $TrainCmd += " -K 3"
            $TrainCmd += " -obj_mem_compute False"
            $TrainCmd += " -rel_mem_compute False"
        }
    }

    Write-LogMessage "Training command: $TrainCmd"
    Add-Content -Path $ModelLogFile -Value "Training command: $TrainCmd"

    $StartTime = Get-Date
    $StartMemory = (Get-Process -Name "python" -ErrorAction SilentlyContinue | Measure-Object WorkingSet -Sum).Sum / 1MB
    $StartDisk = (Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace / 1MB

    Add-Content -Path $ModelLogFile -Value "=== Training Start ==="
    Add-Content -Path $ModelLogFile -Value "Start time: $StartTime"
    Add-Content -Path $ModelLogFile -Value "Start memory usage: $([math]::Round($StartMemory, 2))MB"
    Add-Content -Path $ModelLogFile -Value "Start disk free: $([math]::Round($StartDisk, 2))MB"

    try {
        $GpuInfo = nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>$null
        if ($GpuInfo) { Add-Content -Path $ModelLogFile -Value "GPU info: $GpuInfo" } else { Add-Content -Path $ModelLogFile -Value "GPU info: NVIDIA GPU not available" }
    } catch { Add-Content -Path $ModelLogFile -Value "GPU info: NVIDIA GPU not available" }

    Add-Content -Path $ModelLogFile -Value "====================="

    try {
        # Run training and capture both stdout and stderr
        $process = Start-Process -FilePath "python" -ArgumentList "scripts/training/training.py", "-mode", $Mode, "-dataset", $Dataset, "-data_path", $DataPath, "-model", $ModelType, "-nepoch", $Epochs, "-fraction", $Fraction, "-save_path", $ModelOutputDir, "-lr", "1e-4", "-enc_layer", "1", "-dec_layer", "3", "-num_workers", "4", "-seed", "42" -RedirectStandardOutput "$ModelLogFile.stdout" -RedirectStandardError "$ModelLogFile.stderr" -Wait -PassThru
        
        # Add stdout and stderr to main log
        if (Test-Path "$ModelLogFile.stdout") {
            Add-Content -Path $ModelLogFile -Value "=== STDOUT ==="
            Get-Content "$ModelLogFile.stdout" | Add-Content -Path $ModelLogFile
            Remove-Item "$ModelLogFile.stdout"
        }
        if (Test-Path "$ModelLogFile.stderr") {
            Add-Content -Path $ModelLogFile -Value "=== STDERR ==="
            Get-Content "$ModelLogFile.stderr" | Add-Content -Path $ModelLogFile
            Remove-Item "$ModelLogFile.stderr"
        }

        $EndTime = Get-Date
        $EndMemory = (Get-Process -Name "python" -ErrorAction SilentlyContinue | Measure-Object WorkingSet -Sum).Sum / 1MB
        $EndDisk = (Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace / 1MB
        $Duration = ($EndTime - $StartTime).TotalSeconds

        Add-Content -Path $ModelLogFile -Value "=== Training Complete ==="
        Add-Content -Path $ModelLogFile -Value "End time: $EndTime"
        Add-Content -Path $ModelLogFile -Value "Duration: $Duration seconds ($([math]::Round($Duration / 60, 2)) minutes)"
        Add-Content -Path $ModelLogFile -Value "Exit code: $($process.ExitCode)"
        Add-Content -Path $ModelLogFile -Value "End memory usage: $([math]::Round($EndMemory, 2))MB"
        Add-Content -Path $ModelLogFile -Value "End disk free: $([math]::Round($EndDisk, 2))MB"
        Add-Content -Path $ModelLogFile -Value "Memory delta: $([math]::Round($EndMemory - $StartMemory, 2))MB"
        Add-Content -Path $ModelLogFile -Value "Disk delta: $([math]::Round($StartDisk - $EndDisk, 2))MB"

        try {
            $GpuInfo = nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>$null
            if ($GpuInfo) { Add-Content -Path $ModelLogFile -Value "GPU info: $GpuInfo" } else { Add-Content -Path $ModelLogFile -Value "GPU info: NVIDIA GPU not available" }
        } catch { Add-Content -Path $ModelLogFile -Value "GPU info: NVIDIA GPU not available" }

        Add-Content -Path $ModelLogFile -Value "========================"

        if ($process.ExitCode -eq 0) {
            Write-LogMessage "$ModelName mode=$Mode training completed successfully"
        } else {
            Write-LogMessage "WARNING: $ModelName mode=$Mode training completed with exit code $($process.ExitCode)"
        }

        Export-Metrics -LogFile $ModelLogFile -OutputDir $ModelOutputDir -ModelName $ModelName -Mode $Mode
        return $true
    } catch {
        $EndTime = Get-Date
        $Duration = ($EndTime - $StartTime).TotalSeconds
        Add-Content -Path $ModelLogFile -Value "=== Training Failed ==="
        Add-Content -Path $ModelLogFile -Value "End time: $EndTime"
        Add-Content -Path $ModelLogFile -Value "Duration: $Duration seconds ($([math]::Round($Duration / 60, 2)) minutes)"
        Add-Content -Path $ModelLogFile -Value "Error: $($_.Exception.Message)"
        Add-Content -Path $ModelLogFile -Value "======================"
        Write-LogMessage "ERROR: $ModelName mode=$Mode training failed"
        return $false
    }
}

function Export-Metrics {
    param(
        [string]$LogFile,
        [string]$OutputDir,
        [string]$ModelName,
        [string]$Mode
    )

    $MetricsFile = Join-Path $OutputDir "metrics_summary.txt"

    Add-Content -Path $MetricsFile -Value "=== Metrics Summary for $ModelName mode=$Mode ==="
    Add-Content -Path $MetricsFile -Value "Extracted from: $LogFile"
    Add-Content -Path $MetricsFile -Value "Extraction time: $(Get-Date)"
    Add-Content -Path $MetricsFile -Value ""

    Add-Content -Path $MetricsFile -Value "=== Final Epoch Metrics ==="
    $FinalMetrics = Select-String -Path $LogFile -Pattern "(R@10|R@20|R@50|R@100|MR@10|MR@20|MR@50|MR@100)" | Select-Object -Last 10
    if ($FinalMetrics) { $FinalMetrics | ForEach-Object { Add-Content -Path $MetricsFile -Value $_.Line } } else { Add-Content -Path $MetricsFile -Value "No final metrics found" }
    Add-Content -Path $MetricsFile -Value ""

    Add-Content -Path $MetricsFile -Value "=== Loss Progression ==="
    $LossProgression = Select-String -Path $LogFile -Pattern "(avg_train_loss|avg_val_loss)"
    if ($LossProgression) { $LossProgression | ForEach-Object { Add-Content -Path $MetricsFile -Value $_.Line } } else { Add-Content -Path $MetricsFile -Value "No loss progression found" }
    Add-Content -Path $MetricsFile -Value ""

    Add-Content -Path $MetricsFile -Value "=== Timing Information ==="
    $TimingInfo = Select-String -Path $LogFile -Pattern "(s/batch|m/epoch|Duration|Best model achieved)"
    if ($TimingInfo) { $TimingInfo | ForEach-Object { Add-Content -Path $MetricsFile -Value $_.Line } } else { Add-Content -Path $MetricsFile -Value "No timing information found" }
    Add-Content -Path $MetricsFile -Value ""

    Add-Content -Path $MetricsFile -Value "=== System Resource Usage ==="
    $ResourceUsage = Select-String -Path $LogFile -Pattern "(memory|disk|GPU|Filtered out|Total:|Used:|Free:)"
    if ($ResourceUsage) { $ResourceUsage | ForEach-Object { Add-Content -Path $MetricsFile -Value $_.Line } } else { Add-Content -Path $MetricsFile -Value "No resource usage found" }

    Write-LogMessage "Metrics extracted to: $MetricsFile"
}

function New-SummaryReport {
    $SummaryFile = Join-Path $OutputBase "benchmark_summary.txt"

    Add-Content -Path $SummaryFile -Value "=== AG200 Benchmark Summary ==="
    Add-Content -Path $SummaryFile -Value "Generated: $(Get-Date)"
    Add-Content -Path $SummaryFile -Value "Dataset: $Dataset"
    Add-Content -Path $SummaryFile -Value "Epochs per run: $Epochs"
    Add-Content -Path $SummaryFile -Value "Fraction: $Fraction (1=all, 2=half, 10=10%, 50=2%)"
    Add-Content -Path $SummaryFile -Value "Total runs: $($Models.Count * $Modes.Count)"
    Add-Content -Path $SummaryFile -Value ""

    foreach ($Mode in $Modes) {
        Add-Content -Path $SummaryFile -Value "=== Mode: $Mode ==="
        foreach ($Model in $Models) {
            Add-Content -Path $SummaryFile -Value "--- $Model Results ---"
            $RunDir = Join-Path (Join-Path $OutputBase $Mode) "${Model}_run1"
            $MetricsFile = Join-Path $RunDir "metrics_summary.txt"
            Add-Content -Path $SummaryFile -Value "Results:"
            if (Test-Path $MetricsFile) {
                $FinalMetrics = Select-String -Path $MetricsFile -Pattern "Final Epoch Metrics" -Context 0, 5
                if ($FinalMetrics) { $FinalMetrics | ForEach-Object { Add-Content -Path $SummaryFile -Value $_.Line } } else { Add-Content -Path $SummaryFile -Value "No metrics available" }
            } else {
                Add-Content -Path $SummaryFile -Value "Metrics file not found: $MetricsFile"
            }
            Add-Content -Path $SummaryFile -Value ""
        }
        Add-Content -Path $SummaryFile -Value ""
    }

    Write-LogMessage "Summary report created: $SummaryFile"
}

function Invoke-Ag200Benchmark {
    Write-LogMessage "Starting AG200 Benchmark for Scene Graph Generation"
    Write-Host "==================================================" -ForegroundColor Green

    Get-SystemInfo

    if (-not (Test-Path "scripts/training/training.py")) {
        Write-LogMessage "ERROR: Training script not found! Run from project root."
        exit 1
    }

    if (-not (Test-Path $DataPath)) {
        Write-LogMessage "WARNING: Dataset directory not found: $DataPath"
        Write-LogMessage "Please ensure the ag200 dataset is properly set up."
    }

    $OverallStart = Get-Date

    $RunCount = 0
    foreach ($Mode in $Modes) {
        foreach ($Model in $Models) {
            $RunCount++
            Write-LogMessage "Starting run $RunCount of $($Models.Count * $Modes.Count): $Model mode=$Mode"
            if (-not (Start-ModelTraining -ModelName $Model -ModelType $Model -Mode $Mode)) {
                Write-LogMessage "Training failed for $Model mode=$Mode. Continuing..."
            }
        }
    }

    $OverallEnd = Get-Date
    $TotalDuration = ($OverallEnd - $OverallStart).TotalSeconds

    New-SummaryReport

    Write-Host "==================================================" -ForegroundColor Green
    Write-LogMessage "AG200 Benchmark completed"
    Write-LogMessage "Total duration: $TotalDuration seconds ($([math]::Round($TotalDuration / 60, 2)) minutes)"
    Write-LogMessage "Results saved to: $OutputBase"
    Write-LogMessage "Logs saved to: $LogDir"
    Write-Host "==================================================" -ForegroundColor Green

    try {
        Write-LogMessage "Generating plots and tables"
        python scripts/training/runs/run_benchmark/plot_ag200_benchmark.py --results_dir "$OutputBase" --output_dir "$OutputBase/plots" | Out-Null
    } catch { Write-LogMessage "Plotting step failed: $($_.Exception.Message)" }
}

Invoke-Ag200Benchmark
