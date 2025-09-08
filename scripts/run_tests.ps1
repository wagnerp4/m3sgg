#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Test runner script for M3SGG project

.DESCRIPTION
    This script provides convenient commands to run different types of tests
    with appropriate configurations and options.

.PARAMETER TestType
    Type of tests to run: unit, integration, performance, or all

.PARAMETER PythonVersion
    Python version to use for testing

.PARAMETER Coverage
    Generate coverage report

.PARAMETER Verbose
    Enable verbose output

.PARAMETER Parallel
    Run tests in parallel

.EXAMPLE
    .\run_tests.ps1 -TestType unit
    .\run_tests.ps1 -TestType all -Coverage
    .\run_tests.ps1 -TestType performance -Verbose
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("unit", "integration", "performance", "all")]
    [string]$TestType = "all",
    
    [string]$PythonVersion = "3.10",
    
    [switch]$Coverage,
    
    [switch]$Verbose,
    
    [switch]$Parallel,
    
    [switch]$Help
)

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Definition -Detailed
    exit 0
}

Write-Host "M3SGG Test Runner" -ForegroundColor Green
Write-Host "=================" -ForegroundColor Green
Write-Host ""

# Set up environment
$env:PYTHONPATH = "$PWD\src;$PWD;$env:PYTHONPATH"

# Build pytest command
$pytestArgs = @()

switch ($TestType) {
    "unit" {
        $pytestArgs += "tests/unit/"
        Write-Host "Running unit tests..." -ForegroundColor Yellow
    }
    "integration" {
        $pytestArgs += "tests/integration/"
        Write-Host "Running integration tests..." -ForegroundColor Yellow
    }
    "performance" {
        $pytestArgs += "tests/performance/"
        Write-Host "Running performance tests..." -ForegroundColor Yellow
    }
    "all" {
        $pytestArgs += "tests/"
        Write-Host "Running all tests..." -ForegroundColor Yellow
    }
}

if ($Coverage) {
    $pytestArgs += @("--cov=src/m3sgg", "--cov-report=html", "--cov-report=term-missing")
    Write-Host "Coverage reporting enabled" -ForegroundColor Cyan
}

if ($Verbose) {
    $pytestArgs += "-v"
    Write-Host "Verbose output enabled" -ForegroundColor Cyan
}

if ($Parallel) {
    $pytestArgs += "-n auto"
    Write-Host "Parallel execution enabled" -ForegroundColor Cyan
}

# Add common options
$pytestArgs += @("--tb=short", "--durations=10")

Write-Host ""
Write-Host "Command: pytest $($pytestArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

# Run tests
try {
    & pytest @pytestArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "All tests passed!" -ForegroundColor Green
        
        if ($Coverage) {
            Write-Host "Coverage report generated in htmlcov/index.html" -ForegroundColor Cyan
        }
    } else {
        Write-Host ""
        Write-Host "Some tests failed. Exit code: $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "Error running tests: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
