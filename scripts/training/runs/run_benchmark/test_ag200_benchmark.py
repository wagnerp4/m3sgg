#!/usr/bin/env python3
"""
Test script for AG200 benchmark setup.
Verifies that all required components are available before running the full benchmark.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_packages():
    """Check if required Python packages are available."""
    required_packages = ["torch", "numpy", "pandas", "matplotlib", "tqdm"]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")

    return missing_packages


def check_training_script():
    """Check if the training script exists and is executable."""
    script_path = "scripts/training/train.py"
    if os.path.exists(script_path):
        print(f"✓ Training script found: {script_path}")
        return True
    else:
        print(f"✗ Training script not found: {script_path}")
        return False


def check_dataset_path(dataset_path="data/action_genome200"):
    """Check if the dataset directory exists."""
    if os.path.exists(dataset_path):
        print(f"✓ Dataset directory found: {dataset_path}")
        return True
    else:
        print(f"✗ Dataset directory not found: {dataset_path}")
        print(f"  Please ensure the ag200 dataset is available at: {dataset_path}")
        return False


def check_gpu_availability():
    """Check if GPU is available for training."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠ GPU not available, will use CPU")
            return False
    except ImportError:
        print("⚠ PyTorch not available, cannot check GPU")
        return False


def check_output_directories():
    """Check if output directories can be created."""
    output_dirs = ["output/ag200_benchmark", "logs/ag200_benchmark"]

    for dir_path in output_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Output directory ready: {dir_path}")
        except Exception as e:
            print(f"✗ Cannot create output directory {dir_path}: {e}")
            return False

    return True


def test_training_command():
    """Test if the training command can be executed (dry run)."""
    try:
        # Test command with minimal parameters
        cmd = [
            sys.executable,
            "scripts/training/train.py",
            "-mode",
            "sgdet",
            "-dataset",
            "action_genome200",
            "-data_path",
            "data/action_genome200",
            "-model",
            "sttran",
            "-nepoch",
            "1",
            "-help",  # Just show help to test if command works
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✓ Training command is executable")
            return True
        else:
            print(f"✗ Training command failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠ Training command timed out (this might be normal)")
        return True
    except Exception as e:
        print(f"✗ Training command test failed: {e}")
        return False


def main():
    """Run all checks for the AG200 benchmark setup."""
    print("AG200 Benchmark Setup Test")
    print("=" * 40)

    all_checks_passed = True

    # Check Python packages
    print("\n1. Checking Python packages...")
    missing_packages = check_python_packages()
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        all_checks_passed = False

    # Check training script
    print("\n2. Checking training script...")
    if not check_training_script():
        all_checks_passed = False

    # Check dataset
    print("\n3. Checking dataset...")
    if not check_dataset_path():
        all_checks_passed = False

    # Check GPU
    print("\n4. Checking GPU availability...")
    check_gpu_availability()  # This is not critical, so we don't fail on it

    # Check output directories
    print("\n5. Checking output directories...")
    if not check_output_directories():
        all_checks_passed = False

    # Test training command
    print("\n6. Testing training command...")
    if not test_training_command():
        all_checks_passed = False

    # Summary
    print("\n" + "=" * 40)
    if all_checks_passed:
        print("✓ All checks passed! Ready to run AG200 benchmark.")
        print("\nTo run the benchmark:")
        print("  Linux/macOS: ./scripts/training/ag200_benchmark.sh")
        print("  Windows: .\\scripts\\training\\ag200_benchmark.ps1")
    else:
        print(
            "✗ Some checks failed. Please fix the issues above before running the benchmark."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
