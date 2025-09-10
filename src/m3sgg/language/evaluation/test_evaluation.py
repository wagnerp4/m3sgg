#!/usr/bin/env python3
"""
Test script for language module evaluation framework.

This script tests the basic functionality of the evaluation components
without requiring full model loading or dataset download.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
sys.path.insert(0, project_root)


def test_metrics():
    """Test metrics computation with dummy data."""
    print("Testing metrics computation...")

    try:
        from lib.language.evaluation.metrics import SummarizationMetrics

        # Initialize metrics
        metrics = SummarizationMetrics()

        # Test data
        predictions = [
            "A person is walking in the park with a dog.",
            "A man is cooking food in the kitchen.",
            "Children are playing soccer in the field.",
        ]

        references = [
            "A person walks through the park with their dog.",
            "A man prepares food in the kitchen.",
            "Kids are playing football on the field.",
        ]

        # Compute metrics
        results = metrics.compute_all_metrics(predictions, references)

        # Print results
        print(metrics.format_results(results))
        print("✓ Metrics computation test passed!")
        return True

    except Exception as e:
        print(f"✗ Metrics computation test failed: {e}")
        return False


def test_dataset_loader():
    """Test dataset loader functionality."""
    print("\nTesting dataset loader...")

    try:
        from lib.language.evaluation.dataset_loader import MSRVTTLoader

        # Initialize loader
        loader = MSRVTTLoader(cache_dir="test_cache")

        # Test metadata saving/loading
        test_metadata = {
            "train_size": 10,
            "test_size": 5,
            "random_seed": 42,
            "train_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "test_indices": [0, 1, 2, 3, 4],
            "total_samples": 15,
        }

        # Save metadata
        loader._save_subset_metadata(test_metadata, 10, 5, 42)

        # Load metadata
        loaded_metadata = loader.load_subset_metadata()

        if loaded_metadata and loaded_metadata["total_samples"] == 15:
            print("✓ Dataset loader test passed!")
            return True
        else:
            print("✗ Dataset loader test failed: metadata mismatch")
            return False

    except Exception as e:
        print(f"✗ Dataset loader test failed: {e}")
        return False


def test_imports():
    """Test that all modules can be imported."""
    print("\nTesting imports...")

    try:
        from lib.language.evaluation import (
            MSRVTTLoader,
            SummarizationMetrics,
            SummarizationBenchmark,
        )
        from lib.language.evaluation.dataset_loader import create_subset
        from lib.language.evaluation.metrics import SummarizationMetrics
        from lib.language.evaluation.benchmark import SummarizationBenchmark

        print("✓ All imports successful!")
        return True

    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running language module evaluation tests...")
    print("=" * 50)

    tests = [test_imports, test_metrics, test_dataset_loader]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The evaluation framework is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
