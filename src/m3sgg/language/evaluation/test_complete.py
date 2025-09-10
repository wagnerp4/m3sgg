#!/usr/bin/env python3
"""
Complete test script for language module evaluation framework.

This script tests the full functionality using mock data to avoid
external dependency issues.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
sys.path.insert(0, project_root)


def test_simple_dataset_loader():
    """Test the simple dataset loader."""
    print("Testing simple dataset loader...")

    try:
        from lib.language.evaluation.dataset_loader_simple import (
            SimpleDatasetLoader,
            create_mock_subset,
        )

        # Test loader initialization
        loader = SimpleDatasetLoader(cache_dir="test_cache")

        # Test dataset creation
        dataset = loader.create_mock_dataset(train_size=10, test_size=5)

        # Test sample info
        sample_info = loader.get_sample_info(dataset, "test", 0)

        # Test captions extraction
        captions = loader.get_captions(dataset, "test")

        print(f"✓ Dataset loader test passed!")
        print(f"  - Train samples: {len(dataset['train'])}")
        print(f"  - Test samples: {len(dataset['test'])}")
        print(f"  - Sample caption: {sample_info['caption'][:50]}...")

        return True

    except Exception as e:
        print(f"✗ Dataset loader test failed: {e}")
        return False


def test_metrics_with_mock_data():
    """Test metrics with mock data."""
    print("\nTesting metrics with mock data...")

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
        print("Metrics Results:")
        print(metrics.format_results(results))

        print("✓ Metrics test passed!")
        return True

    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False


def test_benchmark_initialization():
    """Test benchmark initialization."""
    print("\nTesting benchmark initialization...")

    try:
        from lib.language.evaluation.benchmark import SummarizationBenchmark

        # Initialize benchmark
        benchmark = SummarizationBenchmark(
            checkpoint_path="dummy/path/model.tar", device="cpu"
        )

        print("✓ Benchmark initialization test passed!")
        return True

    except Exception as e:
        print(f"✗ Benchmark initialization test failed: {e}")
        return False


def test_end_to_end_mock():
    """Test end-to-end functionality with mock data."""
    print("\nTesting end-to-end functionality...")

    try:
        from lib.language.evaluation.dataset_loader_simple import create_mock_subset
        from lib.language.evaluation.metrics import SummarizationMetrics

        # Create mock dataset
        dataset = create_mock_subset(train_size=5, test_size=3)

        # Get test captions
        test_captions = [sample["caption"] for sample in dataset["test"]]

        # Create mock predictions (simplified versions of captions)
        mock_predictions = []
        for caption in test_captions:
            # Simple mock prediction - just return the caption as-is for testing
            mock_predictions.append(caption)

        # Compute metrics
        metrics = SummarizationMetrics()
        results = metrics.compute_all_metrics(mock_predictions, test_captions)

        print("End-to-end test results:")
        print(metrics.format_results(results))

        print("✓ End-to-end test passed!")
        return True

    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")

    try:
        eval_dir = Path(__file__).parent

        required_files = [
            "__init__.py",
            "dataset_loader.py",
            "dataset_loader_simple.py",
            "metrics.py",
            "benchmark.py",
            "run_benchmark.py",
            "requirements.txt",
            "README.md",
            "test_evaluation.py",
            "test_simple.py",
            "test_complete.py",
        ]

        missing_files = []
        for file in required_files:
            if not (eval_dir / file).exists():
                missing_files.append(file)

        if missing_files:
            print(f"✗ Missing files: {missing_files}")
            return False
        else:
            print("✓ All required files present")
            return True

    except Exception as e:
        print(f"✗ File structure test failed: {e}")
        return False


def main():
    """Run all complete tests."""
    print("Running complete language module evaluation tests...")
    print("=" * 60)

    tests = [
        test_file_structure,
        test_simple_dataset_loader,
        test_metrics_with_mock_data,
        test_benchmark_initialization,
        test_end_to_end_mock,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The evaluation framework is ready to use.")
        print("\nNext steps:")
        print(
            "1. For real dataset evaluation, resolve the local 'datasets' directory conflict"
        )
        print(
            "2. Run benchmark: python lib/language/evaluation/run_benchmark.py --checkpoint <path>"
        )
        print(
            "3. Use mock data for testing: python lib/language/evaluation/dataset_loader_simple.py"
        )
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
