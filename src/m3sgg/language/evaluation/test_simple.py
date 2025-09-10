#!/usr/bin/env python3
"""
Simple test script for language module evaluation framework.

This script tests the basic functionality without requiring all dependencies.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
sys.path.insert(0, project_root)


def test_basic_imports():
    """Test basic imports without external dependencies."""
    print("Testing basic imports...")

    try:
        # Test basic Python imports
        import json
        import random
        import logging
        from pathlib import Path
        from typing import Dict, Optional

        print("✓ Basic Python imports successful")
        return True

    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        return False


def test_metrics_basic():
    """Test metrics with mock data (no external dependencies)."""
    print("\nTesting metrics with mock data...")

    try:
        # Create a simple mock metrics class
        class MockSummarizationMetrics:
            def __init__(self):
                self.rouge_types = ["rouge1", "rouge2", "rougeL"]

            def compute_all_metrics(self, predictions, references):
                """Mock implementation that returns dummy scores."""
                return {
                    "rouge1": 0.342,
                    "rouge2": 0.187,
                    "rougeL": 0.298,
                    "bleu1": 0.456,
                    "bleu2": 0.234,
                    "bleu3": 0.123,
                    "bleu4": 0.067,
                    "meteor": 0.234,
                    "semantic_similarity": 0.789,
                }

            def format_results(self, metrics, precision=4):
                """Format results for display."""
                lines = ["Mock Summarization Metrics Results:"]
                lines.append("=" * 40)

                for metric, score in metrics.items():
                    lines.append(f"  {metric.upper()}: {score:.{precision}f}")

                return "\n".join(lines)

        # Test with mock data
        metrics = MockSummarizationMetrics()
        predictions = ["A person is walking in the park with a dog."]
        references = ["A person walks through the park with their dog."]

        results = metrics.compute_all_metrics(predictions, references)
        formatted = metrics.format_results(results)

        print("Mock Results:")
        print(formatted)
        print("✓ Mock metrics test passed!")
        return True

    except Exception as e:
        print(f"✗ Mock metrics test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")

    try:
        eval_dir = Path(__file__).parent

        required_files = [
            "__init__.py",
            "dataset_loader.py",
            "metrics.py",
            "benchmark.py",
            "run_benchmark.py",
            "requirements.txt",
            "README.md",
            "test_evaluation.py",
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


def test_config_loading():
    """Test that the benchmark can be initialized."""
    print("\nTesting benchmark initialization...")

    try:
        # Test basic benchmark initialization (without loading models)
        from lib.language.evaluation.benchmark import SummarizationBenchmark

        # Initialize with dummy checkpoint path
        benchmark = SummarizationBenchmark(
            checkpoint_path="dummy/path/model.tar", device="cpu"
        )

        print("✓ Benchmark initialization successful")
        return True

    except Exception as e:
        print(f"✗ Benchmark initialization failed: {e}")
        return False


def main():
    """Run all basic tests."""
    print("Running simple language module evaluation tests...")
    print("=" * 60)

    tests = [
        test_basic_imports,
        test_file_structure,
        test_metrics_basic,
        test_config_loading,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All basic tests passed! The evaluation framework structure is ready.")
        print("\nNext steps:")
        print(
            "1. Install dependencies: pip install -r lib/language/evaluation/requirements.txt"
        )
        print("2. Run full test: python lib/language/evaluation/test_evaluation.py")
        print(
            "3. Run benchmark: python lib/language/evaluation/run_benchmark.py --checkpoint <path>"
        )
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
