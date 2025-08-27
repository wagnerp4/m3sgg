#!/usr/bin/env python3
"""Test script to verify all SceneLLM training stages work correctly."""

import sys

from lib.config import Config


def test_stage(stage_name):
    """Test a specific training stage."""
    # Simulate command line arguments
    original_argv = sys.argv.copy()
    sys.argv = ["test", "-scenellm_training_stage", stage_name]

    try:
        conf = Config()
        result = conf.scenellm_training_stage
        print(f"✓ {stage_name}: {result}")
        return result == stage_name
    except Exception as e:
        print(f"✗ {stage_name}: ERROR - {e}")
        return False
    finally:
        sys.argv = original_argv


def test_default():
    """Test default value when no stage is specified."""
    original_argv = sys.argv.copy()
    sys.argv = ["test"]

    try:
        conf = Config()
        result = conf.scenellm_training_stage
        print(f"✓ default: {result}")
        return result == "vqvae"
    except Exception as e:
        print(f"✗ default: ERROR - {e}")
        return False
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    print("Testing SceneLLM training stage parameter...")
    print("=" * 50)

    # Test all stages
    stages = ["vqvae", "stage1", "stage2"]
    all_passed = True

    for stage in stages:
        if not test_stage(stage):
            all_passed = False

    # Test default
    if not test_default():
        all_passed = False

    print("=" * 50)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")

    print("\nTesting actual train.py command format...")
    # Test with full command format
    sys.argv = [
        "train.py",
        "-mode",
        "predcls",
        "-dataset",
        "action_genome",
        "-data_path",
        "data/action_genome",
        "-model",
        "scenellm",
        "-scenellm_training_stage",
        "vqvae",
        "-nepoch",
        "2",
    ]

    try:
        conf = Config()
        print(
            f"✓ Full command test - scenellm_training_stage: {conf.scenellm_training_stage}"
        )
        print(f"  mode: {conf.mode}")
        print(f"  model_type: {conf.model_type}")
        print(f"  dataset: {conf.dataset}")
    except Exception as e:
        print(f"✗ Full command test failed: {e}")
