#!/usr/bin/env python3
"""Test script to verify all SceneLLM training stages work correctly."""

import sys

from m3sgg.core.config.config import Config


def test_stage_vqvae():
    """Test vqvae training stage."""
    original_argv = sys.argv.copy()
    sys.argv = ["test", "-scenellm_training_stage", "vqvae"]

    try:
        conf = Config()
        result = conf.scenellm_training_stage
        print(f"✓ vqvae: {result}")
        assert result == "vqvae"
    except Exception as e:
        print(f"✗ vqvae: ERROR - {e}")
        raise
    finally:
        sys.argv = original_argv


def test_stage_stage1():
    """Test stage1 training stage."""
    original_argv = sys.argv.copy()
    sys.argv = ["test", "-scenellm_training_stage", "stage1"]

    try:
        conf = Config()
        result = conf.scenellm_training_stage
        print(f"✓ stage1: {result}")
        assert result == "stage1"
    except Exception as e:
        print(f"✗ stage1: ERROR - {e}")
        raise
    finally:
        sys.argv = original_argv


def test_stage_stage2():
    """Test stage2 training stage."""
    original_argv = sys.argv.copy()
    sys.argv = ["test", "-scenellm_training_stage", "stage2"]

    try:
        conf = Config()
        result = conf.scenellm_training_stage
        print(f"✓ stage2: {result}")
        assert result == "stage2"
    except Exception as e:
        print(f"✗ stage2: ERROR - {e}")
        raise
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
        assert result == "vqvae"
    except Exception as e:
        print(f"✗ default: ERROR - {e}")
        raise
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    print("Testing SceneLLM training stage parameter...")
    print("=" * 50)

    # Test all stages
    test_functions = [test_stage_vqvae, test_stage_stage1, test_stage_stage2]
    all_passed = True

    for test_func in test_functions:
        try:
            test_func()
        except Exception:
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
