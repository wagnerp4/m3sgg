#!/usr/bin/env python3
"""Test script to verify actual training commands work correctly."""

import sys

from m3sgg.core.config.config import Config


def test_command(stage_name):
    """Test a command with specific training stage."""
    print(f"\nTesting stage: {stage_name}")
    print("-" * 60)

    # Simulate the actual command you would run
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
        stage_name,
        "-nepoch",
        "2",
    ]

    try:
        conf = Config()
        print(
            f"Command: python train.py -mode predcls -dataset action_genome -data_path data/action_genome -model scenellm -scenellm_training_stage {stage_name} -nepoch 2"
        )
        print(f"✓ Parsed scenellm_training_stage: {conf.scenellm_training_stage}")
        print(f"  mode: {conf.mode}")
        print(f"  model_type: {conf.model_type}")
        print(f"  dataset: {conf.dataset}")
        print(f"  nepoch: {conf.nepoch}")
        assert conf.scenellm_training_stage == stage_name
    except Exception as e:
        print(f"✗ Command failed: {e}")
        raise


if __name__ == "__main__":
    print("Testing Real SceneLLM Training Commands")
    print("=" * 60)

    # Test all three stages
    commands = [
        ("vqvae", "Stage 1: VQ-VAE Pretraining"),
        ("stage1", "Stage 2: SGG Training (Frozen VQ-VAE & LLM)"),
        ("stage2", "Stage 3: End-to-End Fine-tuning with LoRA"),
    ]

    all_passed = True
    for stage, description in commands:
        if not test_command(stage, description):
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All command tests passed!")
        print("\nYou can now use any of these commands:")
        print(
            "• python train.py -mode predcls -dataset action_genome -data_path data/action_genome -model scenellm -scenellm_training_stage vqvae -nepoch 2"
        )
        print(
            "• python train.py -mode predcls -dataset action_genome -data_path data/action_genome -model scenellm -scenellm_training_stage stage1 -nepoch 2"
        )
        print(
            "• python train.py -mode predcls -dataset action_genome -data_path data/action_genome -model scenellm -scenellm_training_stage stage2 -nepoch 2"
        )
    else:
        print("✗ Some command tests failed!")
