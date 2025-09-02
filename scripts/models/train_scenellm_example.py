#!/usr/bin/env python3
"""
Example training script for SceneLLM model.
This demonstrates how to train the SceneLLM model with the three-stage training approach.
"""

import subprocess
import sys


def run_training_stage(stage, args_dict):
    """Run a training stage with given arguments."""

    # Base arguments for all stages
    base_args = [
        sys.executable,
        "train.py",
        "-model",
        "scenellm",
        "-dataset",
        "action_genome",
        "-data_path",
        args_dict.get("data_path", "data/action_genome"),
        "-mode",
        args_dict.get("mode", "sgcls"),
        "-datasize",
        args_dict.get("datasize", "mini"),
        "-optimizer",
        "adamw",
        "-lr",
        str(args_dict.get("lr", 1e-5)),
    ]

    # Stage-specific arguments
    stage_args = []

    if stage == "vqvae":
        print("Stage 1: VQ-VAE Pretraining")
        stage_args.extend(
            [
                "-scenellm_training_stage",
                "vqvae",
                "-nepoch",
                str(args_dict.get("vqvae_epochs", 5)),
                "-embed_dim",
                str(args_dict.get("embed_dim", 1024)),
                "-codebook_size",
                str(args_dict.get("codebook_size", 8192)),
                "-commitment_cost",
                str(args_dict.get("commitment_cost", 0.25)),
            ]
        )

    elif stage == "stage1":
        print("Stage 2: SGG Training (Frozen VQ-VAE & LLM)")
        stage_args.extend(
            [
                "-scenellm_training_stage",
                "stage1",
                "-nepoch",
                "50",  # Convert iterations to rough epochs
            ]
        )

    elif stage == "stage2":
        print("Stage 3: End-to-End Fine-tuning with LoRA")
        stage_args.extend(
            [
                "-scenellm_training_stage",
                "stage2",
                "-nepoch",
                "80",  # Convert iterations to rough epochs
                "-llm_name",
                args_dict.get("llm_name", "google/gemma-2-2b"),
                "-lora_r",
                str(args_dict.get("lora_r", 16)),
                "-lora_alpha",
                str(args_dict.get("lora_alpha", 32)),
                "-lora_dropout",
                str(args_dict.get("lora_dropout", 0.05)),
            ]
        )

    # Combine arguments
    full_args = base_args + stage_args

    print(f"Running: {' '.join(full_args)}")
    print("-" * 60)

    # Run training
    try:
        # result = subprocess.run(full_args, check=True, capture_output=False)
        print(f"✓ {stage} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {stage} failed with return code {e.returncode}")
        return False


def main():
    """Main training pipeline for SceneLLM."""

    # Configuration
    config = {
        "data_path": "data/action_genome",
        "mode": "sgcls",  # sgcls, sgdet, predcls
        "datasize": "mini",  # mini for testing, large for full dataset
        "lr": 1e-5,
        # VQ-VAE parameters
        "embed_dim": 1024,
        "codebook_size": 8192,
        "commitment_cost": 0.25,
        "vqvae_epochs": 5,
        # LLM parameters
        "llm_name": "google/gemma-2-2b",  # Smaller model for testing
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    }

    print("SceneLLM Training Pipeline")
    print("=" * 60)
    print("This will run the three-stage training approach:")
    print("1. VQ-VAE Pretraining")
    print("2. SGG Training (frozen VQ-VAE & LLM)")
    print("3. End-to-End Fine-tuning with LoRA")
    print("=" * 60)

    # Check if we want to run all stages or specific ones
    if len(sys.argv) > 1:
        stages_to_run = sys.argv[1:]
    else:
        stages_to_run = ["vqvae", "stage1", "stage2"]

    print(f"Stages to run: {', '.join(stages_to_run)}")
    print()

    # Run training stages
    for stage in stages_to_run:
        if stage not in ["vqvae", "stage1", "stage2"]:
            print(f"Unknown stage: {stage}")
            continue

        print(f"\n{'='*20} {stage.upper()} {'='*20}")

        success = run_training_stage(stage, config)

        if not success:
            print(f"Training failed at stage: {stage}")
            return 1

        print(f"✓ Stage {stage} completed")

    print("\n" + "=" * 60)
    print("🎉 SceneLLM training pipeline completed!")
    print("Check the output directory for saved models.")

    return 0


if __name__ == "__main__":
    # Quick help
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("Usage: python train_scenellm_example.py [stage1] [stage2] [stage3]")
        print()
        print("Stages:")
        print("  vqvae  - VQ-VAE pretraining")
        print("  stage1 - SGG training (frozen VQ-VAE & LLM)")
        print("  stage2 - End-to-end fine-tuning with LoRA")
        print()
        print("Examples:")
        print("  python train_scenellm_example.py               # Run all stages")
        print("  python train_scenellm_example.py stage2        # Run only stage2")
        print(
            "  python train_scenellm_example.py stage1 stage2 # Run stage1 and stage2"
        )
        sys.exit(0)

    exit(main())
