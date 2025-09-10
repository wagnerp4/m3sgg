"""Training CLI command implementation.

This module provides the training command implementation for the M3SGG CLI.

:author: M3SGG Team
:version: 0.1.0
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

# Add the src directory to the path
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Configuration will be handled by the training script


def train_command(
    config: Optional[str],
    model: Optional[str],
    dataset: Optional[str],
    mode: Optional[str],
    epochs: Optional[int],
    lr: Optional[float],
    batch_size: Optional[int],
    device: Optional[str],
    output: Optional[str],
    checkpoint: Optional[str],
    verbose: bool,
    args: List[str],
):
    """Execute the training command.

    :param config: Path to configuration file
    :type config: Optional[str]
    :param model: Model type to train
    :type model: Optional[str]
    :param dataset: Dataset to use
    :type dataset: Optional[str]
    :param mode: Training mode
    :type mode: Optional[str]
    :param epochs: Number of training epochs
    :type epochs: Optional[int]
    :param lr: Learning rate
    :type lr: Optional[float]
    :param batch_size: Batch size
    :type batch_size: Optional[int]
    :param device: Device to use
    :type device: Optional[str]
    :param output: Output directory
    :type output: Optional[str]
    :param checkpoint: Path to checkpoint file
    :type checkpoint: Optional[str]
    :param verbose: Enable verbose logging
    :type verbose: bool
    :param args: Additional arguments
    :type args: List[str]
    """
    # Prepare configuration overrides
    overrides = {}

    if model:
        overrides["model_type"] = model
    if dataset:
        overrides["dataset"] = dataset
    if mode:
        overrides["mode"] = mode
    if epochs:
        overrides["nepoch"] = epochs
    if lr:
        overrides["lr"] = lr
    if batch_size:
        overrides["batch_size"] = batch_size
    if device:
        overrides["device"] = device
    if output:
        overrides["save_path"] = output
    if checkpoint:
        overrides["ckpt"] = checkpoint

    # Set up logging level
    if verbose:
        overrides["log_level"] = "DEBUG"

    # Create configuration (will be handled by the training script)
    conf = None  # The training script will create its own config

    # Display configuration summary
    click.echo("Training Configuration:")
    click.echo(f"  Model: {model or 'sttran'}")
    click.echo(f"  Dataset: {dataset or 'action_genome'}")
    click.echo(f"  Mode: {mode or 'predcls'}")
    click.echo(f"  Epochs: {epochs or 10}")
    click.echo(f"  Learning Rate: {lr or '2e-05'}")
    click.echo(f"  Device: {device or 'cuda:0'}")
    click.echo(f"  Output: {output or 'output'}")

    if checkpoint:
        click.echo(f"  Checkpoint: {checkpoint}")

    # Import and run training
    try:
        # Import the training script
        from scripts.training.training import main as training_main

        # Set up the configuration for the training script
        # Convert CLI arguments to the format expected by the original config
        config_args = []
        if model:
            config_args.extend(["-model", model])
        if dataset:
            config_args.extend(["-dataset", dataset])
        if mode:
            config_args.extend(["-mode", mode])
        if epochs:
            config_args.extend(["-nepoch", str(epochs)])
        if lr:
            config_args.extend(["-lr", str(lr)])
        if batch_size:
            config_args.extend(["-batch_size", str(batch_size)])
        if device:
            config_args.extend(["-device", device])
        if output:
            config_args.extend(["-save_path", output])
        if checkpoint:
            config_args.extend(["-ckpt", checkpoint])

        sys.argv = ["training.py"] + config_args

        # Run training
        training_main()

    except ImportError as e:
        click.echo(f"Error importing training script: {e}", err=True)
        click.echo(
            "Make sure you're running from the project root directory.", err=True
        )
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)
        sys.exit(1)
