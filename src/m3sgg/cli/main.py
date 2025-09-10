"""Main CLI interface for M3SGG.

This module provides the main CLI interface with command routing and help.

:author: M3SGG Team
:version: 0.1.0
"""

import sys
from typing import List, Optional

import click

from .train import train_command


@click.group()
@click.version_option(version="0.1.0", prog_name="m3sgg")
@click.pass_context
def main(ctx):
    """M3SGG: Modular, multi-modal Scene Graph Generation Framework.

    A comprehensive framework for video scene graph generation with support
    for multiple model architectures including STTRAN, Tempura, SceneLLM, OED, and VLM.
    """
    ctx.ensure_object(dict)


@main.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to configuration file"
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["sttran", "stket", "tempura", "scenellm", "oed", "vlm", "easg"]),
    help="Model type to train",
)
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(["action_genome", "EASG"]),
    help="Dataset to use",
)
@click.option(
    "--mode", type=click.Choice(["predcls", "sgcls", "sgdet"]), help="Training mode"
)
@click.option("--epochs", "-e", type=int, help="Number of training epochs")
@click.option("--lr", type=float, help="Learning rate")
@click.option("--batch-size", "-b", type=int, help="Batch size")
@click.option("--device", type=str, help="Device to use (e.g., cuda:0, cpu)")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option(
    "--checkpoint", type=click.Path(exists=True), help="Path to checkpoint file"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.argument("args", nargs=-1)
def train(
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
    """Train a scene graph generation model.

    Examples:
        m3sgg train --model sttran --dataset action_genome --mode predcls
        m3sgg train --config presets/sttran.yaml --epochs 50
        m3sgg train --model tempura --lr 1e-4 --batch-size 4
    """
    train_command(
        config=config,
        model=model,
        dataset=dataset,
        mode=mode,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        output=output,
        checkpoint=checkpoint,
        verbose=verbose,
        args=args,
    )


@main.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to configuration file"
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["sttran", "stket", "tempura", "scenellm", "oed", "vlm", "easg"]),
    help="Model type to evaluate",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to model checkpoint",
)
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(["action_genome", "EASG"]),
    help="Dataset to evaluate on",
)
@click.option("--output", "-o", type=click.Path(), help="Output directory for results")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def evaluate(
    config: Optional[str],
    model: Optional[str],
    checkpoint: str,
    dataset: Optional[str],
    output: Optional[str],
    verbose: bool,
):
    """Evaluate a trained model.

    Examples:
        m3sgg evaluate --model sttran --checkpoint checkpoints/best_model.pth
        m3sgg evaluate --config presets/sttran.yaml --checkpoint checkpoints/best_model.pth
    """
    click.echo("Evaluation functionality will be implemented in a future update.")


@main.command()
@click.option(
    "--model",
    "-m",
    type=click.Choice(["sttran", "stket", "tempura", "scenellm", "oed", "vlm", "easg"]),
    help="Model type to generate config for",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output path for configuration file"
)
@click.option("--template", "-t", is_flag=True, help="Generate from template")
def config(model: Optional[str], output: Optional[str], template: bool):
    """Generate configuration files.

    Examples:
        m3sgg config --model sttran --output my_config.yaml
        m3sgg config --template --output base_config.yaml
    """
    click.echo(
        "Config generation functionality will be implemented in a future update."
    )


if __name__ == "__main__":
    main()
