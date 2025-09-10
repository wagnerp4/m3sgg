"""Model utility functions for the M3SGG Streamlit application.

This module contains functions for finding and managing model checkpoints.
"""

from pathlib import Path
from typing import Dict


def find_available_checkpoints() -> Dict[str, str]:
    """Find available model checkpoints

    :return: Dictionary mapping checkpoint names to file paths
    :rtype: Dict[str, str]
    """
    checkpoints = {}
    default_checkpoint = Path(
        "data/checkpoints/action_genome/sgdet_test/model_best.tar"
    )

    if default_checkpoint.exists():
        checkpoints["action_genome/sgdet_test (default)"] = str(default_checkpoint)

    output_dir = Path("output")
    if output_dir.exists():
        checkpoints.update(
            {
                f"{dataset_dir.name}/{model_dir.name}": str(run_dir / "model_best.tar")
                for dataset_dir in output_dir.iterdir()
                if dataset_dir.is_dir()
                for model_dir in dataset_dir.iterdir()
                if model_dir.is_dir()
                for run_dir in model_dir.iterdir()
                if run_dir.is_dir()
                if (run_dir / "model_best.tar").exists()
            }
        )

    return checkpoints
