#!/usr/bin/env python3
"""
Extract mR@K values from all benchmark logs and create a comprehensive table.
"""

import os
import re
import pandas as pd
import glob
from pathlib import Path


def extract_final_metrics_from_log(log_file):
    """Extract final mR@K metrics from a single training log file."""
    metrics = {
        "R@10": 0,
        "R@20": 0,
        "R@50": 0,
        "R@100": 0,
        "MR@10": 0,
        "MR@20": 0,
        "MR@50": 0,
        "MR@100": 0,
    }

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract the last occurrence of each metric (final epoch)
        patterns = {
            "R@10": r"R@10: ([\d.]+)",
            "R@20": r"R@20: ([\d.]+)",
            "R@50": r"R@50: ([\d.]+)",
            "R@100": r"R@100: ([\d.]+)",
            "MR@10": r"MR@10: ([\d.]+)",
            "MR@20": r"MR@20: ([\d.]+)",
            "MR@50": r"MR@50: ([\d.]+)",
            "MR@100": r"MR@100: ([\d.]+)",
        }

        for metric, pattern in patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                # Get the last occurrence (final epoch)
                metrics[metric] = float(matches[-1])

    except Exception as e:
        print(f"Error reading {log_file}: {e}")

    return metrics


def create_mr_table(results_dir):
    """Create a comprehensive table of mR@K values for all models and modes."""

    models = ["sttran", "dsg-detr", "tempura"]
    modes = ["predcls", "sgcls", "sgdet"]

    table_data = []

    for model in models:
        for mode in modes:
            # Look for log file
            log_file = os.path.join(results_dir, "logs", f"{model}_{mode}_run1.log")

            if os.path.exists(log_file):
                metrics = extract_final_metrics_from_log(log_file)

                table_data.append(
                    {
                        "Model": model.upper(),
                        "Mode": mode,
                        "R@10": f"{metrics['R@10']:.4f}",
                        "R@20": f"{metrics['R@20']:.4f}",
                        "R@50": f"{metrics['R@50']:.4f}",
                        "R@100": f"{metrics['R@100']:.4f}",
                        "MR@10": f"{metrics['MR@10']:.4f}",
                        "MR@20": f"{metrics['MR@20']:.4f}",
                        "MR@50": f"{metrics['MR@50']:.4f}",
                        "MR@100": f"{metrics['MR@100']:.4f}",
                    }
                )
                print(f"Extracted metrics for {model} {mode}")
            else:
                print(f"Log file not found: {log_file}")
                # Add empty row for missing data
                table_data.append(
                    {
                        "Model": model.upper(),
                        "Mode": mode,
                        "R@10": "N/A",
                        "R@20": "N/A",
                        "R@50": "N/A",
                        "R@100": "N/A",
                        "MR@10": "N/A",
                        "MR@20": "N/A",
                        "MR@50": "N/A",
                        "MR@100": "N/A",
                    }
                )

    # Create DataFrame
    df = pd.DataFrame(table_data)

    return df


def main():
    results_dir = "data/benchmark/ag200"

    print("Extracting mR@K values from all benchmark logs...")
    df = create_mr_table(results_dir)

    # Save to CSV
    csv_file = os.path.join(results_dir, "mr_table.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nTable saved to: {csv_file}")

    # Print formatted table
    print("\n" + "=" * 100)
    print("COMPREHENSIVE mR@K TABLE - All Models and Modes")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    # Create summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 50)

    # Calculate averages for each model across all modes
    numeric_cols = [
        "R@10",
        "R@20",
        "R@50",
        "R@100",
        "MR@10",
        "MR@20",
        "MR@50",
        "MR@100",
    ]

    for model in df["Model"].unique():
        model_data = df[df["Model"] == model]
        print(f"\n{model} (Average across all modes):")
        for col in numeric_cols:
            if col in model_data.columns:
                # Convert to numeric, ignoring 'N/A' values
                numeric_values = pd.to_numeric(model_data[col], errors="coerce")
                avg_val = numeric_values.mean()
                if not pd.isna(avg_val):
                    print(f"  {col}: {avg_val:.4f}")

    # Calculate averages for each mode across all models
    print(f"\nMode Averages (across all models):")
    for mode in df["Mode"].unique():
        mode_data = df[df["Mode"] == mode]
        print(f"\n{mode}:")
        for col in numeric_cols:
            if col in mode_data.columns:
                numeric_values = pd.to_numeric(mode_data[col], errors="coerce")
                avg_val = numeric_values.mean()
                if not pd.isna(avg_val):
                    print(f"  {col}: {avg_val:.4f}")


if __name__ == "__main__":
    main()
