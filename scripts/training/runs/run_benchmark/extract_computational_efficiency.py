#!/usr/bin/env python3
"""
Extract computational efficiency metrics from all benchmark logs and create a comprehensive table.
"""

import os
import re
import pandas as pd
import glob
from pathlib import Path
import numpy as np


def extract_computational_metrics_from_log(log_file):
    """Extract computational efficiency metrics from a single training log file."""
    metrics = {
        "total_duration_seconds": 0,
        "total_duration_minutes": 0,
        "avg_epoch_time_seconds": 0,
        "avg_epoch_time_minutes": 0,
        "total_epochs": 0,
        "memory_usage_mb": 0,
        "disk_usage_mb": 0,
        "gpu_memory_used_mb": 0,
        "gpu_memory_total_mb": 0,
        "gpu_utilization_percent": 0,
        "avg_train_loss": 0,
        "final_val_loss": 0,
        "final_r20": 0,
        "final_mr20": 0,
        "training_samples": 0,
        "test_samples": 0,
        "throughput_samples_per_second": 0,
    }

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract duration information
        duration_match = re.search(
            r"Duration: ([\d.]+) seconds \(([\d.]+) minutes\)", content
        )
        if duration_match:
            metrics["total_duration_seconds"] = float(duration_match.group(1))
            metrics["total_duration_minutes"] = float(duration_match.group(2))

        # Extract epoch information
        epoch_matches = re.findall(r"Epoch (\d+) \|", content)
        if epoch_matches:
            metrics["total_epochs"] = max([int(epoch) for epoch in epoch_matches]) + 1
            if metrics["total_duration_seconds"] > 0:
                metrics["avg_epoch_time_seconds"] = (
                    metrics["total_duration_seconds"] / metrics["total_epochs"]
                )
                metrics["avg_epoch_time_minutes"] = (
                    metrics["avg_epoch_time_seconds"] / 60
                )

        # Extract memory usage
        memory_match = re.search(r"End memory usage: ([\d.]+)MB", content)
        if memory_match:
            metrics["memory_usage_mb"] = float(memory_match.group(1))

        # Extract disk usage
        disk_match = re.search(r"Disk delta: ([\d.]+)MB", content)
        if disk_match:
            metrics["disk_usage_mb"] = float(disk_match.group(1))

        # Extract GPU information
        gpu_matches = re.findall(
            r"GPU info: .*?, ([\d.]+), ([\d.]+), ([\d.]+)", content
        )
        if gpu_matches:
            # Get the last GPU info (end of training)
            last_gpu = gpu_matches[-1]
            metrics["gpu_memory_used_mb"] = float(last_gpu[0])
            metrics["gpu_memory_total_mb"] = float(last_gpu[1])
            metrics["gpu_utilization_percent"] = float(last_gpu[2])

        # Extract final performance metrics
        final_r20_match = re.search(r"R@20: ([\d.]+)", content)
        if final_r20_match:
            # Get the last R@20 value
            r20_matches = re.findall(r"R@20: ([\d.]+)", content)
            if r20_matches:
                metrics["final_r20"] = float(r20_matches[-1])

        final_mr20_match = re.search(r"MR@20: ([\d.]+)", content)
        if final_mr20_match:
            # Get the last MR@20 value
            mr20_matches = re.findall(r"MR@20: ([\d.]+)", content)
            if mr20_matches:
                metrics["final_mr20"] = float(mr20_matches[-1])

        # Extract loss information
        train_loss_matches = re.findall(r"avg_train_loss=([\d.]+)", content)
        if train_loss_matches:
            metrics["avg_train_loss"] = np.mean([float(x) for x in train_loss_matches])

        val_loss_matches = re.findall(r"avg_val_loss=([\d.]+)", content)
        if val_loss_matches:
            metrics["final_val_loss"] = float(val_loss_matches[-1])

        # Extract dataset size information
        # First try to get actual sample counts
        train_samples_match = re.search(r"(\d+)/\d+ training samples", content)
        if train_samples_match:
            metrics["training_samples"] = int(train_samples_match.group(1))

        test_samples_match = re.search(r"(\d+)/\d+ test samples", content)
        if test_samples_match:
            metrics["test_samples"] = int(test_samples_match.group(1))

        # If no sample counts found, use batch counts as proxy
        if metrics["training_samples"] == 0:
            train_batches_match = re.search(r"train_batches=(\d+)", content)
            if train_batches_match:
                metrics["training_samples"] = int(train_batches_match.group(1))

        if metrics["test_samples"] == 0:
            test_batches_match = re.search(r"test_batches=(\d+)", content)
            if test_batches_match:
                metrics["test_samples"] = int(test_batches_match.group(1))

        # Calculate throughput
        if metrics["total_duration_seconds"] > 0 and metrics["training_samples"] > 0:
            metrics["throughput_samples_per_second"] = (
                metrics["training_samples"] / metrics["total_duration_seconds"]
            )

    except Exception as e:
        print(f"Error reading {log_file}: {e}")

    return metrics


def create_computational_efficiency_table(results_dir):
    """Create a comprehensive table of computational efficiency metrics for all models and modes."""

    models = ["sttran", "dsg-detr", "tempura"]
    modes = ["predcls", "sgcls", "sgdet"]

    table_data = []

    for model in models:
        for mode in modes:
            # Look for log file
            log_file = os.path.join(results_dir, "logs", f"{model}_{mode}_run1.log")

            if os.path.exists(log_file):
                metrics = extract_computational_metrics_from_log(log_file)

                table_data.append(
                    {
                        "Model": model.upper(),
                        "Mode": mode,
                        "Duration (min)": f"{metrics['total_duration_minutes']:.2f}",
                        "Epochs": metrics["total_epochs"],
                        "Avg Epoch Time (min)": f"{metrics['avg_epoch_time_minutes']:.2f}",
                        "Memory Usage (MB)": f"{metrics['memory_usage_mb']:.1f}",
                        "Disk Usage (MB)": f"{metrics['disk_usage_mb']:.1f}",
                        "GPU Memory Used (MB)": f"{metrics['gpu_memory_used_mb']:.0f}",
                        "GPU Memory Total (MB)": f"{metrics['gpu_memory_total_mb']:.0f}",
                        "GPU Utilization (%)": f"{metrics['gpu_utilization_percent']:.1f}",
                        "Training Samples": metrics["training_samples"],
                        "Test Samples": metrics["test_samples"],
                        "Throughput (samples/s)": f"{metrics['throughput_samples_per_second']:.2f}",
                        "Final R@20": f"{metrics['final_r20']:.4f}",
                        "Final mR@20": f"{metrics['final_mr20']:.4f}",
                        "Avg Train Loss": f"{metrics['avg_train_loss']:.4f}",
                        "Final Val Loss": f"{metrics['final_val_loss']:.4f}",
                    }
                )
                print(f"Extracted computational metrics for {model} {mode}")
            else:
                print(f"Log file not found: {log_file}")
                # Add empty row for missing data
                table_data.append(
                    {
                        "Model": model.upper(),
                        "Mode": mode,
                        "Duration (min)": "N/A",
                        "Epochs": 0,
                        "Avg Epoch Time (min)": "N/A",
                        "Memory Usage (MB)": "N/A",
                        "Disk Usage (MB)": "N/A",
                        "GPU Memory Used (MB)": "N/A",
                        "GPU Memory Total (MB)": "N/A",
                        "GPU Utilization (%)": "N/A",
                        "Training Samples": 0,
                        "Test Samples": 0,
                        "Throughput (samples/s)": "N/A",
                        "Final R@20": "N/A",
                        "Final mR@20": "N/A",
                        "Avg Train Loss": "N/A",
                        "Final Val Loss": "N/A",
                    }
                )

    # Create DataFrame
    df = pd.DataFrame(table_data)

    return df


def create_efficiency_summary_table(df):
    """Create a summary table focusing on key efficiency metrics."""

    # Filter out rows with N/A values for calculations
    numeric_df = df.replace("N/A", np.nan)

    summary_data = []

    for model in df["Model"].unique():
        model_data = numeric_df[numeric_df["Model"] == model]

        # Calculate averages across all modes
        avg_duration = pd.to_numeric(
            model_data["Duration (min)"], errors="coerce"
        ).mean()
        avg_epoch_time = pd.to_numeric(
            model_data["Avg Epoch Time (min)"], errors="coerce"
        ).mean()
        avg_memory = pd.to_numeric(
            model_data["Memory Usage (MB)"], errors="coerce"
        ).mean()
        avg_gpu_util = pd.to_numeric(
            model_data["GPU Utilization (%)"], errors="coerce"
        ).mean()
        avg_throughput = pd.to_numeric(
            model_data["Throughput (samples/s)"], errors="coerce"
        ).mean()
        avg_r20 = pd.to_numeric(model_data["Final R@20"], errors="coerce").mean()
        avg_mr20 = pd.to_numeric(model_data["Final mR@20"], errors="coerce").mean()

        summary_data.append(
            {
                "Model": model,
                "Avg Duration (min)": f"{avg_duration:.2f}"
                if not pd.isna(avg_duration)
                else "N/A",
                "Avg Epoch Time (min)": f"{avg_epoch_time:.2f}"
                if not pd.isna(avg_epoch_time)
                else "N/A",
                "Avg Memory (MB)": f"{avg_memory:.1f}"
                if not pd.isna(avg_memory)
                else "N/A",
                "Avg GPU Util (%)": f"{avg_gpu_util:.1f}"
                if not pd.isna(avg_gpu_util)
                else "N/A",
                "Avg Throughput (samples/s)": f"{avg_throughput:.2f}"
                if not pd.isna(avg_throughput)
                else "N/A",
                "Avg R@20": f"{avg_r20:.4f}" if not pd.isna(avg_r20) else "N/A",
                "Avg mR@20": f"{avg_mr20:.4f}" if not pd.isna(avg_mr20) else "N/A",
            }
        )

    return pd.DataFrame(summary_data)


def main():
    results_dir = "data/benchmark/ag200"

    print("Extracting computational efficiency metrics from all benchmark logs...")
    df = create_computational_efficiency_table(results_dir)

    # Create summary table
    summary_df = create_efficiency_summary_table(df)

    # Save to CSV
    csv_file = os.path.join(results_dir, "computational_efficiency_table.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nFull table saved to: {csv_file}")

    summary_csv_file = os.path.join(results_dir, "computational_efficiency_summary.csv")
    summary_df.to_csv(summary_csv_file, index=False)
    print(f"Summary table saved to: {summary_csv_file}")

    # Print formatted tables
    print("\n" + "=" * 150)
    print("COMPREHENSIVE COMPUTATIONAL EFFICIENCY TABLE")
    print("=" * 150)
    print(df.to_string(index=False))

    print("\n" + "=" * 100)
    print("COMPUTATIONAL EFFICIENCY SUMMARY (Average across all modes)")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    # Print key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)

    # Find fastest and slowest models
    numeric_df = df.replace("N/A", np.nan)
    durations = pd.to_numeric(numeric_df["Duration (min)"], errors="coerce")
    valid_durations = durations.dropna()

    if not valid_durations.empty:
        fastest_idx = valid_durations.idxmin()
        slowest_idx = valid_durations.idxmax()

        print(
            f"Fastest training: {df.iloc[fastest_idx]['Model']} {df.iloc[fastest_idx]['Mode']} ({df.iloc[fastest_idx]['Duration (min)']} min)"
        )
        print(
            f"Slowest training: {df.iloc[slowest_idx]['Model']} {df.iloc[slowest_idx]['Mode']} ({df.iloc[slowest_idx]['Duration (min)']} min)"
        )

    # Find most and least efficient models
    throughputs = pd.to_numeric(numeric_df["Throughput (samples/s)"], errors="coerce")
    valid_throughputs = throughputs.dropna()

    if not valid_throughputs.empty:
        most_efficient_idx = valid_throughputs.idxmax()
        least_efficient_idx = valid_throughputs.idxmin()

        print(
            f"Most efficient: {df.iloc[most_efficient_idx]['Model']} {df.iloc[most_efficient_idx]['Mode']} ({df.iloc[most_efficient_idx]['Throughput (samples/s)']} samples/s)"
        )
        print(
            f"Least efficient: {df.iloc[least_efficient_idx]['Model']} {df.iloc[least_efficient_idx]['Mode']} ({df.iloc[least_efficient_idx]['Throughput (samples/s)']} samples/s)"
        )


if __name__ == "__main__":
    main()
