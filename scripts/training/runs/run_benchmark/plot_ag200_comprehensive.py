#!/usr/bin/env python3
"""
Comprehensive plotting script for AG200 benchmark results.
Extracts metrics from all training logs and creates high-resolution PDF plots.

Plots generated:
1. All 3 models, all 3 modes, R@20
2. Only Tempura, all 3 modes, all R@K's
3. All 3 models, all 3 modes, train loss vs val loss
4. All 3 models, all 3 modes, mR@20
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def extract_metrics_from_log(log_file):
    """Extract metrics from a single training log file."""
    metrics = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "R@10": [],
        "R@20": [],
        "R@50": [],
        "R@100": [],
        "MR@10": [],
        "MR@20": [],
        "MR@50": [],
        "MR@100": [],
    }

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract metrics line by line
        lines = content.split("\n")
        current_epoch = None
        epoch_metrics = {}

        for line in lines:
            # Extract train loss
            train_loss_match = re.search(
                r"Epoch (\d+) \| avg_train_loss=([\d.]+)", line
            )
            if train_loss_match:
                current_epoch = int(train_loss_match.group(1))
                if current_epoch not in epoch_metrics:
                    epoch_metrics[current_epoch] = {}
                epoch_metrics[current_epoch]["train_loss"] = float(
                    train_loss_match.group(2)
                )

            # Extract val loss
            val_loss_match = re.search(r"Epoch (\d+) \| avg_val_loss=([\d.]+)", line)
            if val_loss_match:
                current_epoch = int(val_loss_match.group(1))
                if current_epoch not in epoch_metrics:
                    epoch_metrics[current_epoch] = {}
                epoch_metrics[current_epoch]["val_loss"] = float(
                    val_loss_match.group(2)
                )

            # Extract R@K and mR@K metrics
            rk_match = re.search(
                r"(R@10|R@20|R@50|R@100|MR@10|MR@20|MR@50|MR@100): ([\d.]+)", line
            )
            if rk_match and current_epoch is not None:
                metric_name = rk_match.group(1)
                metric_value = float(rk_match.group(2))
                if current_epoch not in epoch_metrics:
                    epoch_metrics[current_epoch] = {}
                epoch_metrics[current_epoch][metric_name] = metric_value

        # Sort epochs and fill metrics arrays
        sorted_epochs = sorted(epoch_metrics.keys())
        for epoch in sorted_epochs:
            metrics["epochs"].append(epoch)
            metrics["train_loss"].append(epoch_metrics[epoch].get("train_loss", 0))
            metrics["val_loss"].append(epoch_metrics[epoch].get("val_loss", 0))
            metrics["R@10"].append(epoch_metrics[epoch].get("R@10", 0))
            metrics["R@20"].append(epoch_metrics[epoch].get("R@20", 0))
            metrics["R@50"].append(epoch_metrics[epoch].get("R@50", 0))
            metrics["R@100"].append(epoch_metrics[epoch].get("R@100", 0))
            metrics["MR@10"].append(epoch_metrics[epoch].get("MR@10", 0))
            metrics["MR@20"].append(epoch_metrics[epoch].get("MR@20", 0))
            metrics["MR@50"].append(epoch_metrics[epoch].get("MR@50", 0))
            metrics["MR@100"].append(epoch_metrics[epoch].get("MR@100", 0))

    except Exception as e:
        print(f"Error reading {log_file}: {e}")

    return metrics


def load_all_results(results_dir):
    """Load results from all benchmark runs."""
    results = {}

    # Define models and modes
    models = ["sttran", "dsg-detr", "tempura"]
    modes = ["predcls", "sgcls", "sgdet"]

    for model in models:
        results[model] = {}
        for mode in modes:
            # Look for log files
            log_pattern = os.path.join(results_dir, "logs", f"{model}_{mode}_run1.log")
            if os.path.exists(log_pattern):
                metrics = extract_metrics_from_log(log_pattern)
                if metrics["epochs"]:  # Only add if we have data
                    results[model][mode] = metrics
                    print(f"Loaded {model} {mode}: {len(metrics['epochs'])} epochs")
                else:
                    print(f"No data found for {model} {mode}")
            else:
                print(f"Log file not found: {log_pattern}")

    return results


def create_plot1_all_models_r20(results, output_dir):
    """Plot 1: All 3 models, all 3 modes, R@20"""
    plt.figure(figsize=(12, 8))

    models = ["sttran", "dsg-detr", "tempura"]
    modes = ["predcls", "sgcls", "sgdet"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]

    for i, model in enumerate(models):
        for j, mode in enumerate(modes):
            if model in results and mode in results[model]:
                data = results[model][mode]
                if data["epochs"] and data["R@20"]:
                    plt.plot(
                        data["epochs"],
                        data["R@20"],
                        color=colors[i],
                        marker=markers[j],
                        label=f"{model.upper()} {mode}",
                        linewidth=2,
                        markersize=6,
                    )

    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("R@20", fontsize=12, fontweight="bold")
    plt.title(
        "R@20 Performance Across All Models and Modes", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    output_file = os.path.join(output_dir, "plot1_all_models_r20.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 1 saved: {output_file}")


def create_plot2_tempura_all_rk(results, output_dir):
    """Plot 2: Only Tempura, all 3 modes, all R@K's"""
    plt.figure(figsize=(12, 8))

    if "tempura" not in results:
        print("No Tempura data found for Plot 2")
        return

    modes = ["predcls", "sgcls", "sgdet"]
    rk_metrics = ["R@10", "R@20", "R@50", "R@100"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^", "D"]

    for i, mode in enumerate(modes):
        if mode in results["tempura"]:
            data = results["tempura"][mode]
            if data["epochs"]:
                for j, metric in enumerate(rk_metrics):
                    if data[metric]:
                        plt.plot(
                            data["epochs"],
                            data[metric],
                            color=colors[i],
                            marker=markers[j],
                            label=f"{mode} {metric}",
                            linewidth=2,
                            markersize=6,
                        )

    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Recall@K", fontsize=12, fontweight="bold")
    plt.title(
        "Tempura: All R@K Metrics Across All Modes", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    output_file = os.path.join(output_dir, "plot2_tempura_all_rk.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 2 saved: {output_file}")


def create_plot3_train_vs_val_loss(results, output_dir):
    """Plot 3: All 3 models, all 3 modes, train loss vs val loss"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        "Train Loss vs Validation Loss Across All Models and Modes",
        fontsize=16,
        fontweight="bold",
    )

    models = ["sttran", "dsg-detr", "tempura"]
    modes = ["predcls", "sgcls", "sgdet"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, model in enumerate(models):
        for j, mode in enumerate(modes):
            ax = axes[i, j]

            if model in results and mode in results[model]:
                data = results[model][mode]
                if data["epochs"] and data["train_loss"] and data["val_loss"]:
                    ax.plot(
                        data["epochs"],
                        data["train_loss"],
                        color=colors[i],
                        marker="o",
                        label="Train Loss",
                        linewidth=2,
                        markersize=4,
                    )
                    ax.plot(
                        data["epochs"],
                        data["val_loss"],
                        color=colors[i],
                        marker="s",
                        label="Val Loss",
                        linewidth=2,
                        markersize=4,
                        linestyle="--",
                    )

                    ax.set_title(f"{model.upper()} {mode}", fontweight="bold")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    ax.set_title(f"{model.upper()} {mode} (No Data)", fontweight="bold")
                    ax.text(
                        0.5,
                        0.5,
                        "No Data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
            else:
                ax.set_title(f"{model.upper()} {mode} (No Data)", fontweight="bold")
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

    plt.tight_layout()
    output_file = os.path.join(output_dir, "plot3_train_vs_val_loss.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 3 saved: {output_file}")


def create_plot4_all_models_mr20(results, output_dir):
    """Plot 4: All 3 models, all 3 modes, mR@20"""
    plt.figure(figsize=(12, 8))

    models = ["sttran", "dsg-detr", "tempura"]
    modes = ["predcls", "sgcls", "sgdet"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]

    for i, model in enumerate(models):
        for j, mode in enumerate(modes):
            if model in results and mode in results[model]:
                data = results[model][mode]
                if data["epochs"] and data["MR@20"]:
                    plt.plot(
                        data["epochs"],
                        data["MR@20"],
                        color=colors[i],
                        marker=markers[j],
                        label=f"{model.upper()} {mode}",
                        linewidth=2,
                        markersize=6,
                    )

    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("mR@20", fontsize=12, fontweight="bold")
    plt.title(
        "mR@20 Performance Across All Models and Modes", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    output_file = os.path.join(output_dir, "plot4_all_models_mr20.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 4 saved: {output_file}")


def create_summary_table(results, output_dir):
    """Create a summary table with final metrics."""
    summary_data = []

    models = ["sttran", "dsg-detr", "tempura"]
    modes = ["predcls", "sgcls", "sgdet"]

    for model in models:
        for mode in modes:
            if model in results and mode in results[model]:
                data = results[model][mode]
                if data["epochs"] and data["R@20"]:
                    summary_data.append(
                        {
                            "Model": model.upper(),
                            "Mode": mode,
                            "Epochs": max(data["epochs"]) if data["epochs"] else 0,
                            "R@10": data["R@10"][-1] if data["R@10"] else 0,
                            "R@20": data["R@20"][-1] if data["R@20"] else 0,
                            "R@50": data["R@50"][-1] if data["R@50"] else 0,
                            "R@100": data["R@100"][-1] if data["R@100"] else 0,
                            "MR@10": data["MR@10"][-1] if data["MR@10"] else 0,
                            "MR@20": data["MR@20"][-1] if data["MR@20"] else 0,
                            "MR@50": data["MR@50"][-1] if data["MR@50"] else 0,
                            "MR@100": data["MR@100"][-1] if data["MR@100"] else 0,
                            "Final Train Loss": data["train_loss"][-1]
                            if data["train_loss"]
                            else 0,
                            "Final Val Loss": data["val_loss"][-1]
                            if data["val_loss"]
                            else 0,
                        }
                    )

    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_file = os.path.join(output_dir, "benchmark_summary_table.csv")
        df.to_csv(csv_file, index=False)
        print(f"Summary table saved: {csv_file}")

        # Print summary
        print("\n=== Benchmark Summary ===")
        print(df.to_string(index=False))
    else:
        print("No data found for summary table")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive plots for AG200 benchmark"
    )
    parser.add_argument(
        "--results_dir",
        default="data/benchmark/ag200",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output_dir",
        default="data/benchmark/ag200/plots",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading benchmark results...")
    results = load_all_results(args.results_dir)

    print("\nGenerating plots...")

    # Create all plots
    create_plot1_all_models_r20(results, args.output_dir)
    create_plot2_tempura_all_rk(results, args.output_dir)
    create_plot3_train_vs_val_loss(results, args.output_dir)
    create_plot4_all_models_mr20(results, args.output_dir)

    # Create summary table
    create_summary_table(results, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")
    print("Generated plots:")
    print("1. plot1_all_models_r20.pdf - R@20 across all models and modes")
    print("2. plot2_tempura_all_rk.pdf - Tempura R@K metrics across all modes")
    print("3. plot3_train_vs_val_loss.pdf - Train vs validation loss comparison")
    print("4. plot4_all_models_mr20.pdf - mR@20 across all models and modes")
    print("5. benchmark_summary_table.csv - Summary table with final metrics")


if __name__ == "__main__":
    main()
