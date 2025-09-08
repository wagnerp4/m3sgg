#!/usr/bin/env python3
"""
Plotting script for AG200 benchmark results.
Creates plots of R@20/50/100 and mR@20/50/100 vs epoch for three models
across modes, plus train vs eval loss and a compute-performance trade-off plot.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional

# TODO: Consider moving common utilities to a shared plotting utils module


def extract_metrics_from_log(log_file: str) -> Dict[str, List[float]]:
    """Extract metrics and compute stats from a training log file.

    :param log_file: Path to the log file
    :type log_file: str
    :return: Dictionary containing lists of metrics per epoch and compute stats
    :rtype: Dict[str, List[float]]
    """
    metrics: Dict[str, List[float] | float | str] = {
        "epoch": [],
        "r20": [],
        "r50": [],
        "r100": [],
        "mr20": [],
        "mr50": [],
        "mr100": [],
        "train_loss": [],
        "val_loss": [],
        "duration_sec": 0.0,
        "memory_delta_mb": 0.0,
        "final_r20": 0.0,
        "final_mr20": 0.0,
    }

    if not os.path.exists(log_file):
        print(f"Warning: Log file not found: {log_file}")
        return metrics  # type: ignore[return-value]

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Epoch indices from "Starting Epoch X" lines
    epoch_matches = re.findall(r"Starting Epoch (\d+)", content)
    epochs = [int(e) for e in epoch_matches]
    # Deduplicate while preserving order
    seen = set()
    epoch_list: List[int] = []
    for e in epochs:
        if e not in seen:
            seen.add(e)
            epoch_list.append(e)
    metrics["epoch"] = epoch_list

    # R@K and mR@K from evaluator stats printed per epoch
    # These are often printed together; we collect sequences per epoch occurrence order
    def find_all_floats(pattern: str) -> List[float]:
        return [float(v) for v in re.findall(pattern, content)]

    r20 = find_all_floats(r"R@20:\s*([\d.]+)")
    r50 = find_all_floats(r"R@50:\s*([\d.]+)")
    r100 = find_all_floats(r"R@100:\s*([\d.]+)")
    mr20 = find_all_floats(r"mR@20:\s*([\d.]+)")
    mr50 = find_all_floats(r"mR@50:\s*([\d.]+)")
    mr100 = find_all_floats(r"mR@100:\s*([\d.]+)")

    metrics["r20"] = r20
    metrics["r50"] = r50
    metrics["r100"] = r100
    metrics["mr20"] = mr20
    metrics["mr50"] = mr50
    metrics["mr100"] = mr100

    # Loss values: one avg_train_loss per epoch and one avg_val_loss per epoch
    metrics["train_loss"] = [float(v) for v in re.findall(r"avg_train_loss=([\d.]+)", content)]
    metrics["val_loss"] = [float(v) for v in re.findall(r"avg_val_loss=([\d.]+)", content)]

    # Duration and memory delta for compute plot
    duration_match = re.search(r"Duration:\s*([\d.]+) seconds", content)
    if duration_match:
        metrics["duration_sec"] = float(duration_match.group(1))
    mem_delta_match = re.search(r"Memory delta:\s*([\d.]+)MB", content)
    if mem_delta_match:
        metrics["memory_delta_mb"] = float(mem_delta_match.group(1))

    # Final performance snapshot for compute plot
    if r20:
        metrics["final_r20"] = r20[-1]
    if mr20:
        metrics["final_mr20"] = mr20[-1]

    return metrics  # type: ignore[return-value]


def load_results_by_mode(base_dir: str) -> Dict[str, Dict[str, Dict[str, Dict]]]:
    """Load results from all modes, models, and runs.

    Structure:
        results[mode][model][run_name] -> metrics dict

    :param base_dir: Base directory containing results (e.g., data/benchmark/ag200)
    :type base_dir: str
    :return: Nested dictionary with mode -> model -> run -> metrics
    :rtype: Dict[str, Dict[str, Dict[str, Dict]]]
    """
    results: Dict[str, Dict[str, Dict[str, Dict]]] = {}
    modes = [p.name for p in Path(base_dir).iterdir() if p.is_dir() and p.name in {"predcls", "sgcls", "sgdet"}]
    models = ["sttran", "dsg-detr", "tempura"]

    for mode in modes:
        results[mode] = {}
        for model in models:
            results[mode][model] = {}
            for run in range(1, 4):
                run_dir = Path(base_dir) / mode / f"{model}_run{run}"
                log_file = run_dir / "logfile.txt"
                if log_file.exists():
                    metrics = extract_metrics_from_log(str(log_file))
                    results[mode][model][f"run{run}"] = metrics
                else:
                    # Fallback: some runs may only have metrics_summary.txt and external logs
                    # TODO: Optionally parse LogDir logs if needed
                    pass
    return results


def ensure_output_dir(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def plot_combined_metrics(results: Dict[str, Dict[str, Dict[str, Dict]]], output_dir: str) -> None:
    """Create combined plots of R@20/50/100 and mR@20/50/100 vs epoch across modes and models.

    Saves a high-DPI PDF file.
    """
    ensure_output_dir(output_dir)

    modes = ["predcls", "sgcls", "sgdet"]
    metrics_to_plot = [
        ("r20", "R@20"), ("r50", "R@50"), ("r100", "R@100"),
        ("mr20", "mR@20"), ("mr50", "mR@50"), ("mr100", "mR@100"),
    ]

    fig, axes = plt.subplots(len(modes), len(metrics_to_plot), figsize=(24, 12), sharex=True)
    fig.suptitle("AG200: Recall and Mean Recall vs Epoch across Modes and Models", fontsize=14)

    for row, mode in enumerate(modes):
        for col, (key, title) in enumerate(metrics_to_plot):
            ax = axes[row, col]
            ax.set_title(f"{mode} - {title}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)

            if mode not in results:
                continue
            for model, runs in results[mode].items():
                for run_name, run_data in runs.items():
                    y = run_data.get(key, [])
                    x = list(range(len(y))) if len(y) > 0 else []
                    if len(x) > 0:
                        ax.plot(x, y, label=f"{model} {run_name}", linewidth=1, alpha=0.8)
            if row == 0 and col == len(metrics_to_plot) - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, "ag200_combined_metrics.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined metrics PDF: {pdf_path}")


def plot_sttran_sgdet_losses(results: Dict[str, Dict[str, Dict[str, Dict]]], output_dir: str) -> None:
    """Plot train vs eval loss for STTran on sgdet over epochs.

    Saves a high-DPI PDF file.
    """
    ensure_output_dir(output_dir)

    mode = "sgdet"
    model = "sttran"
    if mode not in results or model not in results[mode]:
        print("Warning: No results for STTran sgdet")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("STTran (sgdet): Training vs Validation Loss")

    for run_name, run_data in results[mode][model].items():
        tl = run_data.get("train_loss", [])
        vl = run_data.get("val_loss", [])
        x_t = list(range(len(tl)))
        x_v = list(range(len(vl)))
        if len(x_t) > 0:
            ax.plot(x_t, tl, label=f"{run_name} train", alpha=0.8)
        if len(x_v) > 0:
            ax.plot(x_v, vl, label=f"{run_name} val", alpha=0.8, linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    pdf_path = os.path.join(output_dir, "sttran_sgdet_train_vs_val_loss.pdf")
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved STTran sgdet loss PDF: {pdf_path}")


def plot_compute_performance_tradeoff(results: Dict[str, Dict[str, Dict[str, Dict]]], output_dir: str) -> None:
    """Plot runtime, memory, and performance trade-offs across models and modes.

    We visualize duration (x), memory delta (y), and color/marker by final R@20.
    Saves a high-DPI PDF file.
    """
    ensure_output_dir(output_dir)

    points = []
    for mode, models in results.items():
        for model, runs in models.items():
            for run_name, run_data in runs.items():
                points.append({
                    "mode": mode,
                    "model": model,
                    "run": run_name,
                    "duration_sec": float(run_data.get("duration_sec", 0.0)),
                    "memory_delta_mb": float(run_data.get("memory_delta_mb", 0.0)),
                    "final_r20": float(run_data.get("final_r20", 0.0)),
                    "final_mr20": float(run_data.get("final_mr20", 0.0)),
                })
    if not points:
        print("Warning: No compute stats found in logs. Skipping trade-off plot.")
        return

    df = pd.DataFrame(points)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Compute vs Performance Trade-off (AG200)")

    # Scatter by mode with color mapping to final R@20
    modes = sorted(df["mode"].unique())
    markers = {"predcls": "o", "sgcls": "s", "sgdet": "^"}
    for mode in modes:
        dmode = df[df["mode"] == mode]
        sc = ax.scatter(
            dmode["duration_sec"], dmode["memory_delta_mb"],
            c=dmode["final_r20"], cmap="viridis", marker=markers.get(mode, "o"),
            alpha=0.8, label=mode, edgecolors="k", linewidths=0.3
        )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Final R@20")

    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Memory Delta (MB)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Mode")

    pdf_path = os.path.join(output_dir, "compute_performance_tradeoff.pdf")
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved compute vs performance PDF: {pdf_path}")


def main() -> None:
    """Main function to generate all plots and summaries."""
    parser = argparse.ArgumentParser(description="Plot AG200 benchmark results")
    parser.add_argument(
        "--results_dir",
        default="data/benchmark/ag200",
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--output_dir",
        default="data/benchmark/ag200/plots",
        help="Directory to save plots"
    )

    args = parser.parse_args()

    print("Loading AG200 benchmark results...")
    results = load_results_by_mode(args.results_dir)

    print("Creating plots and summaries...")
    plot_combined_metrics(results, args.output_dir)
    plot_sttran_sgdet_losses(results, args.output_dir)
    plot_compute_performance_tradeoff(results, args.output_dir)

    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
