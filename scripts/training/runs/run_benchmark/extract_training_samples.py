#!/usr/bin/env python3
"""
Extract exact number of batches and samples used for training each model.
"""

import os
import re
import pandas as pd


def extract_training_samples_from_log(log_file):
    """Extract training samples and batches from a single training log file."""
    metrics = {
        "training_samples": 0,
        "test_samples": 0,
        "train_batches": 0,
        "test_batches": 0,
        "fraction_used": 0,
        "total_epochs": 0,
    }

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract sample counts - look for the actual pattern in logs
        # Pattern: "Using 31/158" followed by "training samples and 6/33 test samples"
        train_samples_match = re.search(
            r"Using (\d+)/\d+\s+training samples", content, re.DOTALL
        )
        if train_samples_match:
            metrics["training_samples"] = int(train_samples_match.group(1))

        test_samples_match = re.search(r"(\d+)/\d+ test samples", content)
        if test_samples_match:
            metrics["test_samples"] = int(test_samples_match.group(1))

        # Extract batch counts
        train_batches_match = re.search(r"train_batches=(\d+)", content)
        if train_batches_match:
            metrics["train_batches"] = int(train_batches_match.group(1))

        test_batches_match = re.search(r"test_batches=(\d+)", content)
        if test_batches_match:
            metrics["test_batches"] = int(test_batches_match.group(1))

        # Extract fraction used from the training command
        fraction_match = re.search(r"-fraction (\d+)", content)
        if fraction_match:
            metrics["fraction_used"] = int(fraction_match.group(1))

        # Extract total epochs
        epoch_matches = re.findall(r"Epoch (\d+) \|", content)
        if epoch_matches:
            metrics["total_epochs"] = max([int(epoch) for epoch in epoch_matches]) + 1

        # Calculate total samples processed during training
        if metrics["training_samples"] > 0 and metrics["total_epochs"] > 0:
            metrics["total_samples_processed"] = (
                metrics["training_samples"] * metrics["total_epochs"]
            )
        else:
            metrics["total_samples_processed"] = 0

    except Exception as e:
        print(f"Error reading {log_file}: {e}")

    return metrics


def create_training_samples_table(results_dir):
    """Create a comprehensive table of training samples and batches for all models and modes."""

    models = ["sttran", "dsg-detr", "tempura"]
    modes = ["predcls", "sgcls", "sgdet"]

    table_data = []

    for model in models:
        for mode in modes:
            # Look for log file
            log_file = os.path.join(results_dir, "logs", f"{model}_{mode}_run1.log")

            if os.path.exists(log_file):
                metrics = extract_training_samples_from_log(log_file)

                table_data.append(
                    {
                        "Model": model.upper(),
                        "Mode": mode,
                        "Training Samples": metrics["training_samples"],
                        "Test Samples": metrics["test_samples"],
                        "Train Batches": metrics["train_batches"],
                        "Test Batches": metrics["test_batches"],
                        "Fraction Used": metrics["fraction_used"],
                        "Total Epochs": metrics["total_epochs"],
                        "Total Samples Processed": metrics["total_samples_processed"],
                    }
                )
                print(f"Extracted training samples for {model} {mode}")
            else:
                print(f"Log file not found: {log_file}")
                # Add empty row for missing data
                table_data.append(
                    {
                        "Model": model.upper(),
                        "Mode": mode,
                        "Training Samples": 0,
                        "Test Samples": 0,
                        "Train Batches": 0,
                        "Test Batches": 0,
                        "Fraction Used": 0,
                        "Total Epochs": 0,
                        "Total Samples Processed": 0,
                    }
                )

    # Create DataFrame
    df = pd.DataFrame(table_data)

    return df


def create_summary_table(df):
    """Create a summary table with totals and averages."""

    summary_data = []

    # Overall summary
    total_training_samples = df["Training Samples"].sum()
    total_test_samples = df["Test Samples"].sum()
    total_train_batches = df["Train Batches"].sum()
    total_test_batches = df["Test Batches"].sum()
    total_samples_processed = df["Total Samples Processed"].sum()

    summary_data.append(
        {
            "Metric": "Total Training Samples (All Models)",
            "Value": total_training_samples,
        }
    )
    summary_data.append(
        {"Metric": "Total Test Samples (All Models)", "Value": total_test_samples}
    )
    summary_data.append(
        {"Metric": "Total Train Batches (All Models)", "Value": total_train_batches}
    )
    summary_data.append(
        {"Metric": "Total Test Batches (All Models)", "Value": total_test_batches}
    )
    summary_data.append(
        {
            "Metric": "Total Samples Processed (All Models)",
            "Value": total_samples_processed,
        }
    )

    # Per model summary
    for model in df["Model"].unique():
        model_data = df[df["Model"] == model]
        model_training_samples = model_data["Training Samples"].sum()
        model_test_samples = model_data["Test Samples"].sum()
        model_samples_processed = model_data["Total Samples Processed"].sum()

        summary_data.append(
            {"Metric": f"{model} - Training Samples", "Value": model_training_samples}
        )
        summary_data.append(
            {"Metric": f"{model} - Test Samples", "Value": model_test_samples}
        )
        summary_data.append(
            {
                "Metric": f"{model} - Total Samples Processed",
                "Value": model_samples_processed,
            }
        )

    return pd.DataFrame(summary_data)


def main():
    results_dir = "data/benchmark/ag200"

    print("Extracting training samples and batches from all benchmark logs...")
    df = create_training_samples_table(results_dir)

    # Create summary table
    summary_df = create_summary_table(df)

    # Save to CSV
    csv_file = os.path.join(results_dir, "training_samples_table.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nTable saved to: {csv_file}")

    summary_csv_file = os.path.join(results_dir, "training_samples_summary.csv")
    summary_df.to_csv(summary_csv_file, index=False)
    print(f"Summary saved to: {summary_csv_file}")

    # Print formatted tables
    print("\n" + "=" * 120)
    print("TRAINING SAMPLES AND BATCHES TABLE")
    print("=" * 120)
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # Print key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)

    # Check if all models used the same dataset size
    unique_training_samples = df["Training Samples"].unique()
    unique_test_samples = df["Test Samples"].unique()
    unique_fractions = df["Fraction Used"].unique()

    print(f"Training samples per model-mode: {unique_training_samples}")
    print(f"Test samples per model-mode: {unique_test_samples}")
    print(f"Fraction used: {unique_fractions}")

    if len(unique_training_samples) == 1 and unique_training_samples[0] > 0:
        print(
            f"✓ All models used the same dataset size: {unique_training_samples[0]} training samples"
        )
    else:
        print("⚠ Different models used different dataset sizes")

    if len(unique_fractions) == 1 and unique_fractions[0] > 0:
        print(
            f"✓ All models used the same fraction: {unique_fractions[0]} (1=all, 2=half, 5=20%, 10=10%)"
        )
    else:
        print("⚠ Different models used different fractions")


if __name__ == "__main__":
    main()
