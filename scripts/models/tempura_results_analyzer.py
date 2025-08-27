#!/usr/bin/env python3
"""
TEMPURA Results Analyzer
Analyzes and compares results from different TEMPURA hyperparameter combinations.
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class TEMPURAResultsAnalyzer:
    """Analyzer for TEMPURA hyperparameter search results"""

    def __init__(self, results_dir: str = "output/tempura_hp_search"):
        self.results_dir = results_dir
        self.results_data = []

    def load_results(self) -> List[Dict[str, Any]]:
        """Load results from all training runs"""
        results = []

        # Find all run directories
        run_dirs = glob.glob(os.path.join(self.results_dir, "run_*"))
        run_dirs.sort(key=lambda x: int(x.split("_")[-1]))

        for run_dir in run_dirs:
            run_number = int(os.path.basename(run_dir).split("_")[-1])

            # Load hyperparameters
            hp_file = os.path.join(run_dir, "hyperparameters.json")
            if os.path.exists(hp_file):
                with open(hp_file, "r") as f:
                    hyperparameters = json.load(f)
            else:
                hyperparameters = {}

            # Load training log
            log_file = os.path.join(run_dir, "logfile.txt")
            if os.path.exists(log_file):
                metrics = self._extract_metrics_from_log(log_file)
            else:
                metrics = {}

            # Load predictions if available
            predictions_file = os.path.join(run_dir, "predictions.csv")
            if os.path.exists(predictions_file):
                predictions_df = pd.read_csv(predictions_file)
                # Extract best metrics from predictions
                if not predictions_df.empty:
                    best_metrics = {
                        "best_r20": predictions_df.get("r20", [0]).iloc[0],
                        "best_mrecall": predictions_df.get("mrecall", [0]).iloc[0],
                        "best_epoch": predictions_df.get("best_epoch", [-1]).iloc[0],
                    }
                    metrics.update(best_metrics)

            result = {
                "run_number": run_number,
                "run_dir": run_dir,
                "hyperparameters": hyperparameters,
                "metrics": metrics,
            }
            results.append(result)

        self.results_data = results
        return results

    def _extract_metrics_from_log(self, log_file: str) -> Dict[str, Any]:
        """Extract metrics from training log file"""
        metrics = {}

        try:
            with open(log_file, "r") as f:
                lines = f.readlines()

            # Look for best model information
            for line in lines:
                if "NEW BEST!" in line:
                    # Extract epoch number
                    if "epoch" in line:
                        try:
                            epoch = int(line.split("epoch")[1].split()[0])
                            metrics["best_epoch"] = epoch
                        except:
                            pass

                # Look for final best score
                if "Best model saved at epoch" in line:
                    try:
                        parts = line.split()
                        epoch_idx = parts.index("epoch") + 1
                        score_idx = parts.index("score:") + 1
                        metrics["best_epoch"] = int(parts[epoch_idx])
                        metrics["best_score"] = float(parts[score_idx])
                    except:
                        pass

                # Look for recall metrics
                if "R@20:" in line:
                    try:
                        r20 = float(line.split("R@20:")[1].split()[0])
                        metrics["r20"] = r20
                    except:
                        pass

                if "R@50:" in line:
                    try:
                        r50 = float(line.split("R@50:")[1].split()[0])
                        metrics["r50"] = r50
                    except:
                        pass

        except Exception as e:
            print(f"Warning: Could not parse log file {log_file}: {e}")

        return metrics

    def create_results_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with all results for analysis"""
        if not self.results_data:
            self.load_results()

        rows = []
        for result in self.results_data:
            row = {"run_number": result["run_number"], "run_dir": result["run_dir"]}

            # Add hyperparameters
            for key, value in result["hyperparameters"].items():
                row[f"hp_{key}"] = value

            # Add metrics
            for key, value in result["metrics"].items():
                row[f"metric_{key}"] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def find_best_combinations(
        self, metric: str = "best_score", top_k: int = 5
    ) -> pd.DataFrame:
        """Find the best hyperparameter combinations based on a metric"""
        df = self.create_results_dataframe()

        if f"metric_{metric}" not in df.columns:
            print(f"Warning: Metric '{metric}' not found in results")
            return df.head(top_k)

        # Sort by metric (descending for scores, ascending for losses)
        if "loss" in metric.lower():
            df_sorted = df.sort_values(f"metric_{metric}")
        else:
            df_sorted = df.sort_values(f"metric_{metric}", ascending=False)

        return df_sorted.head(top_k)

    def analyze_hyperparameter_importance(
        self, metric: str = "best_score"
    ) -> Dict[str, float]:
        """Analyze the importance of different hyperparameters"""
        df = self.create_results_dataframe()

        if f"metric_{metric}" not in df.columns:
            print(f"Warning: Metric '{metric}' not found in results")
            return {}

        # Get hyperparameter columns
        hp_cols = [col for col in df.columns if col.startswith("hp_")]

        importance = {}
        for hp_col in hp_cols:
            hp_name = hp_col[3:]  # Remove "hp_" prefix

            # Calculate correlation with metric
            correlation = df[hp_col].corr(df[f"metric_{metric}"])
            importance[hp_name] = abs(correlation) if not pd.isna(correlation) else 0.0

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance

    def plot_results(self, output_dir: str = "output/tempura_analysis"):
        """Create visualization plots for the results"""
        os.makedirs(output_dir, exist_ok=True)

        df = self.create_results_dataframe()
        if df.empty:
            print("No results data available for plotting")
            return

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Score distribution
        if "metric_best_score" in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df["metric_best_score"], bins=20, alpha=0.7, edgecolor="black")
            plt.xlabel("Best Score")
            plt.ylabel("Frequency")
            plt.title("Distribution of Best Scores Across Runs")
            plt.grid(True, alpha=0.3)
            plt.savefig(
                os.path.join(output_dir, "score_distribution.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # 2. Hyperparameter importance
        importance = self.analyze_hyperparameter_importance()
        if importance:
            plt.figure(figsize=(12, 8))
            hp_names = list(importance.keys())[:15]  # Top 15
            hp_values = [importance[name] for name in hp_names]

            plt.barh(hp_names, hp_values)
            plt.xlabel("Importance (Absolute Correlation)")
            plt.title("Hyperparameter Importance")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "hyperparameter_importance.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # 3. Learning rate vs score
        if "hp_lr" in df.columns and "metric_best_score" in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(df["hp_lr"], df["metric_best_score"], alpha=0.6)
            plt.xlabel("Learning Rate")
            plt.ylabel("Best Score")
            plt.title("Learning Rate vs Best Score")
            plt.xscale("log")
            plt.grid(True, alpha=0.3)
            plt.savefig(
                os.path.join(output_dir, "lr_vs_score.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # 4. Architecture parameters vs score
        arch_params = ["hp_enc_layer", "hp_dec_layer", "hp_K"]
        if (
            all(param in df.columns for param in arch_params)
            and "metric_best_score" in df.columns
        ):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for i, param in enumerate(arch_params):
                axes[i].scatter(df[param], df["metric_best_score"], alpha=0.6)
                axes[i].set_xlabel(param[3:])  # Remove "hp_" prefix
                axes[i].set_ylabel("Best Score")
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "architecture_vs_score.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def generate_report(
        self, output_file: str = "output/tempura_analysis/results_report.txt"
    ):
        """Generate a comprehensive text report"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df = self.create_results_dataframe()

        with open(output_file, "w") as f:
            f.write("TEMPURA Hyperparameter Search Results Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total runs analyzed: {len(df)}\n\n")

            # Best combinations
            f.write("TOP 5 BEST COMBINATIONS:\n")
            f.write("-" * 30 + "\n")
            best_combinations = self.find_best_combinations()
            for i, (_, row) in enumerate(best_combinations.iterrows()):
                f.write(f"\n{i+1}. Run #{row['run_number']}\n")
                f.write(f"   Best Score: {row.get('metric_best_score', 'N/A')}\n")
                f.write(f"   Best Epoch: {row.get('metric_best_epoch', 'N/A')}\n")

                # Show key hyperparameters
                hp_cols = [col for col in row.index if col.startswith("hp_")]
                for hp_col in hp_cols[:10]:  # Show first 10 hyperparameters
                    hp_name = hp_col[3:]
                    hp_value = row[hp_col]
                    f.write(f"   {hp_name}: {hp_value}\n")

            # Hyperparameter importance
            f.write("\n\nHYPERPARAMETER IMPORTANCE:\n")
            f.write("-" * 30 + "\n")
            importance = self.analyze_hyperparameter_importance()
            for hp_name, importance_score in list(importance.items())[:10]:
                f.write(f"{hp_name}: {importance_score:.4f}\n")

            # Summary statistics
            if "metric_best_score" in df.columns:
                f.write(f"\n\nSUMMARY STATISTICS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Mean Best Score: {df['metric_best_score'].mean():.4f}\n")
                f.write(f"Std Best Score: {df['metric_best_score'].std():.4f}\n")
                f.write(f"Min Best Score: {df['metric_best_score'].min():.4f}\n")
                f.write(f"Max Best Score: {df['metric_best_score'].max():.4f}\n")

        print(f"Report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TEMPURA hyperparameter search results"
    )
    parser.add_argument(
        "--results_dir",
        default="output/tempura_hp_search",
        help="Directory containing training results",
    )
    parser.add_argument(
        "--output_dir",
        default="output/tempura_analysis",
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--metric",
        default="best_score",
        help="Metric to use for finding best combinations",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = TEMPURAResultsAnalyzer(args.results_dir)

    # Load and analyze results
    results = analyzer.load_results()
    print(f"Loaded {len(results)} training runs")

    if not results:
        print("No results found. Make sure training runs have completed.")
        return

    # Find best combinations
    best_combinations = analyzer.find_best_combinations(args.metric)
    print(f"\nTop 5 best combinations based on {args.metric}:")
    print(
        best_combinations[["run_number", f"metric_{args.metric}"]].to_string(
            index=False
        )
    )

    # Generate visualizations
    analyzer.plot_results(args.output_dir)
    print(f"Plots saved to {args.output_dir}")

    # Generate report
    report_file = os.path.join(args.output_dir, "results_report.txt")
    analyzer.generate_report(report_file)


if __name__ == "__main__":
    main()
