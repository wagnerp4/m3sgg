"""
ActionGenome ground truth scene graph summarization benchmark.

This module provides a benchmark for evaluating summarization models using
ActionGenome ground truth scene graphs with different linearization strategies.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch

# Import project modules
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from m3sgg.language.summarization.wrappers import (
    T5SummarizationWrapper,
    PegasusSummarizationWrapper,
)
from m3sgg.language.evaluation.action_genome_loader import ActionGenomeLoader
from m3sgg.language.evaluation.metrics import SummarizationMetrics

logger = logging.getLogger(__name__)


class ActionGenomeSummarizationBenchmark:
    """Benchmark for ActionGenome ground truth scene graph summarization.

    :param data_path: Path to ActionGenome dataset
    :type data_path: str
    :param device: Device to run inference on
    :type device: str
    :param split: Dataset split to use
    :type split: str
    """

    def __init__(self, data_path: str, device: str = "cuda:0", split: str = "val"):
        """Initialize ActionGenome summarization benchmark.

        :param data_path: Path to ActionGenome dataset
        :type data_path: str
        :param device: Device to run inference on
        :type device: str
        :param split: Dataset split to use
        :type split: str
        """
        self.data_path = data_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.split = split

        # Initialize components
        self.dataset_loader = None
        self.summarization_models = {}
        self.metrics = SummarizationMetrics()

        logger.info(f"Initialized ActionGenome benchmark with device: {self.device}")

    def load_models(self):
        """Load all required models for evaluation."""
        logger.info("Loading summarization models...")

        # Initialize summarization models
        self.summarization_models = {
            "t5_base": T5SummarizationWrapper("google-t5/t5-base", device=self.device),
            "t5_large": T5SummarizationWrapper(
                "google-t5/t5-large", device=self.device
            ),
            "pegasus_xsum": PegasusSummarizationWrapper(
                "google/pegasus-xsum", device=self.device
            ),
            "pegasus_cnn": PegasusSummarizationWrapper(
                "google/pegasus-cnn_dailymail", device=self.device
            ),
        }

        logger.info("All models loaded successfully")

    def load_dataset(self, max_videos: Optional[int] = None):
        """Load ActionGenome dataset.

        :param max_videos: Maximum number of videos to load
        :type max_videos: int, optional
        """
        logger.info(f"Loading ActionGenome {self.split} dataset...")
        self.dataset_loader = ActionGenomeLoader(
            data_path=self.data_path, split=self.split, max_videos=max_videos
        )
        logger.info(
            f"Dataset loaded with {len(self.dataset_loader.get_video_list())} videos"
        )

    def linearize_triples(self, triples: List[tuple], mode: str = "flat") -> List[str]:
        """Linearize scene graph triples into sentences.

        :param triples: List of (subject, predicate, object) triples
        :type triples: List[tuple]
        :param mode: Linearization mode (flat, majority, time)
        :type mode: str
        :return: List of sentences
        :rtype: List[str]
        """
        from m3sgg.language.summarization.summarize import linearize_triples

        return linearize_triples(triples, mode=mode)

    def generate_summary(self, text: str, model_name: str) -> str:
        """Generate summary using specified model.

        :param text: Input text to summarize
        :type text: str
        :param model_name: Name of summarization model
        :type model_name: str
        :return: Generated summary
        :rtype: str
        """
        if model_name not in self.summarization_models:
            raise ValueError(f"Model {model_name} not available")

        model = self.summarization_models[model_name]
        return model.summarize(text)

    def run_benchmark(
        self,
        subset_size: int = 100,
        models: Optional[List[str]] = None,
        linearizers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark on ActionGenome ground truth.

        :param subset_size: Number of test samples to use
        :type subset_size: int
        :param models: List of model names to evaluate
        :type models: List[str], optional
        :param linearizers: List of linearization modes to test
        :type linearizers: List[str], optional
        :return: Benchmark results
        :rtype: Dict[str, Any]
        """
        if models is None:
            models = list(self.summarization_models.keys())

        if linearizers is None:
            linearizers = ["flat", "majority", "time"]

        logger.info(f"Running ActionGenome benchmark with {subset_size} samples")
        logger.info(f"Models: {models}")
        logger.info(f"Linearizers: {linearizers}")

        # Create dataset subset
        subset = self.dataset_loader.create_subset(subset_size)

        # Check if we have any data
        if not subset["test"] or len(subset["test"]) == 0:
            logger.error(
                "No data available for benchmark. Check ActionGenome dataset loading."
            )
            return {"error": "No data available"}

        results = {}

        for linearizer in linearizers:
            logger.info(f"Evaluating linearizer: {linearizer}")
            results[linearizer] = {}

            for model_name in models:
                logger.info(f"Evaluating model: {model_name}")
                predictions = []
                references = []

                for i, sample in enumerate(subset["test"]):
                    # Get ground truth caption
                    reference = sample["caption"]
                    references.append(reference)

                    # Generate scene graph text using specified linearizer
                    triples = sample["triples"]
                    sentences = self.linearize_triples(triples, mode=linearizer)
                    scene_graph_text = (
                        " ".join(sentences)
                        if sentences
                        else "No objects or relationships detected."
                    )

                    # Generate summary
                    prediction = self.generate_summary(scene_graph_text, model_name)
                    predictions.append(prediction)

                    if i % 10 == 0:
                        logger.info(f"Processed {i+1}/{len(subset['test'])} samples")

                # Compute metrics
                metrics = self.metrics.compute_all_metrics(predictions, references)
                results[linearizer][model_name] = metrics

                logger.info(
                    f"Completed evaluation for {model_name} with {linearizer} linearizer"
                )

        return results

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save benchmark results to file.

        :param results: Benchmark results
        :type results: Dict[str, Any]
        :param output_path: Path to save results
        :type output_path: str
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        results_with_metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.device,
            "dataset": "ActionGenome",
            "split": self.split,
            "data_path": self.data_path,
            "results": results,
        }

        def _to_serializable(value: Any) -> Any:
            """Convert values to JSON-serializable types.

            :param value: Value to convert
            :type value: Any
            :return: JSON-serializable value
            :rtype: Any
            """
            try:
                import numpy as np
            except Exception:
                np = None
            if isinstance(value, dict):
                return {k: _to_serializable(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_to_serializable(v) for v in value]
            if np is not None and isinstance(value, (np.generic,)):
                return value.item()
            return value

        serializable = _to_serializable(results_with_metadata)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def print_results(self, results: Dict[str, Any]):
        """Print formatted benchmark results.

        :param results: Benchmark results
        :type results: Dict[str, Any]
        """
        print("\n" + "=" * 80)
        print("ACTIONGENOME GROUND TRUTH SUMMARIZATION BENCHMARK RESULTS")
        print("=" * 80)

        for linearizer, model_results in results.items():
            print(f"\n{linearizer.upper()} LINEARIZER:")
            print("-" * 60)

            for model_name, metrics in model_results.items():
                print(f"\n{model_name.upper()}:")
                formatted = self.metrics.format_results(metrics)
                # Indent lines for readability
                print("\n".join(["  " + line for line in formatted.splitlines()]))

        print("\n" + "=" * 80)

    def print_comparison_table(self, results: Dict[str, Any]):
        """Print a comparison table across linearizers and models.

        :param results: Benchmark results
        :type results: Dict[str, Any]
        """
        print("\n" + "=" * 100)
        print("LINEARIZER COMPARISON TABLE")
        print("=" * 100)

        # Get all models and metrics
        all_models = set()
        all_metrics = set()

        for linearizer_results in results.values():
            for model_results in linearizer_results.values():
                all_models.update(model_results.keys())
                all_metrics.update(model_results.keys())

        all_models = sorted(list(all_models))
        all_metrics = sorted(list(all_metrics))

        # Print header
        print(f"{'Linearizer':<12} {'Model':<15}", end="")
        for metric in all_metrics:
            print(f"{metric:<8}", end="")
        print()
        print("-" * (12 + 15 + 8 * len(all_metrics)))

        # Print results
        for linearizer, model_results in results.items():
            for model_name in all_models:
                if model_name in model_results:
                    metrics = model_results[model_name]
                    print(f"{linearizer:<12} {model_name:<15}", end="")
                    for metric in all_metrics:
                        value = metrics.get(metric, 0.0)
                        if isinstance(value, float):
                            print(f"{value:<8.3f}", end="")
                        else:
                            print(f"{str(value):<8}", end="")
                    print()

        print("=" * 100)


def main():
    """Example usage of ActionGenomeSummarizationBenchmark."""
    # Initialize benchmark
    data_path = "data/action_genome"
    benchmark = ActionGenomeSummarizationBenchmark(data_path)

    # Load models and dataset
    benchmark.load_models()
    benchmark.load_dataset(max_videos=50)  # Limit for testing

    # Run benchmark
    results = benchmark.run_benchmark(
        subset_size=20,
        models=["t5_base", "t5_large"],
        linearizers=["flat", "majority", "time"],
    )

    # Print results
    benchmark.print_results(results)
    benchmark.print_comparison_table(results)

    # Save results
    benchmark.save_results(results, "data/summarization/ag_ground_truth_results.json")


if __name__ == "__main__":
    main()
