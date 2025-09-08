"""
Benchmark execution and result management for language module evaluation.

This module provides the main benchmark class for evaluating summarization models
on video caption generation tasks using scene graph data.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
# TODO: Add numpy import back if needed

# Import project modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from m3sgg.language.summarization.wrappers import (
    T5SummarizationWrapper, 
    PegasusSummarizationWrapper
)
from m3sgg.language.evaluation.dataset_loader import MSRVTTLoader
from m3sgg.language.evaluation.metrics import SummarizationMetrics

logger = logging.getLogger(__name__)


class SummarizationBenchmark:
    """Main benchmark class for summarization evaluation.
    
    Provides functionality to run comprehensive benchmarks on summarization models
    using scene graph generation and text summarization pipelines.
    
    :param checkpoint_path: Path to STTran checkpoint
    :type checkpoint_path: str
    :param device: Device to run inference on
    :type device: str, optional
    :param cache_dir: Directory to cache datasets
    :type cache_dir: str, optional
    """
    
    def __init__(self, checkpoint_path: str, device: str = "cuda:0", 
                 cache_dir: str = "data/msr_vtt",
                 video_root: str = "data/msr_vtt/videos",
                 sg_cache_dir: str = "data/summarization/cache",
                 frames_per_clip: int = 8,
                 linearizer: str = "flat",
                 variant: str = "sg",
                 linearizers: Optional[List[str]] = None,
                 variants: Optional[List[str]] = None):
        """Initialize summarization benchmark.
        
        :param checkpoint_path: Path to STTran checkpoint
        :type checkpoint_path: str
        :param device: Device to run inference on
        :type device: str
        :param cache_dir: Directory to cache datasets
        :type cache_dir: str
        """
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        self.video_root = video_root
        self.sg_cache_dir = Path(sg_cache_dir)
        self.frames_per_clip = frames_per_clip
        self.linearizer = linearizer
        self.variant = variant
        self.linearizers = linearizers
        self.variants = variants
        self.sg_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.config = None
        self.object_detector = None
        self.sgg_model = None
        self.summarization_models = {}
        self.metrics = SummarizationMetrics()
        
        logger.info(f"Initialized benchmark with device: {self.device}")
    
    def load_models(self, config_path: Optional[str] = None):
        """Load all required models for evaluation.
        
        :param config_path: Path to config file, if None uses default
        :type config_path: Optional[str]
        """
        logger.info("Loading models...")
        
        # TODO: Load configuration if/when scene graph generation is implemented
        # Skipping object detector and SGG model loading to avoid CLI side-effects from training modules
        self.config = None
        self.object_detector = None
        self.sgg_model = None
        
        # Initialize summarization models
        logger.info("Initializing summarization models...")
        self.summarization_models = {
            "t5_base": T5SummarizationWrapper("google-t5/t5-base", device=self.device),
            "t5_large": T5SummarizationWrapper("google-t5/t5-large", device=self.device),
            "pegasus_xsum": PegasusSummarizationWrapper("google/pegasus-xsum", device=self.device),
            "pegasus_cnn": PegasusSummarizationWrapper("google/pegasus-cnn_dailymail", device=self.device),
        }
        
        logger.info("All models loaded successfully")
    
    def generate_scene_graph(self, video_path: str) -> Dict[str, Any]:
        """Generate scene graph for a video.
        
        :param video_path: Path to video file
        :type video_path: str
        :return: Scene graph data
        :rtype: Dict[str, Any]
        """
        # TODO: Implement video processing and scene graph generation
        # This is a placeholder - you'll need to implement the actual pipeline
        # that processes videos, extracts frames, runs object detection,
        # and generates scene graphs
        
        # Cache by video stem
        cache_key = Path(video_path).stem + ".json"
        cache_path = self.sg_cache_dir / cache_key
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        logger.warning("Scene graph generation not implemented yet")
        graph = {'objects': [], 'relationships': [], 'triples': []}
        try:
            with open(cache_path, 'w') as f:
                json.dump(graph, f)
        except Exception:
            pass
        return graph
    
    def scene_graph_to_text(self, scene_graph: Dict[str, Any]) -> str:
        """Convert scene graph to text description.
        
        :param scene_graph: Scene graph data
        :type scene_graph: Dict[str, Any]
        :return: Text description
        :rtype: str
        """
        from m3sgg.language.summarization.summarize import linearize_triples
        
        # Extract triples from scene graph
        triples = scene_graph.get('triples', [])
        
        if not triples:
            return "No objects or relationships detected in the scene."
        
        # Convert triples to sentences
        sentences = linearize_triples(triples, mode=self.linearizer)
        
        # Join sentences
        return " ".join(sentences)
    
    def generate_summary(self, text: str, model_name: str = 't5_base') -> str:
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
    
    def run_scenario1_benchmark(self, subset_size: int = 100, 
                               models: List[str] = None) -> Dict[str, Any]:
        """Run Scenario 1: Video Caption Generation benchmark.
        
        :param subset_size: Number of test samples to use
        :type subset_size: int
        :param models: List of model names to evaluate
        :type models: List[str], optional
        :return: Benchmark results
        :rtype: Dict[str, Any]
        """
        if models is None:
            models = list(self.summarization_models.keys())
        
        logger.info(f"Running Scenario 1 benchmark with {subset_size} samples")
        
        # Load dataset with fallback to mock data if HF dataset is unavailable
        try:
            loader = MSRVTTLoader(cache_dir=self.cache_dir)
            subset = loader.create_subset(train_size=400, test_size=subset_size)
        except Exception as dataset_error:
            logger.warning(f"Falling back to mock dataset due to error: {dataset_error}")
            from m3sgg.language.evaluation.dataset_loader_simple import SimpleDatasetLoader
            simple_loader = SimpleDatasetLoader(cache_dir="data/mock_dataset")
            subset = simple_loader.create_mock_dataset(train_size=400, test_size=subset_size)
        
        results = {}
        
        # Determine sweeps
        variants = self.variants or [self.variant]
        linearizers = self.linearizers or [self.linearizer]

        for variant in variants:
            results.setdefault(variant, {})
            for linearizer in linearizers:
                logger.info(f"Evaluating variant={variant}, linearizer={linearizer}")
                self.variant = variant
                self.linearizer = linearizer
                for model_name in models:
                    logger.info(f"Evaluating model: {model_name}")
                    predictions = []
                    references = []
                    for i in range(min(subset_size, len(subset['test']))):
                        sample = subset['test'][i]
                        reference = sample.get('caption', '')
                        references.append(reference)
                        if self.variant == "sg":
                            video_id = sample.get('video_id', '')
                            video_path = os.path.join(self.video_root, f"{video_id}.mp4")
                            scene_graph = self.generate_scene_graph(video_path)
                            text_description = self.scene_graph_to_text(scene_graph)
                        elif self.variant == "no_sg":
                            text_description = ""
                        elif self.variant == "caption_input":
                            text_description = reference
                        else:
                            text_description = ""
                        prediction = self.generate_summary(text_description, model_name)
                        predictions.append(prediction)
                        if i % 10 == 0:
                            logger.info(f"Processed {i+1}/{subset_size} samples")
                    metrics = self.metrics.compute_all_metrics(predictions, references)
                    results[variant].setdefault(linearizer, {})
                    results[variant][linearizer][model_name] = metrics
                    logger.info(f"Completed evaluation for {model_name} (variant={variant}, linearizer={linearizer})")
        
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
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': self.device,
            'checkpoint_path': self.checkpoint_path,
            'results': results
        }
        
        def _to_serializable(value: Any) -> Any:
            """Convert values to JSON-serializable types.
            
            :param value: Value to convert
            :type value: Any
            :return: JSON-serializable value
            :rtype: Any
            """
            try:
                import numpy as np  # local import to avoid hard dependency at module import
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
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted benchmark results.
        
        :param results: Benchmark results
        :type results: Dict[str, Any]
        """
        print("\n" + "="*60)
        print("SUMMARIZATION BENCHMARK RESULTS")
        print("="*60)
        
        # Support both flat (model->metrics) and nested (variant->linearizer->model->metrics)
        nested = False
        if results and isinstance(next(iter(results.values())), dict):
            # Heuristic: nested if values contain dicts whose values are also dicts of models
            first_val = next(iter(results.values()))
            if first_val and isinstance(next(iter(first_val.values())), dict):
                nested = True
        
        if not nested:
            for model_name, metrics in results.items():
                print(f"\n{model_name.upper()}:")
                print("-" * 40)
                print(self.metrics.format_results(metrics))
        else:
            for variant, by_lin in results.items():
                print(f"\n{variant.upper()}:")
                print("-" * 40)
                for linearizer, by_model in by_lin.items():
                    print(f"\n  Linearizer: {linearizer}")
                    for model_name, metrics in by_model.items():
                        print(f"    {model_name.upper()}:")
                        formatted = self.metrics.format_results(metrics)
                        # Indent lines for readability
                        print("\n".join(["      "+line for line in formatted.splitlines()]))
        
        print("\n" + "="*60)


def main():
    """Example usage of SummarizationBenchmark."""
    # Initialize benchmark
    checkpoint_path = "data/checkpoints/sgdet_test/model_best.tar"
    benchmark = SummarizationBenchmark(checkpoint_path)
    
    # Load models
    benchmark.load_models()
    
    # Run benchmark
    results = benchmark.run_scenario1_benchmark(subset_size=10)
    
    # Print results
    benchmark.print_results(results)
    
    # Save results
    benchmark.save_results(results, "output/summarization_benchmark_results.json")


if __name__ == "__main__":
    main()
