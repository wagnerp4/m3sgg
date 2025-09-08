#!/usr/bin/env python3
"""
Command-line interface for running ActionGenome ground truth summarization benchmark.

This script provides a CLI for evaluating summarization models on ActionGenome
ground truth scene graphs with different linearization strategies.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from m3sgg.language.evaluation.ag_benchmark import ActionGenomeSummarizationBenchmark


def setup_logging(log_file: str = "data/summarization/ag_benchmark.log", verbose: bool = False):
    """Set up logging configuration.
    
    :param log_file: Path to log file
    :type log_file: str
    :param verbose: Enable verbose logging
    :type verbose: bool
    """
    # Create log directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Run ActionGenome ground truth summarization benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/action_genome",
        help="Path to ActionGenome dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to load (None for all)"
    )
    
    # Benchmark arguments
    parser.add_argument(
        "--subset-size",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on"
    )
    
    # Model arguments
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models to evaluate (default: all available)"
    )
    parser.add_argument(
        "--linearizers",
        nargs="+",
        default=["flat", "majority", "time"],
        help="List of linearization modes to test"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="data/summarization/ag_ground_truth_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="data/summarization/ag_benchmark.log",
        help="Log file path"
    )
    
    # Other arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_file, args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize benchmark
        logger.info("Initializing ActionGenome benchmark...")
        benchmark = ActionGenomeSummarizationBenchmark(
            data_path=args.data_path,
            device=args.device,
            split=args.split
        )
        
        # Load models
        logger.info("Loading models...")
        benchmark.load_models()
        
        # Load dataset
        logger.info("Loading dataset...")
        benchmark.load_dataset(max_videos=args.max_videos)
        
        # Run benchmark
        logger.info("Running benchmark...")
        results = benchmark.run_benchmark(
            subset_size=args.subset_size,
            models=args.models,
            linearizers=args.linearizers
        )
        
        # Print results
        benchmark.print_results(results)
        benchmark.print_comparison_table(results)
        
        # Save results
        logger.info(f"Saving results to {args.output}")
        benchmark.save_results(results, args.output)
        
        logger.info("Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

