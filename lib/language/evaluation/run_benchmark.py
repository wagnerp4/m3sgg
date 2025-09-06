#!/usr/bin/env python3
"""
Main script to run summarization benchmarks.

This script provides a command-line interface for running the summarization
benchmark evaluation with various configuration options.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.insert(0, project_root)

from lib.language.evaluation.benchmark import SummarizationBenchmark

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('summarization_benchmark.log')
        ]
    )

def main():
    """Main function to run summarization benchmark."""
    parser = argparse.ArgumentParser(description="Run summarization benchmark evaluation")
    
    # Required arguments
    parser.add_argument("--checkpoint", required=True, 
                       help="Path to STTran checkpoint file", default="data/checkpoints/sgdet_test/model_best.tar")
    
    # Optional arguments
    parser.add_argument("--subset-size", type=int, default=100,
                       help="Number of test samples to use (default: 100)")
    parser.add_argument("--models", nargs="+", 
                       default=['t5_base', 'pegasus_xsum', 'pegasus_custom'],
                       help="List of models to evaluate")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to run inference on (default: cuda:0)")
    parser.add_argument("--cache-dir", default="data/msr_vtt",
                       help="Directory to cache datasets (default: data/msr_vtt)")
    parser.add_argument("--output", default="output/summarization_benchmark_results.json",
                       help="Output file for results (default: output/summarization_benchmark_results.json)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize benchmark
        logger.info("Initializing summarization benchmark...")
        benchmark = SummarizationBenchmark(
            checkpoint_path=args.checkpoint,
            device=args.device,
            cache_dir=args.cache_dir
        )
        
        # Load models
        logger.info("Loading models...")
        benchmark.load_models()
        
        # Run benchmark
        logger.info(f"Running benchmark with {args.subset_size} samples...")
        results = benchmark.run_scenario1_benchmark(
            subset_size=args.subset_size,
            models=args.models
        )
        
        # Print results
        benchmark.print_results(results)
        
        # Save results
        benchmark.save_results(results, args.output)
        
        logger.info("Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
