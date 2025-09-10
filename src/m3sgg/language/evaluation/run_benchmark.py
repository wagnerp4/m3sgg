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
project_root = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
sys.path.insert(0, project_root)

from m3sgg.language.evaluation.benchmark import SummarizationBenchmark


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    # Ensure log directory exists
    log_dir = Path("data/summarization")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_dir / "summarization_benchmark.log")),
        ],
    )


def main():
    """Main function to run summarization benchmark."""
    parser = argparse.ArgumentParser(
        description="Run summarization benchmark evaluation"
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to STTran checkpoint file",
        default="data/checkpoints/sgdet_test/model_best.tar",
    )

    # Optional arguments
    parser.add_argument(
        "--subset-size",
        type=int,
        default=100,
        help="Number of test samples to use (default: 100)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models to evaluate; if omitted, all available wrappers are used",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to run inference on (default: cuda:0)",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/msr_vtt",
        help="Directory to cache datasets (default: data/msr_vtt)",
    )
    parser.add_argument(
        "--output",
        default="data/summarization/summarization_benchmark_results.json",
        help="Output file for results (default: data/summarization/summarization_benchmark_results.json)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Scene graph and input control
    parser.add_argument(
        "--video-root",
        default="data/msr_vtt/videos",
        help="Root directory containing MSR-VTT video files",
    )
    parser.add_argument(
        "--sg-cache-dir",
        default="data/summarization/cache",
        help="Directory to cache generated scene graphs",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=8,
        help="Number of frames to sample per clip for SGG",
    )
    parser.add_argument(
        "--linearizer",
        choices=["flat", "majority", "time"],
        default="flat",
        help="Linearization strategy for scene graphs",
    )
    parser.add_argument(
        "--variant",
        choices=["sg", "no_sg", "caption_input"],
        default="sg",
        help="Input variant: use scene graph, empty input, or reference caption as input",
    )
    parser.add_argument(
        "--linearizers",
        nargs="+",
        default=None,
        help="Optional list to compare multiple linearizers e.g. flat majority time",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional list to compare multiple input variants e.g. sg no_sg caption_input",
    )

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
            cache_dir=args.cache_dir,
            video_root=args.video_root,
            sg_cache_dir=args.sg_cache_dir,
            frames_per_clip=args.frames_per_clip,
            linearizer=args.linearizer,
            variant=args.variant,
            linearizers=args.linearizers,
            variants=args.variants,
        )

        # Load models
        logger.info("Loading models...")
        benchmark.load_models()

        # Run benchmark
        logger.info(f"Running benchmark with {args.subset_size} samples...")
        results = benchmark.run_scenario1_benchmark(
            subset_size=args.subset_size, models=args.models
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
