#!/usr/bin/env python3
"""
Demo benchmark script using mock data.

This script demonstrates the complete evaluation pipeline using mock data
to avoid external dataset dependency issues.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.insert(0, project_root)

from m3sgg.language.evaluation.dataset_loader_simple import create_mock_subset
from m3sgg.language.evaluation.metrics import SummarizationMetrics

def demo_benchmark():
    """Run a demo benchmark with mock data."""
    print("🎬 Running Demo Summarization Benchmark")
    print("=" * 50)
    
    # Create mock dataset
    print("📊 Creating mock dataset...")
    dataset = create_mock_subset(train_size=20, test_size=10)
    
    # Get test captions (ground truth)
    test_captions = [sample['caption'] for sample in dataset['test']]
    
    # Create mock predictions (simulating different summarization models)
    print("🤖 Generating mock predictions...")
    
    # Model 1: T5-style predictions (slightly different wording)
    t5_predictions = []
    for caption in test_captions:
        # Simple mock transformation
        if "person" in caption:
            pred = caption.replace("person", "individual")
        elif "man" in caption:
            pred = caption.replace("man", "gentleman")
        elif "woman" in caption:
            pred = caption.replace("woman", "lady")
        else:
            pred = caption
        t5_predictions.append(pred)
    
    # Model 2: Pegasus-style predictions (more concise)
    pegasus_predictions = []
    for caption in test_captions:
        # Make it more concise
        words = caption.split()
        if len(words) > 6:
            pred = " ".join(words[:6]) + "."
        else:
            pred = caption
        pegasus_predictions.append(pred)
    
    # Model 3: Custom predictions (different style)
    custom_predictions = []
    for caption in test_captions:
        # Add some variation
        if "is" in caption:
            pred = caption.replace("is", "was")
        else:
            pred = caption
        custom_predictions.append(pred)
    
    # Initialize metrics
    print("📏 Computing evaluation metrics...")
    metrics = SummarizationMetrics()
    
    # Evaluate each model
    models = {
        'T5-base': t5_predictions,
        'Pegasus-xsum': pegasus_predictions,
        'Pegasus-custom': custom_predictions
    }
    
    results = {}
    
    for model_name, predictions in models.items():
        print(f"\n🔍 Evaluating {model_name}...")
        
        # Compute all metrics
        model_results = metrics.compute_all_metrics(predictions, test_captions)
        results[model_name] = model_results
        
        # Print results for this model
        print(metrics.format_results(model_results))
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    # Create comparison table
    print(f"{'Model':<15} {'ROUGE-1':<8} {'ROUGE-2':<8} {'ROUGE-L':<8} {'BLEU-1':<8} {'METEOR':<8}")
    print("-" * 60)
    
    for model_name, model_results in results.items():
        print(f"{model_name:<15} "
              f"{model_results['rouge1']:<8.3f} "
              f"{model_results['rouge2']:<8.3f} "
              f"{model_results['rougeL']:<8.3f} "
              f"{model_results['bleu1']:<8.3f} "
              f"{model_results['meteor']:<8.3f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['rouge1'])
    print(f"\n🏆 Best performing model: {best_model}")
    print(f"   ROUGE-1 Score: {results[best_model]['rouge1']:.3f}")
    
    # Sample predictions
    print(f"\n📝 Sample Predictions (Model: {best_model}):")
    print("-" * 40)
    for i in range(min(3, len(test_captions))):
        print(f"Ground Truth: {test_captions[i]}")
        print(f"Prediction:   {models[best_model][i]}")
        print()
    
    print("✅ Demo benchmark completed successfully!")
    return results

def main():
    """Main function."""
    try:
        results = demo_benchmark()
        return True
    except Exception as e:
        print(f"❌ Demo benchmark failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
