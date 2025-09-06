"""
Dataset loading and preprocessing utilities for language module evaluation.

This module provides functionality to download, load, and preprocess datasets
for summarization evaluation, with a focus on MSR-VTT dataset.
"""

import json
import random
from typing import Dict, Optional
from pathlib import Path
import logging

try:
    # Import huggingface datasets directly
    import importlib
    datasets_module = importlib.import_module('datasets')
    if hasattr(datasets_module, 'load_dataset'):
        from datasets import load_dataset
        DATASETS_AVAILABLE = True
    else:
        DATASETS_AVAILABLE = False
        logging.warning("datasets module found but load_dataset not available")
except ImportError as e:
    DATASETS_AVAILABLE = False
    logging.warning(f"huggingface datasets library not available: {e}. Install with: pip install datasets")

logger = logging.getLogger(__name__)


class MSRVTTLoader:
    """Loader for MSR-VTT dataset with subset creation capabilities.
    
    Provides functionality to download MSR-VTT dataset from Hugging Face
    and create train/test subsets for evaluation.
    
    :param cache_dir: Directory to cache downloaded datasets
    :type cache_dir: str, optional
    :param subset_size: Size of subset to create (train + test)
    :type subset_size: int, optional
    """
    
    def __init__(self, cache_dir: str = "data/msr_vtt", subset_size: int = 500):
        """Initialize MSR-VTT loader.
        
        :param cache_dir: Directory to cache downloaded datasets
        :type cache_dir: str
        :param subset_size: Size of subset to create (train + test)
        :type subset_size: int
        """
        self.cache_dir = Path(cache_dir)
        self.subset_size = subset_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")
    
    def download_dataset(self) -> Dict:
        """Download MSR-VTT dataset from Hugging Face.
        
        :return: Dictionary containing train and test splits
        :rtype: Dict
        """
        logger.info("Downloading MSR-VTT dataset from Hugging Face...")
        
        try:
            # Load the full dataset
            dataset = load_dataset("zhanjun/MSR-VTT", cache_dir=str(self.cache_dir))
            logger.info("Successfully downloaded MSR-VTT dataset")
            logger.info(f"Train split: {len(dataset['train'])} samples")
            logger.info(f"Test split: {len(dataset['test'])} samples")
            
            return dataset
        except Exception as e:
            logger.error(f"Failed to download MSR-VTT dataset: {e}")
            raise
    
    def create_subset(self, dataset: Optional[Dict] = None, 
                     train_size: int = 400, test_size: int = 100,
                     random_seed: int = 42) -> Dict:
        """Create a subset of MSR-VTT dataset for evaluation.
        
        :param dataset: Pre-loaded dataset, if None will download
        :type dataset: Dict, optional
        :param train_size: Number of training samples
        :type train_size: int
        :param test_size: Number of test samples
        :type test_size: int
        :param random_seed: Random seed for reproducibility
        :type random_seed: int
        :return: Dictionary containing train and test subsets
        :rtype: Dict
        """
        if dataset is None:
            dataset = self.download_dataset()
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Get available samples
        train_data = dataset['train']
        test_data = dataset['test']
        
        # Create subsets
        train_indices = random.sample(range(len(train_data)), min(train_size, len(train_data)))
        test_indices = random.sample(range(len(test_data)), min(test_size, len(test_data)))
        
        train_subset = train_data.select(train_indices)
        test_subset = test_data.select(test_indices)
        
        subset = {
            'train': train_subset,
            'test': test_subset,
            'train_indices': train_indices,
            'test_indices': test_indices
        }
        
        # Save subset metadata
        self._save_subset_metadata(subset, train_size, test_size, random_seed)
        
        logger.info(f"Created subset: {len(train_subset)} train, {len(test_subset)} test samples")
        
        return subset
    
    def _save_subset_metadata(self, subset: Dict, train_size: int, 
                            test_size: int, random_seed: int):
        """Save subset metadata for reproducibility.
        
        :param subset: Created subset
        :type subset: Dict
        :param train_size: Number of training samples
        :type train_size: int
        :param test_size: Number of test samples
        :type test_size: int
        :param random_seed: Random seed used
        :type random_seed: int
        """
        metadata = {
            'train_size': train_size,
            'test_size': test_size,
            'random_seed': random_seed,
            'train_indices': subset['train_indices'],
            'test_indices': subset['test_indices'],
            'total_samples': len(subset['train']) + len(subset['test'])
        }
        
        metadata_path = self.cache_dir / "subset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved subset metadata to {metadata_path}")
    
    def load_subset_metadata(self) -> Optional[Dict]:
        """Load previously saved subset metadata.
        
        :return: Subset metadata if available, None otherwise
        :rtype: Optional[Dict]
        """
        metadata_path = self.cache_dir / "subset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_sample_info(self, subset: Dict, split: str = 'test', 
                       sample_idx: int = 0) -> Dict:
        """Get information about a specific sample.
        
        :param subset: Dataset subset
        :type subset: Dict
        :param split: Split name ('train' or 'test')
        :type split: str
        :param sample_idx: Sample index
        :type sample_idx: int
        :return: Sample information
        :rtype: Dict
        """
        if split not in subset:
            raise ValueError(f"Split '{split}' not found in subset")
        
        sample = subset[split][sample_idx]
        
        return {
            'video_id': sample.get('video_id', f'sample_{sample_idx}'),
            'caption': sample.get('caption', ''),
            'video_url': sample.get('video_url', ''),
            'start_time': sample.get('start_time', 0),
            'end_time': sample.get('end_time', 0),
            'duration': sample.get('end_time', 0) - sample.get('start_time', 0)
        }


def create_subset(train_size: int = 400, test_size: int = 100, 
                 cache_dir: str = "data/msr_vtt", 
                 random_seed: int = 42) -> Dict:
    """Convenience function to create MSR-VTT subset.
    
    :param train_size: Number of training samples
    :type train_size: int
    :param test_size: Number of test samples
    :type test_size: int
    :param cache_dir: Directory to cache dataset
    :type cache_dir: str
    :param random_seed: Random seed for reproducibility
    :type random_seed: int
    :return: Dataset subset
    :rtype: Dict
    """
    loader = MSRVTTLoader(cache_dir=cache_dir)
    return loader.create_subset(train_size=train_size, test_size=test_size, 
                               random_seed=random_seed)


def main():
    """Example usage of MSRVTTLoader."""
    # Create loader
    loader = MSRVTTLoader(cache_dir="data/msr_vtt")
    
    # Create subset
    subset = loader.create_subset(train_size=400, test_size=100)
    
    # Get sample info
    sample_info = loader.get_sample_info(subset, 'test', 0)
    print(f"Sample info: {sample_info}")
    
    # Print some statistics
    print(f"Train samples: {len(subset['train'])}")
    print(f"Test samples: {len(subset['test'])}")
    
    # Show a few captions
    print("\nSample captions:")
    for i in range(min(3, len(subset['test']))):
        sample = subset['test'][i]
        print(f"Sample {i}: {sample.get('caption', 'No caption')[:100]}...")


if __name__ == "__main__":
    main()
