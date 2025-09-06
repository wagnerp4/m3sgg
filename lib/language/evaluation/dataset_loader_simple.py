"""
Simple dataset loading utilities for language module evaluation.

This module provides a simplified approach to dataset loading that works
around the local datasets directory conflict by using mock data for testing.
"""

import json
import random
from typing import Dict, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SimpleDatasetLoader:
    """Simple dataset loader that creates mock data for testing.
    
    This loader creates synthetic video caption data for testing the
    evaluation framework without requiring external dataset downloads.
    
    :param cache_dir: Directory to cache data
    :type cache_dir: str, optional
    :param subset_size: Size of subset to create (train + test)
    :type cache_dir: int, optional
    """
    
    def __init__(self, cache_dir: str = "data/mock_dataset", subset_size: int = 500):
        """Initialize simple dataset loader.
        
        :param cache_dir: Directory to cache data
        :type cache_dir: str
        :param subset_size: Size of subset to create (train + test)
        :type subset_size: int
        """
        self.cache_dir = Path(cache_dir)
        self.subset_size = subset_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock video captions for testing
        self.mock_captions = [
            "A person is walking in the park with a dog.",
            "A man is cooking food in the kitchen.",
            "Children are playing soccer in the field.",
            "A woman is reading a book on the sofa.",
            "A cat is sleeping on the bed.",
            "A person is riding a bicycle down the street.",
            "A family is having dinner at the table.",
            "A person is working on a laptop computer.",
            "A dog is running in the backyard.",
            "A person is driving a car on the highway.",
            "A child is playing with toys on the floor.",
            "A person is exercising at the gym.",
            "A bird is flying in the sky.",
            "A person is shopping at the grocery store.",
            "A person is watching television on the couch.",
            "A person is taking a shower in the bathroom.",
            "A person is gardening in the yard.",
            "A person is playing guitar on the stage.",
            "A person is painting a picture in the studio.",
            "A person is swimming in the pool."
        ]
    
    def create_mock_dataset(self, train_size: int = 400, test_size: int = 100,
                           random_seed: int = 42) -> Dict:
        """Create a mock dataset for testing.
        
        :param train_size: Number of training samples
        :type train_size: int
        :param test_size: Number of test samples
        :type test_size: int
        :param random_seed: Random seed for reproducibility
        :type random_seed: int
        :return: Dictionary containing train and test subsets
        :rtype: Dict
        """
        random.seed(random_seed)
        
        # Create mock data
        train_data = []
        test_data = []
        
        # Generate training data
        for i in range(train_size):
            caption = random.choice(self.mock_captions)
            train_data.append({
                'video_id': f'train_video_{i:04d}',
                'caption': caption,
                'video_url': f'https://example.com/videos/train_{i:04d}.mp4',
                'start_time': 0,
                'end_time': 30,
                'duration': 30
            })
        
        # Generate test data
        for i in range(test_size):
            caption = random.choice(self.mock_captions)
            test_data.append({
                'video_id': f'test_video_{i:04d}',
                'caption': caption,
                'video_url': f'https://example.com/videos/test_{i:04d}.mp4',
                'start_time': 0,
                'end_time': 30,
                'duration': 30
            })
        
        # Create mock dataset structure
        dataset = {
            'train': train_data,
            'test': test_data
        }
        
        # Save dataset
        self._save_mock_dataset(dataset, train_size, test_size, random_seed)
        
        logger.info(f"Created mock dataset: {len(train_data)} train, {len(test_data)} test samples")
        
        return dataset
    
    def _save_mock_dataset(self, dataset: Dict, train_size: int, 
                          test_size: int, random_seed: int):
        """Save mock dataset to files.
        
        :param dataset: Mock dataset
        :type dataset: Dict
        :param train_size: Number of training samples
        :type train_size: int
        :param test_size: Number of test samples
        :type test_size: int
        :param random_seed: Random seed used
        :type random_seed: int
        """
        # Save train data
        train_path = self.cache_dir / "train_data.json"
        with open(train_path, 'w') as f:
            json.dump(dataset['train'], f, indent=2)
        
        # Save test data
        test_path = self.cache_dir / "test_data.json"
        with open(test_path, 'w') as f:
            json.dump(dataset['test'], f, indent=2)
        
        # Save metadata
        metadata = {
            'train_size': train_size,
            'test_size': test_size,
            'random_seed': random_seed,
            'total_samples': len(dataset['train']) + len(dataset['test']),
            'dataset_type': 'mock'
        }
        
        metadata_path = self.cache_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved mock dataset to {self.cache_dir}")
    
    def load_mock_dataset(self) -> Optional[Dict]:
        """Load previously saved mock dataset.
        
        :return: Mock dataset if available, None otherwise
        :rtype: Optional[Dict]
        """
        train_path = self.cache_dir / "train_data.json"
        test_path = self.cache_dir / "test_data.json"
        
        if train_path.exists() and test_path.exists():
            with open(train_path, 'r') as f:
                train_data = json.load(f)
            with open(test_path, 'r') as f:
                test_data = json.load(f)
            
            return {
                'train': train_data,
                'test': test_data
            }
        
        return None
    
    def get_sample_info(self, dataset: Dict, split: str = 'test', 
                       sample_idx: int = 0) -> Dict:
        """Get information about a specific sample.
        
        :param dataset: Dataset
        :type dataset: Dict
        :param split: Split name ('train' or 'test')
        :type split: str
        :param sample_idx: Sample index
        :type sample_idx: int
        :return: Sample information
        :rtype: Dict
        """
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found in dataset")
        
        if sample_idx >= len(dataset[split]):
            raise ValueError(f"Sample index {sample_idx} out of range for {split} split")
        
        return dataset[split][sample_idx]
    
    def get_captions(self, dataset: Dict, split: str = 'test') -> List[str]:
        """Get all captions from a dataset split.
        
        :param dataset: Dataset
        :type dataset: Dict
        :param split: Split name ('train' or 'test')
        :type split: str
        :return: List of captions
        :rtype: List[str]
        """
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found in dataset")
        
        return [sample['caption'] for sample in dataset[split]]


def create_mock_subset(train_size: int = 400, test_size: int = 100, 
                      cache_dir: str = "data/mock_dataset", 
                      random_seed: int = 42) -> Dict:
    """Convenience function to create mock dataset subset.
    
    :param train_size: Number of training samples
    :type train_size: int
    :param test_size: Number of test samples
    :type test_size: int
    :param cache_dir: Directory to cache dataset
    :type cache_dir: str
    :param random_seed: Random seed for reproducibility
    :type random_seed: int
    :return: Mock dataset
    :rtype: Dict
    """
    loader = SimpleDatasetLoader(cache_dir=cache_dir)
    return loader.create_mock_dataset(train_size=train_size, test_size=test_size, 
                                     random_seed=random_seed)


def main():
    """Example usage of SimpleDatasetLoader."""
    # Create loader
    loader = SimpleDatasetLoader(cache_dir="data/mock_dataset")
    
    # Create mock dataset
    dataset = loader.create_mock_dataset(train_size=10, test_size=5)
    
    # Get sample info
    sample_info = loader.get_sample_info(dataset, 'test', 0)
    print(f"Sample info: {sample_info}")
    
    # Print some statistics
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Show a few captions
    print("\nSample captions:")
    for i in range(min(3, len(dataset['test']))):
        sample = dataset['test'][i]
        print(f"Sample {i}: {sample['caption']}")


if __name__ == "__main__":
    main()
