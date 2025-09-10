"""
Shared pytest fixtures and configuration
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


@pytest.fixture
def test_data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_video_path():
    """Path to sample video for testing"""
    return test_data_dir() / "sample_videos"


@pytest.fixture
def sample_annotation_path():
    """Path to sample annotations for testing"""
    return test_data_dir() / "sample_annotations"


@pytest.fixture
def stage_name():
    """Training stage name for testing"""
    return "vqvae"


@pytest.fixture
def entry():
    """Mock entry data for testing"""
    import torch

    return {
        "labels": torch.tensor([0, 1, 2], dtype=torch.long),
        "boxes": torch.tensor(
            [
                [0, 100, 100, 200, 200],
                [0, 150, 150, 250, 250],
                [0, 200, 200, 300, 300],
            ],
            dtype=torch.float32,
        ),
        "pair_idx": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    }


@pytest.fixture
def dataset():
    """Mock dataset for testing"""

    class MockDataset:
        def __init__(self):
            self.video_list = [f"video_{i}.mp4" for i in range(10)]
            self.gt_annotations = [
                {"objects": [], "relationships": []} for _ in range(10)
            ]
            self.object_classes = ["person", "car", "bike", "dog", "cat"]
            self.attention_relationships = ["looking_at", "watching", "observing"]
            self.spatial_relationships = [
                "above",
                "below",
                "left_of",
                "right_of",
                "near",
                "far",
            ]
            self.contacting_relationships = [
                "touching",
                "holding",
                "carrying",
                "pushing",
                "pulling",
                "sitting_on",
                "standing_on",
                "lying_on",
                "riding",
                "wearing",
                "eating",
                "drinking",
                "using",
                "playing_with",
                "opening",
                "closing",
                "cleaning",
            ]
            self.relationship_classes = (
                self.attention_relationships
                + self.spatial_relationships
                + self.contacting_relationships
            )

    return MockDataset()


@pytest.fixture
def device():
    """Device for testing"""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
