"""
Test configuration and utilities for M3SGG tests
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Test configuration
TEST_CONFIG: Dict[str, Any] = {
    "test_data_dir": project_root / "tests" / "fixtures",
    "sample_video_dir": project_root / "tests" / "fixtures" / "sample_videos",
    "sample_annotation_dir": project_root / "tests" / "fixtures" / "sample_annotations",
    "temp_dir": project_root / "tests" / "temp",
    "cache_dir": project_root / "tests" / "cache",
    "timeout": {
        "unit": 30,
        "integration": 300,
        "performance": 600
    },
    "memory_limit": "2GB",
    "gpu_required": False
}

def get_test_data_path(relative_path: str) -> Path:
    """Get absolute path to test data file"""
    return TEST_CONFIG["test_data_dir"] / relative_path

def get_temp_path(relative_path: str) -> Path:
    """Get absolute path to temporary test file"""
    temp_dir = TEST_CONFIG["temp_dir"]
    temp_dir.mkdir(exist_ok=True)
    return temp_dir / relative_path

def setup_test_environment():
    """Set up test environment variables and paths"""
    os.environ["PYTHONPATH"] = f"{project_root}{os.pathsep}{project_root / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    os.environ["M3SGG_TEST_MODE"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU for tests unless explicitly needed

def cleanup_test_environment():
    """Clean up test environment"""
    if "M3SGG_TEST_MODE" in os.environ:
        del os.environ["M3SGG_TEST_MODE"]

# Test markers for pytest
TEST_MARKERS = {
    "unit": "Fast, isolated unit tests",
    "integration": "Integration tests that test component interactions",
    "performance": "Performance and resource-intensive tests",
    "slow": "Tests that take more than 30 seconds",
    "gpu": "Tests that require GPU acceleration",
    "gui": "Tests that require GUI components",
    "network": "Tests that require network access",
    "data": "Tests that require large data files"
}
