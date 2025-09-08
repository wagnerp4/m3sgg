"""
Test script to verify the testing setup works correctly
"""

import pytest
import sys
from pathlib import Path

# Test that imports work correctly
def test_imports():
    """Test that basic imports work"""
    try:
        import numpy as np
        import torch
        import cv2
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required packages: {e}")

def test_project_structure():
    """Test that project structure is correct"""
    project_root = Path(__file__).parent.parent.parent.parent
    src_dir = project_root / "src" / "m3sgg"
    
    assert project_root.exists(), "Project root should exist"
    assert src_dir.exists(), "Source directory should exist"
    assert (project_root / "pyproject.toml").exists(), "pyproject.toml should exist"

def test_test_config():
    """Test that test configuration loads correctly"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from test_config import TEST_CONFIG
    
    assert "test_data_dir" in TEST_CONFIG
    assert "timeout" in TEST_CONFIG
    assert TEST_CONFIG["timeout"]["unit"] == 30

def test_pytest_markers():
    """Test that pytest markers are defined"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from test_config import TEST_MARKERS
    
    expected_markers = ["unit", "integration", "performance", "slow", "gpu", "gui", "network", "data"]
    for marker in expected_markers:
        assert marker in TEST_MARKERS, f"Marker '{marker}' should be defined"

@pytest.mark.unit
def test_unit_marker():
    """Test that unit marker works"""
    assert True

@pytest.mark.slow
def test_slow_marker():
    """Test that slow marker works"""
    assert True

def test_environment_setup():
    """Test that environment setup works"""
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from test_config import setup_test_environment, cleanup_test_environment
    
    setup_test_environment()
    assert "M3SGG_TEST_MODE" in os.environ
    cleanup_test_environment()
    assert "M3SGG_TEST_MODE" not in os.environ
