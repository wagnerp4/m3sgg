"""M3SGG CLI entry point.

This module provides the main CLI interface for the M3SGG framework.
It can be run using: python -m m3sgg <command> [options]

:author: M3SGG Team
:version: 0.1.0
"""

import sys
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from m3sgg.cli.main import main

if __name__ == "__main__":
    main()
