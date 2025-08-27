#!/usr/bin/env python3
"""
Script to check the status of word vector caches.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.word_vectors import get_cache_status


def main():
    """Check and display word vector cache status."""
    print("Word Vector Cache Status")
    print("=" * 50)

    status = get_cache_status()

    print(f"Memory Cache Entries: {status['memory_cache']}")
    if status["memory_cache_keys"]:
        print("Memory Cache Keys:")
        for key in status["memory_cache_keys"]:
            print(f"  - {key}")

    print(f"\nDisk Cache Directory: {status['cache_dir']}")
    if status["disk_cache"]:
        print("Disk Cache Files:")
        for filename, info in status["disk_cache"].items():
            print(f"  - {filename}: {info['size_mb']:.2f} MB")
    else:
        print("  No disk cache files found")

    print("\nCache Status Summary:")
    if status["memory_cache"] > 0:
        print("✅ Memory cache is populated")
    else:
        print("❌ Memory cache is empty")

    if status["disk_cache"]:
        print("✅ Disk cache files exist")
    else:
        print("❌ No disk cache files found")
        print("   Run precache_word_vectors.py to create disk cache")


if __name__ == "__main__":
    main()
