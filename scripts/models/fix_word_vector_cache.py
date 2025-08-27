#!/usr/bin/env python3
"""
Script to fix word vector cache issues by clearing and recreating the cache.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.word_vectors import clear_word_vector_cache, load_word_vectors


def main():
    """Fix word vector cache by clearing and recreating."""
    print("Fixing Word Vector Cache")
    print("=" * 50)

    # Clear all caches
    print("Clearing existing caches...")
    clear_word_vector_cache()

    # Recreate caches
    configs = [
        {"wv_type": "glove.6B", "wv_dir": "data", "wv_dim": 200},
        {"wv_type": "glove.6B", "wv_dir": "data", "wv_dim": 300},
    ]

    for config in configs:
        print(f"\nRecreating cache for {config['wv_type']} {config['wv_dim']}d...")
        try:
            word_vectors = load_word_vectors(**config)
            print(f"✅ Successfully cached {len(word_vectors)} vectors")
        except Exception as e:
            print(f"❌ Error caching {config}: {e}")

    print("\nCache recreation complete!")


if __name__ == "__main__":
    main()
