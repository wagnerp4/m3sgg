#!/usr/bin/env python3
"""
Script to pre-cache word vectors for faster loading.
This script loads the word vectors once and saves them to a pickle file
for much faster subsequent loading.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.word_vectors import load_word_vectors


def main():
    """Pre-cache word vectors for common configurations."""
    print("Pre-caching word vectors for faster loading...")

    # Common configurations used in the models
    configurations = [
        {"wv_type": "glove.6B", "wv_dir": "data", "wv_dim": 200},
        {"wv_type": "glove.6B", "wv_dir": "data", "wv_dim": 300},
    ]

    for config in configurations:
        print(f"\nCaching {config['wv_type']} {config['wv_dim']}d vectors...")
        try:
            word_vectors = load_word_vectors(**config)
            print(f"Successfully cached {len(word_vectors)} word vectors")
        except Exception as e:
            print(f"Error caching {config}: {e}")

    print("\nWord vector caching complete!")
    print("Subsequent runs should load much faster from cache.")


if __name__ == "__main__":
    main()
