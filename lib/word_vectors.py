import array
import os
import pickle
import ssl
import sys
import urllib.request
import zipfile

import numpy as np
import six
import torch
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm

# Global cache for word vectors to avoid loading the same file multiple times
_word_vector_cache = {}


def create_ssl_context():
    """Create an unverified SSL context to bypass certificate verification."""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context


def download_word_vectors(wv_type, wv_dir, wv_dim):
    """Download word vectors if they don't exist."""
    if not os.path.exists(wv_dir):
        os.makedirs(wv_dir)

    wv_path = os.path.join(wv_dir, f"{wv_type}.{wv_dim}d.txt")
    if not os.path.exists(wv_path):
        print(f"Downloading {wv_type} word vectors...")
        try:
            url = f"https://nlp.stanford.edu/data/{wv_type}.zip"
            zip_path = os.path.join(wv_dir, f"{wv_type}.zip")

            # Download with SSL verification disabled
            context = create_ssl_context()
            urllib.request.urlretrieve(url, zip_path, context=context)

            # Extract the specific dimension file
            import zipfile

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extract(f"{wv_type}.{wv_dim}d.txt", wv_dir)

            # Clean up zip file
            os.remove(zip_path)
        except Exception as e:
            print(f"Error downloading word vectors: {e}")
            print(
                f"Please download {wv_type}.zip manually from http://nlp.stanford.edu/data/{wv_type}.zip"
            )
            print(f"Extract {wv_type}.{wv_dim}d.txt to {wv_dir}")
            raise
    return wv_path


def get_cache_path(wv_type, wv_dir, wv_dim):
    """Get the path for the cached word vectors."""
    cache_dir = os.path.join(wv_dir, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return os.path.join(cache_dir, f"{wv_type}.{wv_dim}d.pkl")


def get_cache_status():
    """Get the status of word vector caches."""
    cache_dir = os.path.join("data", "cache")
    status = {
        "memory_cache": len(_word_vector_cache),
        "memory_cache_keys": list(_word_vector_cache.keys()),
        "disk_cache": {},
        "cache_dir": cache_dir,
    }

    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.endswith(".pkl"):
                file_path = os.path.join(cache_dir, file)
                file_size = os.path.getsize(file_path)
                status["disk_cache"][file] = {
                    "size_mb": file_size / (1024 * 1024),
                    "path": file_path,
                }

    return status


def clear_word_vector_cache(wv_type=None, wv_dir="data", wv_dim=None):
    """Clear word vector cache (both memory and disk cache)."""
    global _word_vector_cache

    # Clear memory cache
    if wv_type is None and wv_dim is None:
        # Clear all memory cache
        _word_vector_cache.clear()
        print("Cleared all word vector memory cache")
    else:
        # Clear specific cache entries
        keys_to_remove = []
        for key in _word_vector_cache.keys():
            if wv_type is None or wv_type in key:
                if wv_dim is None or f"{wv_dim}d" in key:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del _word_vector_cache[key]
        print(f"Cleared {len(keys_to_remove)} word vector memory cache entries")

    # Clear disk cache
    cache_path = get_cache_path(wv_type or "glove.6B", wv_dir, wv_dim or 200)
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            print(f"Removed disk cache: {cache_path}")
        except Exception as e:
            print(f"Error removing disk cache: {e}")


def load_word_vectors(wv_type="glove.6B", wv_dir="data", wv_dim=200):
    """Load word vectors from file or download if not present."""
    # Check in-memory cache first
    cache_key = f"{wv_type}_{wv_dir}_{wv_dim}"
    if cache_key in _word_vector_cache:
        return _word_vector_cache[cache_key]

    # Check persistent cache
    cache_path = get_cache_path(wv_type, wv_dir, wv_dim)
    if os.path.exists(cache_path):
        try:
            print(f"Loading {wv_type} word vectors from cache...")
            with open(cache_path, "rb") as f:
                word_vectors = pickle.load(f)
            print(f"Loaded {len(word_vectors)} word vectors from cache")
            # Store in memory cache
            _word_vector_cache[cache_key] = word_vectors
            return word_vectors
        except Exception as e:
            print(f"Error loading from cache: {e}, falling back to text file...")

    # Load from text file if cache doesn't exist
    wv_path = download_word_vectors(wv_type, wv_dir, wv_dim)

    # Load word vectors (only show progress bar on first load)
    word_vectors = {}
    print(f"Loading {wv_type} word vectors from disk...")
    with open(wv_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading word vectors"):
            word, vector = line.split(" ", 1)
            word_vectors[word] = np.array([float(x) for x in vector.split()])

    # Save to persistent cache for future use
    try:
        print(f"Saving {len(word_vectors)} word vectors to cache...")
        with open(cache_path, "wb") as f:
            pickle.dump(word_vectors, f)
        print(f"Word vectors cached to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save to cache: {e}")

    # Cache the result in memory
    _word_vector_cache[cache_key] = word_vectors
    print(f"Loaded and cached {len(word_vectors)} word vectors")
    return word_vectors


def obj_edge_vectors(names, wv_type="glove.6B", wv_dir="data", wv_dim=200):
    """Create word vectors for object classes."""
    wv_dict = load_word_vectors(wv_type, wv_dir, wv_dim)

    vectors = []
    for name in names:
        name = name.lower().strip()
        if name in wv_dict:
            vectors.append(wv_dict[name])
        else:
            # If word not found, use zero vector
            vectors.append(np.zeros(wv_dim))

    vectors = np.array(vectors)
    return torch.from_numpy(vectors).float()


def verb_edge_vectors(names, wv_type="glove.6B", wv_dir=None, wv_dim=300):
    """Create word vectors for verb classes. For now, using the same logic as obj_edge_vectors."""
    return obj_edge_vectors(names, wv_type, wv_dir, wv_dim)


URL = {
    "glove.42B": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
    "glove.840B": "http://nlp.stanford.edu/data/glove.840B.300d.zip",
    "glove.twitter.27B": "http://nlp.stanford.edu/data/glove.twitter.27B.zip",
    "glove.6B": "http://nlp.stanford.edu/data/glove.6B.zip",
}


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner
