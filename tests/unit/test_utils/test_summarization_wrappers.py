#!/usr/bin/env python3
"""
Test script for the summarization wrapper classes.
Demonstrates usage of different summarization models and wrapper classes.
"""

import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if SentencePiece is available
try:
    import sentencepiece

    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

from m3sgg.language.summarization.wrappers import (
    PegasusCustomConfig,
    PegasusSeparateLoader,
    PegasusSummarizationWrapper,
    T5SummarizationWrapper,
)


@pytest.mark.skipif(not SENTENCEPIECE_AVAILABLE, reason="SentencePiece not available")
def test_t5_wrapper():
    """Test the T5 summarization wrapper."""
    print("=" * 50)
    print("Testing T5 Summarization Wrapper")
    print("=" * 50)

    # Initialize T5 wrapper
    t5_wrapper = T5SummarizationWrapper("google-t5/t5-base")

    # Test text
    test_text = """
    The man is riding a bicycle through the park. The woman is holding an umbrella 
    while walking. A child is playing with a dog in the grass. A girl is wearing 
    a red dress and standing near a tree. A boy is sitting on a bench reading a book.
    """

    # Generate summary
    summary = t5_wrapper.summarize(test_text)
    print(f"Original text: {test_text.strip()}")
    print(f"T5 Summary: {summary}")
    print()


@pytest.mark.skipif(not SENTENCEPIECE_AVAILABLE, reason="SentencePiece not available")
def test_pegasus_wrapper():
    """Test the Pegasus summarization wrapper."""
    print("=" * 50)
    print("Testing Pegasus Summarization Wrapper")
    print("=" * 50)

    # Initialize Pegasus wrapper
    pegasus_wrapper = PegasusSummarizationWrapper("google/pegasus-xsum")

    # Test text
    test_text = """
    The man is riding a bicycle through the park. The woman is holding an umbrella 
    while walking. A child is playing with a dog in the grass. A girl is wearing 
    a red dress and standing near a tree. A boy is sitting on a bench reading a book.
    """

    # Generate summary
    summary = pegasus_wrapper.summarize(test_text)
    print(f"Original text: {test_text.strip()}")
    print(f"Pegasus Summary: {summary}")
    print()


@pytest.mark.skipif(not SENTENCEPIECE_AVAILABLE, reason="SentencePiece not available")
def test_pegasus_separate_loader():
    """Test the Pegasus separate loader."""
    print("=" * 50)
    print("Testing Pegasus Separate Loader")
    print("=" * 50)

    # Initialize separate loader
    separate_loader = PegasusSeparateLoader("google/pegasus-xsum")

    # Load tokenizer and model separately
    print("Loading tokenizer...")
    tokenizer = separate_loader.load_tokenizer()
    print("Loading model...")
    model = separate_loader.load_model()

    # Test text
    test_text = """
    The man is riding a bicycle through the park. The woman is holding an umbrella 
    while walking. A child is playing with a dog in the grass. A girl is wearing 
    a red dress and standing near a tree. A boy is sitting on a bench reading a book.
    """

    # Generate summary
    summary = separate_loader.summarize(test_text)
    print(f"Original text: {test_text.strip()}")
    print(f"Pegasus Separate Loader Summary: {summary}")
    print()


@pytest.mark.skipif(not SENTENCEPIECE_AVAILABLE, reason="SentencePiece not available")
@pytest.mark.skip(reason="PyTorch meta tensor issue - needs investigation")
def test_pegasus_custom_config():
    """Test the Pegasus custom config wrapper."""
    print("=" * 50)
    print("Testing Pegasus Custom Config")
    print("=" * 50)

    # Initialize custom config wrapper
    custom_config = PegasusCustomConfig("google/pegasus-xsum")

    # Load with custom configuration
    print("Loading model with custom configuration...")
    custom_config.load_with_config(
        config_kwargs={
            "max_position_embeddings": 1024,
            "num_attention_heads": 16,
            "num_encoder_layers": 12,
            "num_decoder_layers": 12,
        },
        model_kwargs={"low_cpu_mem_usage": True},
    )

    # Test text
    test_text = """
    The man is riding a bicycle through the park. The woman is holding an umbrella 
    while walking. A child is playing with a dog in the grass. A girl is wearing 
    a red dress and standing near a tree. A boy is sitting on a bench reading a book.
    """

    # Generate summary with custom generation parameters
    summary = custom_config.summarize(
        test_text,
        max_length=100,
        min_length=15,
        length_penalty=1.5,
        num_beams=6,
        temperature=0.8,
    )
    print(f"Original text: {test_text.strip()}")
    print(f"Pegasus Custom Config Summary: {summary}")
    print()


@pytest.mark.skipif(not SENTENCEPIECE_AVAILABLE, reason="SentencePiece not available")
def test_batch_summarization():
    """Test batch summarization functionality."""
    print("=" * 50)
    print("Testing Batch Summarization")
    print("=" * 50)

    # Initialize T5 wrapper for batch processing
    t5_wrapper = T5SummarizationWrapper("google-t5/t5-base")

    # Test texts
    test_texts = [
        "The man is riding a bicycle through the park. The woman is holding an umbrella while walking.",
        "A child is playing with a dog in the grass. A girl is wearing a red dress and standing near a tree.",
        "A boy is sitting on a bench reading a book. The weather is sunny and pleasant.",
    ]

    # Generate batch summaries
    summaries = t5_wrapper.summarize_batch(test_texts)

    for i, (text, summary) in enumerate(zip(test_texts, summaries)):
        print(f"Text {i+1}: {text}")
        print(f"Summary {i+1}: {summary}")
        print()


def main():
    """Run all tests."""
    print("Starting Summarization Wrapper Tests")
    print("=" * 60)

    try:
        # Test T5 wrapper
        test_t5_wrapper()

        # Test Pegasus wrapper
        test_pegasus_wrapper()

        # Test separate loader
        test_pegasus_separate_loader()

        # Test custom config
        test_pegasus_custom_config()

        # Test batch summarization
        test_batch_summarization()

        print("=" * 60)
        print("All tests completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
