#!/usr/bin/env python3
"""
Test script for GUI summarization integration.
Tests the summarization wrapper classes and their integration with the GUI.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.nlp_module.summarization_wrapper import (
    PegasusCustomConfig,
    PegasusSeparateLoader,
    PegasusSummarizationWrapper,
    T5SummarizationWrapper,
)


def test_summarization_wrappers():
    """Test all summarization wrapper classes"""
    print("=" * 60)
    print("Testing Summarization Wrapper Classes")
    print("=" * 60)

    # Test text
    test_text = """
    The scene contains multiple objects including a person, a bicycle, and various background elements. 
    The person is riding the bicycle through a park setting. There are spatial relationships between 
    the objects and attention patterns that indicate the person is focused on the bicycle.
    """

    # Test T5 Summarization
    print("\n1. Testing T5 Summarization:")
    try:
        t5_wrapper = T5SummarizationWrapper("google-t5/t5-base")
        summary = t5_wrapper.summarize(test_text, max_length=100)
        print(f"T5 Summary: {summary}")
    except Exception as e:
        print(f"T5 Error: {str(e)}")

    # Test Pegasus Summarization
    print("\n2. Testing Pegasus Summarization:")
    try:
        pegasus_wrapper = PegasusSummarizationWrapper("google/pegasus-xsum")
        summary = pegasus_wrapper.summarize(test_text, max_length=100)
        print(f"Pegasus Summary: {summary}")
    except Exception as e:
        print(f"Pegasus Error: {str(e)}")

    # Test Separate Loader
    print("\n3. Testing Pegasus Separate Loader:")
    try:
        separate_loader = PegasusSeparateLoader("google/pegasus-xsum")
        separate_loader.load_tokenizer()
        separate_loader.load_model()
        summary = separate_loader.summarize(test_text, max_length=100)
        print(f"Separate Loader Summary: {summary}")
    except Exception as e:
        print(f"Separate Loader Error: {str(e)}")

    # Test Custom Config
    print("\n4. Testing Pegasus Custom Config:")
    try:
        custom_config = PegasusCustomConfig("google/pegasus-xsum")
        custom_config.load_with_config()
        summary = custom_config.summarize(test_text, max_length=100)
        print(f"Custom Config Summary: {summary}")
    except Exception as e:
        print(f"Custom Config Error: {str(e)}")


def test_model_mapping():
    """Test the model name mapping used in the GUI"""
    print("\n" + "=" * 60)
    print("Testing Model Name Mapping")
    print("=" * 60)

    # Model mappings used in GUI
    model_mappings = {
        "T5 Base": "google-t5/t5-base",
        "T5 Large": "google-t5/t5-large",
        "Pegasus XSum": "google/pegasus-xsum",
        "Pegasus CNN/DailyMail": "google/pegasus-cnn_dailymail",
        "Pegasus Newsroom": "google/pegasus-newsroom",
        "Pegasus Multi-News": "google/pegasus-multi_news",
    }

    test_text = "The scene shows a person interacting with objects in the environment."

    for display_name, model_name in model_mappings.items():
        print(f"\nTesting {display_name}:")
        try:
            if display_name.startswith("T5"):
                if "Large" in display_name:
                    wrapper = T5SummarizationWrapper("google-t5/t5-large")
                else:
                    wrapper = T5SummarizationWrapper("google-t5/t5-base")
            elif display_name.startswith("Pegasus"):
                wrapper = PegasusSummarizationWrapper(model_name)
            else:
                continue

            summary = wrapper.summarize(test_text, max_length=80)
            print(f"  Model: {model_name}")
            print(f"  Summary: {summary}")

        except Exception as e:
            print(f"  Error: {str(e)}")


def test_gui_integration():
    """Test the GUI integration logic"""
    print("\n" + "=" * 60)
    print("Testing GUI Integration Logic")
    print("=" * 60)

    # Simulate the GUI model selection logic
    def simulate_gui_model_selection(model_name):
        """Simulate the GUI model selection logic"""
        try:
            if model_name.startswith("T5"):
                if "Large" in model_name:
                    return T5SummarizationWrapper("google-t5/t5-large")
                else:
                    return T5SummarizationWrapper("google-t5/t5-base")
            elif model_name.startswith("Pegasus"):
                if "CNN/DailyMail" in model_name:
                    return PegasusSummarizationWrapper("google/pegasus-cnn_dailymail")
                elif "Newsroom" in model_name:
                    return PegasusSummarizationWrapper("google/pegasus-newsroom")
                elif "Multi-News" in model_name:
                    return PegasusSummarizationWrapper("google/pegasus-multi_news")
                else:
                    return PegasusSummarizationWrapper("google/pegasus-xsum")
            return None
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
            return None

    # Test all GUI model options
    gui_model_names = [
        "T5 Base",
        "T5 Large",
        "Pegasus XSum",
        "Pegasus CNN/DailyMail",
        "Pegasus Newsroom",
        "Pegasus Multi-News",
    ]

    test_text = "A person is performing actions with various objects in the scene."

    for model_name in gui_model_names:
        print(f"\nTesting GUI model: {model_name}")
        try:
            wrapper = simulate_gui_model_selection(model_name)
            if wrapper:
                summary = wrapper.summarize(test_text, max_length=60)
                print(f"  Summary: {summary}")
            else:
                print("  Failed to load model")
        except Exception as e:
            print(f"  Error: {str(e)}")


def main():
    """Run all tests"""
    print("Starting GUI Summarization Integration Tests")
    print("=" * 80)

    try:
        # Test basic wrapper functionality
        test_summarization_wrappers()

        # Test model name mapping
        test_model_mapping()

        # Test GUI integration
        test_gui_integration()

        print("\n" + "=" * 80)
        print("All tests completed!")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
