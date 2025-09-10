"""
Example script demonstrating LLM-based chat for scene graph analysis.

This script shows how to use the LLM wrapper system to have natural language
conversations about scene graph analysis results.

Usage:
    python examples/llm_chat_example.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from m3sgg.language.language_modeling.llm import (
    create_conversation_manager,
    SceneGraphFormatter,
)


def main():
    """Main example function demonstrating LLM chat capabilities."""
    print("Scene Graph LLM Chat Example")
    print("=" * 50)

    # Create conversation manager with Gemma 3 270M
    print("Initializing conversation manager...")
    try:
        conv_manager = create_conversation_manager(
            model_name="google/gemma-3-270m", model_type="gemma"
        )
        print("✓ Conversation manager initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize conversation manager: {e}")
        print("Make sure you have the required dependencies installed:")
        print("  pip install transformers torch")
        return

    # Example scene graph data (similar to what would come from video analysis)
    example_scene_graph = {
        "total_frames": 30,
        "processed_frames": 30,
        "detections": [3, 4, 3, 5, 4, 3, 4, 5, 4, 3],
        "relationships": [2, 3, 2, 4, 3, 2, 3, 4, 3, 2],
        "confidences": [0.85, 0.92, 0.78, 0.88, 0.91, 0.83, 0.89, 0.87, 0.85, 0.82],
        "frame_objects": [
            [{"object_name": "person", "confidence": 0.95}],
            [
                {"object_name": "person", "confidence": 0.92},
                {"object_name": "chair", "confidence": 0.88},
            ],
            [{"object_name": "person", "confidence": 0.89}],
            [
                {"object_name": "person", "confidence": 0.94},
                {"object_name": "table", "confidence": 0.85},
            ],
            [{"object_name": "person", "confidence": 0.91}],
            [{"object_name": "person", "confidence": 0.87}],
            [
                {"object_name": "person", "confidence": 0.93},
                {"object_name": "laptop", "confidence": 0.79},
            ],
            [
                {"object_name": "person", "confidence": 0.90},
                {"object_name": "book", "confidence": 0.82},
            ],
            [{"object_name": "person", "confidence": 0.88}],
            [{"object_name": "person", "confidence": 0.86}],
        ],
        "frame_relationships": [
            [],
            [
                {
                    "subject_class": "person",
                    "object_class": "chair",
                    "predicate": "sitting_on",
                    "confidence": 0.87,
                }
            ],
            [],
            [
                {
                    "subject_class": "person",
                    "object_class": "table",
                    "predicate": "touching",
                    "confidence": 0.82,
                }
            ],
            [],
            [],
            [
                {
                    "subject_class": "person",
                    "object_class": "laptop",
                    "predicate": "using",
                    "confidence": 0.85,
                }
            ],
            [
                {
                    "subject_class": "person",
                    "object_class": "book",
                    "predicate": "reading",
                    "confidence": 0.78,
                }
            ],
            [],
            [],
        ],
    }

    # Set scene graph context
    print("\nSetting scene graph context...")
    conv_manager.set_scene_graph(example_scene_graph)
    print("✓ Scene graph context set")

    # Display scene graph information
    formatter = SceneGraphFormatter()
    scene_description = formatter.format_scene_graph(example_scene_graph)
    print(f"\nScene Description:\n{scene_description}")

    # Example conversation
    print("\n" + "=" * 50)
    print("Starting conversation...")
    print("=" * 50)

    example_questions = [
        "What objects do you see in this video?",
        "What relationships are happening between the objects?",
        "Can you summarize what's happening in this scene?",
        "What activities is the person doing?",
        "How confident are the detections overall?",
    ]

    for i, question in enumerate(example_questions, 1):
        print(f"\n[Question {i}]")
        print(f"User: {question}")

        try:
            response = conv_manager.get_response(question)
            print(f"Assistant: {response}")
        except Exception as e:
            print(f"Assistant: I apologize, but I encountered an error: {e}")

        print("-" * 30)

    # Interactive mode
    print("\n" + "=" * 50)
    print("Interactive Mode (type 'quit' to exit)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            if user_input:
                response = conv_manager.get_response(user_input)
                print(f"Assistant: {response}")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nConversation ended. Thank you for using the scene graph chat system!")


if __name__ == "__main__":
    main()
