# Language Module

This module provides natural language processing capabilities for scene graph analysis, including summarization, conversation, and text generation.

## Features

### LLM Wrappers
- **Gemma 3 270M Support**: Default model for scene graph conversations
- **Extensible Architecture**: Easy to add new language models
- **Memory Efficient**: Optional quantization support for large models

### Conversation System
- **Scene Graph Context**: Natural language conversations about video analysis
- **Streamlit Integration**: Ready-to-use chat interface
- **Context Management**: Maintains conversation history and scene context

### Summarization
- **Multiple Models**: T5, Pegasus, and custom configurations
- **Scene Graph Linearization**: Converts detection results to natural language
- **Flexible Output**: Various summarization modes and parameters

## Quick Start

### Basic Usage

```python
from m3sgg.language.conversation import SceneGraphChatInterface

# Create chat interface
chat = SceneGraphChatInterface(
    model_name="google/gemma-3-270m",
    model_type="gemma"
)

# Set scene graph context
chat.set_scene_graph_context(scene_graph_data)

# Get response
response = chat.get_chat_response("What objects do you see?")
```

### Streamlit Integration

```python
import streamlit as st
from m3sgg.language.conversation import SceneGraphChatInterface

# In your Streamlit app
if "chat_interface" not in st.session_state:
    st.session_state.chat_interface = SceneGraphChatInterface()

# Set context when you have scene graph results
if "results" in st.session_state:
    st.session_state.chat_interface.set_scene_graph_context(
        st.session_state["results"]
    )

# Render chat interface
st.session_state.chat_interface.render_chat_interface()
```

### Direct LLM Usage

```python
from m3sgg.language.language modeling.llm import create_conversation_manager

# Create conversation manager
conv_manager = create_conversation_manager(
    model_name="google/gemma-3-270m",
    model_type="gemma"
)

# Set scene graph context
conv_manager.set_scene_graph(scene_graph_data)

# Get response
response = conv_manager.get_response("Tell me about this scene")
```

## Model Configuration

### Gemma 3 270M (Default)
- **Model**: `google/gemma-3-270m`
- **Parameters**: 270M
- **Context Length**: 32K tokens
- **Multimodal**: Text and image input support
- **Languages**: 140+ languages supported

### Custom Model Configuration

```python
# Custom model with specific parameters
chat = SceneGraphChatInterface(
    model_name="your-custom-model",
    model_type="gemma",
    max_context_length=20,
    temperature=0.8
)
```

## Dependencies

Install the required dependencies:

```bash
pip install -r src/m3sgg/language/requirements.txt
```

### Core Dependencies
- `transformers>=4.30.0`: Hugging Face transformers library
- `torch>=2.0.0`: PyTorch for model execution
- `streamlit>=1.28.0`: Web interface framework

### Optional Dependencies
- `accelerate>=0.20.0`: Model optimization
- `bitsandbytes>=0.39.0`: Quantization support
- `flash-attn>=2.0.0`: Flash attention for faster inference

## Examples

### Example 1: Basic Chat
```python
from m3sgg.language.conversation import SceneGraphChatInterface

# Initialize chat
chat = SceneGraphChatInterface()

# Example scene graph data
scene_data = {
    "frame_objects": [
        [{"object_name": "person", "confidence": 0.95}],
        [{"object_name": "person", "confidence": 0.92}, {"object_name": "chair", "confidence": 0.88}]
    ],
    "frame_relationships": [
        [],
        [{"subject_class": "person", "object_class": "chair", "predicate": "sitting_on", "confidence": 0.87}]
    ]
}

# Set context and chat
chat.set_scene_graph_context(scene_data)
response = chat.get_chat_response("What is the person doing?")
print(response)
```

### Example 2: Summarization
```python
from m3sgg.language.summarization.summarize import linearize_triples, summarize_sentences

# Example scene graph triples
triples = [
    ("person", "sitting_on", "chair"),
    ("person", "looking_at", "laptop"),
    ("person", "touching", "keyboard")
]

# Convert to natural language
sentences = linearize_triples(triples)

# Summarize
summary = summarize_sentences(sentences, model_type="t5")
print(summary)
```

## Architecture

### Core Components

1. **BaseLLMWrapper**: Abstract base class for all language models
2. **GemmaLLMWrapper**: Specific implementation for Gemma models
3. **ConversationManager**: Handles chat dialogue and context
4. **SceneGraphFormatter**: Converts scene graphs to natural language
5. **SceneGraphChatInterface**: High-level Streamlit integration

### Data Flow

```
Scene Graph Data → SceneGraphFormatter → Natural Language Description
                                                      ↓
User Question → ConversationManager → LLM Wrapper → Response
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure you have sufficient memory (4GB+ recommended)
   - Try using CPU if CUDA is not available
   - Check internet connection for model download

2. **Import Errors**
   - Install required dependencies: `pip install -r requirements.txt`
   - Ensure Python path includes the project root

3. **Memory Issues**
   - Use quantization: `BitsAndBytesConfig` in model loading
   - Reduce `max_context_length` parameter
   - Use smaller models if available

### Performance Tips

1. **Memory Optimization**
   - Use 4-bit quantization for large models
   - Set appropriate `max_length` parameters
   - Clear conversation history periodically

2. **Speed Optimization**
   - Use GPU when available
   - Enable flash attention if supported
   - Batch multiple requests when possible

## Contributing

To add support for new language models:

1. Create a new wrapper class inheriting from `BaseLLMWrapper`
2. Implement required abstract methods
3. Add model type to factory function
4. Update documentation and examples

## License

This module is part of the M3SGG project and follows the same license terms.
