# M3SGG Examples

This directory contains comprehensive example notebooks demonstrating video scene graph generation and summarization capabilities.

## Notebooks Overview

### 1. Basic Video Scene Graph Generation
**File**: `01_basic_video_scene_graph_generation.ipynb`

Demonstrates the fundamental workflow for generating scene graphs from video data:
- Video loading and frame extraction
- Object detection and classification
- Scene graph generation using STTran
- Visualization of results
- Export functionality

**Key Features**:
- Complete pipeline from video to scene graph
- Error handling and troubleshooting
- Configurable parameters
- Results analysis and export

### 2. Scene Graph to Text Summarization
**File**: `02_scene_graph_to_text_summarization.ipynb`

Shows how to convert scene graphs into natural language descriptions and summaries:
- Scene graph triple linearization
- Multiple summarization models (T5, Pegasus)
- Advanced prompting strategies
- Batch processing and comparison
- Performance metrics

**Key Features**:
- Multiple summarization approaches
- Custom configuration options
- Error handling and fallbacks
- Comprehensive analysis

### 3. End-to-End Video to Summary Pipeline
**File**: `03_end_to_end_video_to_summary.ipynb`

Complete pipeline combining video processing, scene graph generation, and text summarization:
- Integrated `VideoToSummaryPipeline` class
- Modular design with error handling
- Combined visualization (scene graphs + text)
- Export and analysis capabilities

**Key Features**:
- Single-class solution for complete workflow
- Robust error handling
- Visual and text output
- Easy configuration and customization

### 4. Advanced VLM-based Scene Graph Generation
**File**: `04_advanced_vlm_scene_graph_generation.ipynb`

Advanced scene graph generation using Vision-Language Models with reasoning capabilities:
- VLM integration and configuration
- Few-shot learning examples
- Chain-of-thought and tree-of-thought reasoning
- Multiple prompting strategies
- Performance evaluation

**Key Features**:
- Advanced reasoning capabilities
- Few-shot learning support
- Multiple prompting strategies
- Comprehensive evaluation

### 5. Model Comparison and Evaluation
**File**: `05_model_comparison_and_evaluation.ipynb`

Comprehensive evaluation framework for comparing different scene graph generation models:
- Standardized evaluation metrics
- Model comparison and ranking
- Visualization of results
- Performance analysis

**Key Features**:
- Flexible evaluation framework
- Multiple metrics and rankings
- Comparative visualizations
- Export functionality

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install torch torchvision opencv-python matplotlib networkx transformers
```

### Running the Notebooks

1. **Start with the basic notebook** (`01_basic_video_scene_graph_generation.ipynb`) to understand the fundamental concepts
2. **Explore summarization** (`02_scene_graph_to_text_summarization.ipynb`) to see text generation capabilities
3. **Try the end-to-end pipeline** (`03_end_to_end_video_to_summary.ipynb`) for complete workflows
4. **Experiment with advanced VLM** (`04_advanced_vlm_scene_graph_generation.ipynb`) for cutting-edge approaches
5. **Compare models** (`05_model_comparison_and_evaluation.ipynb`) to evaluate performance

### Data Requirements

- **Action Genome Dataset**: Required for video scene graph generation
- **Model Checkpoints**: Pre-trained models for inference
- **Dependencies**: See individual notebook prerequisites

## Notebook Structure

Each notebook follows a consistent structure:

1. **Introduction**: Overview and prerequisites
2. **Setup**: Configuration and data loading
3. **Core Functionality**: Main implementation
4. **Visualization**: Results display
5. **Analysis**: Performance evaluation
6. **Export**: Results saving
7. **Summary**: Key takeaways and next steps

## Key Concepts

### Scene Graph Generation
- **Objects**: Detected entities in video frames
- **Relationships**: Spatial, temporal, and semantic connections
- **Triples**: Subject-predicate-object representations

### Text Summarization
- **Linearization**: Converting triples to natural language
- **Summarization**: Generating concise descriptions
- **Models**: T5, Pegasus, and other transformer-based models

### Evaluation Metrics
- **Accuracy**: Correctness of predictions
- **Inference Time**: Speed of processing
- **Compression Ratio**: Summary length vs. original text
- **Success Rate**: Percentage of successful operations

## Troubleshooting

### Common Issues

1. **Data Not Found**: Ensure Action Genome dataset is properly downloaded and extracted
2. **Model Loading Errors**: Check model availability and compatibility
3. **CUDA Issues**: Verify GPU availability and model compatibility
4. **Memory Issues**: Reduce batch size or use smaller models
5. **Import Errors**: Verify m3sgg package installation

### Getting Help

- Check individual notebook troubleshooting sections
- Review error messages and logs
- Ensure all dependencies are installed
- Verify data paths and model availability

## Contributing

To add new examples:

1. Follow the existing notebook structure
2. Include comprehensive documentation
3. Add error handling and troubleshooting
4. Provide clear explanations and comments
5. Test with different datasets and configurations

## License

These examples are part of the M3SGG project and follow the same license terms.
