"""
Language Model Wrappers for Scene Graph Conversation

This module provides wrappers for various language models to enable natural language
conversation about scene graphs generated from video analysis. Supports multiple
model architectures with a unified interface.

Classes:
    BaseLLMWrapper: Abstract base class for all LLM wrappers
    GemmaLLMWrapper: Wrapper for Google's Gemma models
    ConversationManager: Manages chat dialogue with scene graph context
    SceneGraphFormatter: Converts scene graph data to natural language
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import re
from m3sgg.language.summarization.summarize import linearize_triples

try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logging.warning(f"Transformers not available: {e}")

logger = logging.getLogger(__name__)


class BaseLLMWrapper(ABC):
    """Abstract base class for language model wrappers.
    
    Provides a unified interface for different language models to enable
    natural language conversation about scene graphs and video analysis.
    
    :param ABC: Abstract Base Class
    :type ABC: class
    """

    def __init__(self, model_name: str, device: Optional[str] = None, 
                 max_length: int = 512, temperature: float = 0.7, **kwargs):
        """Initialize the LLM wrapper.
        
        :param model_name: Name or path of the pretrained model
        :type model_name: str
        :param device: Device to load model on ('cpu', 'cuda', etc.), defaults to None
        :type device: str, optional
        :param max_length: Maximum sequence length for generation, defaults to 512
        :type max_length: int, optional
        :param temperature: Sampling temperature for generation, defaults to 0.7
        :type temperature: float, optional
        :param kwargs: Additional parameters (ignored for base class)
        :type kwargs: dict
        :return: None
        :rtype: None
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.temperature = temperature
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the tokenizer and model.
        
        Abstract method that must be implemented by subclasses to load
        the specific tokenizer and model for the language model.
        
        :return: None
        :rtype: None
        """
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the model.
        
        Abstract method that must be implemented by subclasses to generate
        text responses from the language model.
        
        :param prompt: Input prompt for generation
        :type prompt: str
        :param kwargs: Additional generation parameters
        :type kwargs: dict
        :return: Generated response text
        :rtype: str
        """
        pass
    
    def chat_about_scene_graph(self, scene_graph_data: Dict[str, Any], 
                              user_question: str, **kwargs) -> str:
        """Generate a response about scene graph data.
        
        :param scene_graph_data: Scene graph detection results
        :type scene_graph_data: Dict[str, Any]
        :param user_question: User's question about the scene graph
        :type user_question: str
        :param kwargs: Additional generation parameters
        :type kwargs: dict
        :return: Generated response about the scene graph
        :rtype: str
        """
        if not self.is_loaded:
            self.load_model()
            
        # Format scene graph data into natural language
        scene_description = self._format_scene_graph(scene_graph_data)
        
        # Create conversation prompt
        prompt = self._create_conversation_prompt(scene_description, user_question)
        
        # Generate response
        response = self.generate_response(prompt, **kwargs)
        
        return response
    
    def _format_scene_graph(self, scene_graph_data: Dict[str, Any]) -> str:
        """Format scene graph data into natural language description.
        
        :param scene_graph_data: Scene graph detection results
        :type scene_graph_data: Dict[str, Any]
        :return: Natural language description of the scene
        :rtype: str
        """
        triples = scene_graph_data.get("triples", [])
        if triples:
            try:
                sentences = linearize_triples(triples, mode="flat")
                return " ".join(sentences) if sentences else "No objects or relationships detected in the scene."
            except Exception:
                pass
        # Extract objects and relationships
        objects = scene_graph_data.get("frame_objects", [])
        relationships = scene_graph_data.get("frame_relationships", [])
        
        description_parts = []
        
        # Describe detected objects
        if objects:
            object_names = [obj.get("object_name", "unknown") for obj in objects]
            unique_objects = list(set(object_names))
            if unique_objects:
                description_parts.append(f"Detected objects: {', '.join(unique_objects)}")
        
        # Describe relationships
        if relationships:
            rel_descriptions = []
            for rel in relationships:
                subject = rel.get("subject_class", "person")
                obj = rel.get("object_class", "object")
                predicate = rel.get("predicate", "interacts_with")
                rel_descriptions.append(f"{subject} {predicate} {obj}")
            
            if rel_descriptions:
                description_parts.append(f"Relationships: {'; '.join(rel_descriptions)}")
        
        # Add frame statistics
        total_frames = scene_graph_data.get("processed_frames", 0)
        avg_objects = scene_graph_data.get("detections", [])
        if avg_objects:
            avg_count = sum(avg_objects) / len(avg_objects)
            description_parts.append(f"Average {avg_count:.1f} objects per frame across {total_frames} frames")
        
        return ". ".join(description_parts) if description_parts else "No scene graph data available"
    
    def _create_conversation_prompt(self, scene_description: str, user_question: str) -> str:
        """Create a conversation prompt for the model.
        
        :param scene_description: Natural language description of the scene
        :type scene_description: str
        :param user_question: User's question
        :type user_question: str
        :return: Formatted conversation prompt
        :rtype: str
        """
        system_prompt = """You are an AI assistant specialized in analyzing video scene graphs. 
You help users understand what's happening in videos by interpreting object detections, 
relationships, and activities. Be helpful, accurate, and conversational."""
        
        prompt = f"""{system_prompt}

Scene Analysis:
{scene_description}

User Question: {user_question}

Assistant:"""
        
        return prompt


class GemmaLLMWrapper(BaseLLMWrapper):
    """Wrapper for Google's Gemma language models.
    
    Provides interface for Gemma models with optimized configuration
    for scene graph conversation tasks.
    
    :param BaseLLMWrapper: Base class for LLM wrappers
    :type BaseLLMWrapper: class
    """
    
    def __init__(self, model_name: str = "google/gemma-3-270m", 
                 device: Optional[str] = None, **kwargs):
        """Initialize Gemma wrapper.
        
        :param model_name: Name of the Gemma model, defaults to "google/gemma-3-270m"
        :type model_name: str, optional
        :param device: Device to load model on, defaults to None
        :type device: str, optional
        :param kwargs: Additional initialization parameters
        :type kwargs: dict
        :return: None
        :rtype: None
        """
        super().__init__(model_name, device, **kwargs)
        
        # Gemma-specific configuration
        self.generation_config = {
            "max_new_tokens": 256,
            "temperature": self.temperature,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": None,  # Will be set after tokenizer loading
            "eos_token_id": None,  # Will be set after tokenizer loading
        }
    
    def load_model(self) -> None:
        """Load Gemma tokenizer and model.
        
        :return: None
        :rtype: None
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for Gemma models")
        
        try:
            logger.info(f"Loading Gemma model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate configuration
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            # Add quantization for memory efficiency if needed
            if self.device == "cuda":
                try:
                    # Try to use 4-bit quantization for memory efficiency
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                except Exception as e:
                    logger.warning(f"Could not apply quantization: {e}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            # Update generation config with tokenizer info
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            self.generation_config["eos_token_id"] = self.tokenizer.eos_token_id
            
            self.is_loaded = True
            logger.info("Gemma model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}")
            raise
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Gemma model.
        
        :param prompt: Input prompt for generation
        :type prompt: str
        :param kwargs: Additional generation parameters
        :type kwargs: dict
        :return: Generated response text
        :rtype: str
        """
        if not self.is_loaded:
            self.load_model()
        
        # Merge user kwargs with default config
        generation_config = self.generation_config.copy()
        generation_config.update(kwargs)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.max_length
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Clean up response
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response.
        
        :param response: Raw generated response
        :type response: str
        :return: Cleaned response text
        :rtype: str
        """
        # Remove extra whitespace
        response = re.sub(r'\s+', ' ', response.strip())
        
        # Remove common artifacts
        response = re.sub(r'^(Assistant:|User:|Human:)\s*', '', response, flags=re.IGNORECASE)
        
        # Ensure response ends properly
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response


class ConversationManager:
    """Manages chat dialogue with scene graph context.
    
    Handles conversation state, context management, and response generation
    for scene graph-based chat interactions.
    
    :param llm_wrapper: LLM wrapper instance for response generation
    :type llm_wrapper: BaseLLMWrapper
    :param max_context_length: Maximum number of conversation turns to keep
    :type max_context_length: int, optional
    """
    
    def __init__(self, llm_wrapper: BaseLLMWrapper, max_context_length: int = 10):
        """Initialize conversation manager.
        
        :param llm_wrapper: LLM wrapper instance
        :type llm_wrapper: BaseLLMWrapper
        :param max_context_length: Maximum conversation context length, defaults to 10
        :type max_context_length: int, optional
        :return: None
        :rtype: None
        """
        self.llm_wrapper = llm_wrapper
        self.max_context_length = max_context_length
        self.conversation_history = []
        self.current_scene_graph = None
        
    def set_scene_graph(self, scene_graph_data: Dict[str, Any]) -> None:
        """Set the current scene graph context.
        
        :param scene_graph_data: Scene graph detection results
        :type scene_graph_data: Dict[str, Any]
        :return: None
        :rtype: None
        """
        self.current_scene_graph = scene_graph_data
        logger.info("Scene graph context updated")
    
    def add_message(self, user_message: str, is_user: bool = True) -> None:
        """Add a message to conversation history.
        
        :param user_message: Message content
        :type user_message: str
        :param is_user: Whether the message is from user, defaults to True
        :type is_user: bool, optional
        :return: None
        :rtype: None
        """
        self.conversation_history.append({
            "message": user_message,
            "is_user": is_user,
            "timestamp": self._get_timestamp()
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_context_length * 2:
            self.conversation_history = self.conversation_history[-self.max_context_length * 2:]
    
    def get_response(self, user_question: str, **kwargs) -> str:
        """Get response to user question with scene graph context.
        
        :param user_question: User's question
        :type user_question: str
        :param kwargs: Additional generation parameters
        :type kwargs: dict
        :return: Generated response
        :rtype: str
        """
        # Add user message to history
        self.add_message(user_question, is_user=True)
        
        # Generate response
        if self.current_scene_graph:
            response = self.llm_wrapper.chat_about_scene_graph(
                self.current_scene_graph, 
                user_question, 
                **kwargs
            )
        else:
            response = self.llm_wrapper.generate_response(
                f"User: {user_question}\nAssistant:",
                **kwargs
            )
        
        # Add response to history
        self.add_message(response, is_user=False)
        
        return response
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get current conversation history.
        
        :return: List of conversation messages
        :rtype: List[Dict[str, Any]]
        """
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history.
        
        :return: None
        :rtype: None
        """
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string.
        
        :return: Formatted timestamp
        :rtype: str
        """
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")


class SceneGraphFormatter:
    """Formats scene graph data into natural language descriptions.
    
    Converts detection results, relationships, and temporal information
    into human-readable text for LLM consumption.
    
    :param object_classes: List of object class names
    :type object_classes: List[str], optional
    :param relationship_classes: List of relationship class names
    :type relationship_classes: List[str], optional
    """
    
    def __init__(self, object_classes: Optional[List[str]] = None,
                 relationship_classes: Optional[List[str]] = None):
        """Initialize scene graph formatter.
        
        :param object_classes: List of object class names, defaults to None
        :type object_classes: List[str], optional
        :param relationship_classes: List of relationship class names, defaults to None
        :type relationship_classes: List[str], optional
        :return: None
        :rtype: None
        """
        self.object_classes = object_classes or []
        self.relationship_classes = relationship_classes or []
        
        # Default relationship patterns
        self.relationship_patterns = {
            "looking_at": "is looking at",
            "not_looking_at": "is not looking at",
            "in_front_of": "is in front of",
            "behind": "is behind",
            "on_the_side_of": "is on the side of",
            "covered_by": "is covered by",
            "drinking_from": "is drinking from",
            "have_it_on_the_back": "has it on the back",
            "leaning_on": "is leaning on",
            "lying_on": "is lying on",
            "not_contacting": "is not contacting",
            "other_relationship": "has some relationship with",
            "sitting_on": "is sitting on",
            "standing_on": "is standing on",
            "writing_on": "is writing on",
        }
    
    def format_scene_graph(self, scene_graph_data: Dict[str, Any]) -> str:
        """Format scene graph data into natural language.
        
        :param scene_graph_data: Scene graph detection results
        :type scene_graph_data: Dict[str, Any]
        :return: Natural language description
        :rtype: str
        """
        triples = scene_graph_data.get("triples", [])
        if triples:
            try:
                sentences = linearize_triples(triples, mode="flat")
                return " ".join(sentences) if sentences else "No scene graph data available"
            except Exception:
                pass
        descriptions = []
        
        # Add video metadata
        if "total_frames" in scene_graph_data:
            total_frames = scene_graph_data["total_frames"]
            processed_frames = scene_graph_data.get("processed_frames", 0)
            descriptions.append(f"Video has {total_frames} total frames, analyzed {processed_frames} frames")
        
        # Add frame-by-frame analysis
        frame_objects = scene_graph_data.get("frame_objects", [])
        frame_relationships = scene_graph_data.get("frame_relationships", [])
        
        if frame_objects:
            descriptions.append(self._format_frame_objects(frame_objects))
        
        if frame_relationships:
            descriptions.append(self._format_frame_relationships(frame_relationships))
        
        # Add temporal analysis
        if "detections" in scene_graph_data:
            descriptions.append(self._format_temporal_analysis(scene_graph_data))
        
        return ". ".join(descriptions) if descriptions else "No scene graph data available"
    
    def _format_frame_objects(self, frame_objects: List[List[Dict[str, Any]]]) -> str:
        """Format object detections across frames.
        
        :param frame_objects: List of object detections per frame
        :type frame_objects: List[List[Dict[str, Any]]]
        :return: Formatted object description
        :rtype: str
        """
        # Collect all unique objects
        all_objects = set()
        object_confidence = {}
        
        for frame_objs in frame_objects:
            for obj in frame_objs:
                obj_name = obj.get("object_name", "unknown")
                confidence = obj.get("confidence", 0.0)
                all_objects.add(obj_name)
                
                if obj_name not in object_confidence:
                    object_confidence[obj_name] = []
                object_confidence[obj_name].append(confidence)
        
        if not all_objects:
            return "No objects detected"
        
        # Calculate average confidence for each object
        obj_descriptions = []
        for obj_name in sorted(all_objects):
            confidences = object_confidence[obj_name]
            avg_confidence = sum(confidences) / len(confidences)
            presence_frames = len(confidences)
            obj_descriptions.append(f"{obj_name} (avg confidence: {avg_confidence:.2f}, present in {presence_frames} frames)")
        
        return f"Detected objects: {', '.join(obj_descriptions)}"
    
    def _format_frame_relationships(self, frame_relationships: List[List[Dict[str, Any]]]) -> str:
        """Format relationships across frames.
        
        :param frame_relationships: List of relationships per frame
        :type frame_relationships: List[List[Dict[str, Any]]]
        :return: Formatted relationship description
        :rtype: str
        """
        # Collect all relationships
        all_relationships = []
        
        for frame_rels in frame_relationships:
            for rel in frame_rels:
                all_relationships.append(rel)
        
        if not all_relationships:
            return "No relationships detected"
        
        # Group by relationship type
        rel_by_type = {}
        for rel in all_relationships:
            rel_type = rel.get("predicate", "interacts_with")
            if rel_type not in rel_by_type:
                rel_by_type[rel_type] = []
            rel_by_type[rel_type].append(rel)
        
        # Format relationship descriptions
        rel_descriptions = []
        for rel_type, rels in rel_by_type.items():
            count = len(rels)
            avg_confidence = sum(r.get("confidence", 0.0) for r in rels) / len(rels)
            rel_descriptions.append(f"{rel_type} ({count} instances, avg confidence: {avg_confidence:.2f})")
        
        return f"Relationships: {', '.join(rel_descriptions)}"
    
    def _format_temporal_analysis(self, scene_graph_data: Dict[str, Any]) -> str:
        """Format temporal analysis of the scene.
        
        :param scene_graph_data: Scene graph data with temporal information
        :type scene_graph_data: Dict[str, Any]
        :return: Formatted temporal description
        :rtype: str
        """
        detections = scene_graph_data.get("detections", [])
        relationships = scene_graph_data.get("relationships", [])
        confidences = scene_graph_data.get("confidences", [])
        
        if not detections:
            return "No temporal analysis available"
        
        avg_objects = sum(detections) / len(detections)
        avg_relationships = sum(relationships) / len(relationships) if relationships else 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        temporal_desc = f"Temporal analysis: average {avg_objects:.1f} objects per frame"
        if avg_relationships > 0:
            temporal_desc += f", {avg_relationships:.1f} relationships per frame"
        if avg_confidence > 0:
            temporal_desc += f", average confidence {avg_confidence:.2f}"
        
        return temporal_desc


def create_llm_wrapper(model_name: str = "google/gemma-3-270m", 
                      model_type: str = "gemma", **kwargs) -> BaseLLMWrapper:
    """Factory function to create LLM wrapper instances.
    
    :param model_name: Name of the model to load, defaults to "google/gemma-3-270m"
    :type model_name: str, optional
    :param model_type: Type of model wrapper to create, defaults to "gemma"
    :type model_type: str, optional
    :param kwargs: Additional initialization parameters
    :type kwargs: dict
    :return: Initialized LLM wrapper instance
    :rtype: BaseLLMWrapper
    """
    if model_type.lower() == "gemma":
        return GemmaLLMWrapper(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_conversation_manager(model_name: str = "google/gemma-3-270m",
                               model_type: str = "gemma", max_context_length: int = 10, **kwargs) -> ConversationManager:
    """Create a conversation manager with LLM wrapper.
    
    :param model_name: Name of the model to load, defaults to "google/gemma-3-270m"
    :type model_name: str, optional
    :param model_type: Type of model wrapper to create, defaults to "gemma"
    :type model_type: str, optional
    :param max_context_length: Maximum conversation context length, defaults to 10
    :type max_context_length: int, optional
    :param kwargs: Additional initialization parameters
    :type kwargs: dict
    :return: Initialized conversation manager
    :rtype: ConversationManager
    """
    llm_wrapper = create_llm_wrapper(model_name, model_type, **kwargs)
    return ConversationManager(llm_wrapper, max_context_length)


if __name__ == "__main__":
    # Example usage
    print("Testing LLM wrapper system...")
    
    # Create conversation manager
    conv_manager = create_conversation_manager()
    
    # Example scene graph data
    example_scene_graph = {
        "total_frames": 30,
        "processed_frames": 30,
        "detections": [3, 4, 3, 5, 4],
        "relationships": [2, 3, 2, 4, 3],
        "confidences": [0.85, 0.92, 0.78, 0.88, 0.91],
        "frame_objects": [
            [{"object_name": "person", "confidence": 0.95}],
            [{"object_name": "person", "confidence": 0.92}, {"object_name": "chair", "confidence": 0.88}],
            [{"object_name": "person", "confidence": 0.89}],
            [{"object_name": "person", "confidence": 0.94}, {"object_name": "table", "confidence": 0.85}],
            [{"object_name": "person", "confidence": 0.91}]
        ],
        "frame_relationships": [
            [],
            [{"subject_class": "person", "object_class": "chair", "predicate": "sitting_on", "confidence": 0.87}],
            [],
            [{"subject_class": "person", "object_class": "table", "predicate": "touching", "confidence": 0.82}],
            []
        ]
    }
    
    # Set scene graph context
    conv_manager.set_scene_graph(example_scene_graph)
    
    # Test conversation
    questions = [
        "What objects do you see in this video?",
        "What relationships are happening?",
        "Can you summarize what's happening in this scene?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        response = conv_manager.get_response(question)
        print(f"Assistant: {response}")
