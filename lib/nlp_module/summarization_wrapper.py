from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


class BaseSummarizationWrapper(ABC):
    """
    Abstract base class for summarization model wrappers.
    Provides a unified interface for different summarization models.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the summarization wrapper.

        Args:
            model_name (str): Name of the pretrained model
            device (Optional[str]): Device to load model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Load the tokenizer and model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _prepare_input(self, text: str) -> Dict[str, torch.Tensor]:
        """Prepare input for the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _generate_summary(self, input_ids: torch.Tensor) -> str:
        """Generate summary from input. Must be implemented by subclasses."""
        pass

    def summarize(self, text: str, **kwargs) -> str:
        """
        Summarize the given text.

        Args:
            text (str): Text to summarize
            **kwargs: Additional generation parameters

        Returns:
            str: Generated summary
        """
        input_data = self._prepare_input(text)
        return self._generate_summary(input_data["input_ids"], **kwargs)

    def summarize_batch(self, texts: List[str], **kwargs) -> List[str]:
        """
        Summarize a batch of texts.

        Args:
            texts (List[str]): List of texts to summarize
            **kwargs: Additional generation parameters

        Returns:
            List[str]: List of generated summaries
        """
        summaries = []
        for text in texts:
            summary = self.summarize(text, **kwargs)
            summaries.append(summary)
        return summaries


class T5SummarizationWrapper(BaseSummarizationWrapper):
    """
    Wrapper for T5-based summarization models.
    """

    def _load_model(self):
        """Load T5 tokenizer and model."""
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_input(self, text: str) -> Dict[str, torch.Tensor]:
        """Prepare input for T5 model."""
        input_text = f"summarize: {text}"
        input_ids = self.tokenizer(
            input_text, return_tensors="pt", max_length=512, truncation=True
        )
        return {k: v.to(self.device) for k, v in input_ids.items()}

    def _generate_summary(self, input_ids: torch.Tensor, **kwargs) -> str:
        """Generate summary using T5 model."""
        default_params = {
            "max_length": 100,
            "min_length": 20,
            "length_penalty": 1.0,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
        }
        default_params.update(kwargs)

        with torch.no_grad():
            summary_ids = self.model.generate(input_ids, **default_params)

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


class PegasusSummarizationWrapper(BaseSummarizationWrapper):
    """
    Wrapper for Pegasus-based summarization models.
    """

    def _load_model(self):
        """Load Pegasus tokenizer and model."""
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_input(self, text: str) -> Dict[str, torch.Tensor]:
        """Prepare input for Pegasus model."""
        input_ids = self.tokenizer(
            text, return_tensors="pt", max_length=1024, truncation=True
        )
        return {k: v.to(self.device) for k, v in input_ids.items()}

    def _generate_summary(self, input_ids: torch.Tensor, **kwargs) -> str:
        """Generate summary using Pegasus model."""
        default_params = {
            "max_length": 128,
            "min_length": 20,
            "length_penalty": 2.0,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.2,
            "do_sample": False,  # Pegasus typically works better with beam search
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
        }
        default_params.update(kwargs)

        with torch.no_grad():
            summary_ids = self.model.generate(input_ids, **default_params)

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


class PegasusSeparateLoader:
    """
    Extension class that loads Pegasus tokenizer and model separately.
    Useful for custom loading strategies or when you need more control.
    """

    def __init__(
        self, model_name: str = "google/pegasus-xsum", device: Optional[str] = None
    ):
        """
        Initialize with separate tokenizer and model loading.

        Args:
            model_name (str): Name of the Pegasus model
            device (Optional[str]): Device to load model on
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_tokenizer(self, **kwargs) -> PegasusTokenizer:
        """
        Load the Pegasus tokenizer separately.

        Args:
            **kwargs: Additional arguments for tokenizer loading

        Returns:
            PegasusTokenizer: Loaded tokenizer
        """
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name, **kwargs)
        return self.tokenizer

    def load_model(self, **kwargs) -> PegasusForConditionalGeneration:
        """
        Load the Pegasus model separately.

        Args:
            **kwargs: Additional arguments for model loading

        Returns:
            PegasusForConditionalGeneration: Loaded model
        """
        self.model = PegasusForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        )
        self.model.to(self.device)
        self.model.eval()
        return self.model

    def is_loaded(self) -> bool:
        """Check if both tokenizer and model are loaded."""
        return self.tokenizer is not None and self.model is not None

    def summarize(self, text: str, **kwargs) -> str:
        """
        Summarize text using the separately loaded components.

        Args:
            text (str): Text to summarize
            **kwargs: Generation parameters

        Returns:
            str: Generated summary
        """
        if not self.is_loaded():
            raise RuntimeError("Tokenizer and model must be loaded before summarizing")

        input_ids = self.tokenizer(
            text, return_tensors="pt", max_length=1024, truncation=True
        )
        input_ids = {k: v.to(self.device) for k, v in input_ids.items()}

        default_params = {
            "max_length": 128,
            "min_length": 20,
            "length_penalty": 2.0,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.2,
            "do_sample": False,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
        }
        default_params.update(kwargs)

        with torch.no_grad():
            summary_ids = self.model.generate(input_ids["input_ids"], **default_params)

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


class PegasusCustomConfig:
    """
    Extension class for Pegasus with custom configuration options.
    Allows for more granular control over model behavior.
    """

    def __init__(
        self, model_name: str = "google/pegasus-xsum", device: Optional[str] = None
    ):
        """
        Initialize with custom configuration options.

        Args:
            model_name (str): Name of the Pegasus model
            device (Optional[str]): Device to load model on
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.config = None

    def load_with_config(
        self, config_kwargs: Dict[str, Any] = None, model_kwargs: Dict[str, Any] = None
    ) -> None:
        """
        Load model with custom configuration.

        Args:
            config_kwargs (Dict[str, Any]): Configuration parameters
            model_kwargs (Dict[str, Any]): Model loading parameters
        """
        from transformers import PegasusConfig

        # Load tokenizer
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)

        # Load configuration
        if config_kwargs:
            self.config = PegasusConfig.from_pretrained(
                self.model_name, **config_kwargs
            )
        else:
            self.config = PegasusConfig.from_pretrained(self.model_name)

        # Load model with custom config
        model_kwargs = model_kwargs or {}
        self.model = PegasusForConditionalGeneration.from_pretrained(
            self.model_name, config=self.config, **model_kwargs
        )
        self.model.to(self.device)
        self.model.eval()

    def set_generation_config(self, **kwargs) -> Dict[str, Any]:
        """
        Set custom generation configuration.

        Args:
            **kwargs: Generation parameters

        Returns:
            Dict[str, Any]: Updated generation config
        """
        default_config = {
            "max_length": 128,
            "min_length": 20,
            "length_penalty": 2.0,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.2,
            "do_sample": False,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else None,
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else None,
        }
        default_config.update(kwargs)
        return default_config

    def summarize(self, text: str, **kwargs) -> str:
        """
        Summarize text with custom configuration.

        Args:
            text (str): Text to summarize
            **kwargs: Generation parameters

        Returns:
            str: Generated summary
        """
        if not self.is_loaded():
            raise RuntimeError("Model must be loaded before summarizing")

        input_ids = self.tokenizer(
            text, return_tensors="pt", max_length=1024, truncation=True
        )
        input_ids = {k: v.to(self.device) for k, v in input_ids.items()}

        generation_config = self.set_generation_config(**kwargs)

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids["input_ids"], **generation_config
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.tokenizer is not None and self.model is not None
