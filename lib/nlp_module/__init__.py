"""
NLP Module for text summarization and processing.
"""

from .summarization_wrapper import (
    BaseSummarizationWrapper,
    PegasusCustomConfig,
    PegasusSeparateLoader,
    PegasusSummarizationWrapper,
    T5SummarizationWrapper,
)
from .summarize import (
    linearize_triples,
    summarize_sentences,
    summarize_with_pegasus_custom,
    summarize_with_pegasus_separate,
)

__all__ = [
    "linearize_triples",
    "summarize_sentences",
    "summarize_with_pegasus_separate",
    "summarize_with_pegasus_custom",
    "BaseSummarizationWrapper",
    "T5SummarizationWrapper",
    "PegasusSummarizationWrapper",
    "PegasusSeparateLoader",
    "PegasusCustomConfig",
]
