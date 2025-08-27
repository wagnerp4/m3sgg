"""
NLP Module for text summarization and processing.
"""

from .summarize import (
    linearize_triples,
    summarize_sentences,
    summarize_with_pegasus_separate,
    summarize_with_pegasus_custom,
)
from .summarization_wrapper import (
    BaseSummarizationWrapper,
    T5SummarizationWrapper,
    PegasusSummarizationWrapper,
    PegasusSeparateLoader,
    PegasusCustomConfig,
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
