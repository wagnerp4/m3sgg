"""
Evaluation metrics for summarization quality assessment.

This module provides comprehensive metrics for evaluating summarization models
including ROUGE, BLEU, METEOR, and semantic similarity metrics.
"""

from typing import List, Dict
import logging

try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError as e:
    ROUGE_AVAILABLE = False
    logging.warning(
        f"rouge_score library not available: {e}. Install with: pip install rouge-score"
    )

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError as e:
    NLTK_AVAILABLE = False
    logging.warning(f"nltk library not available: {e}. Install with: pip install nltk")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning(
        f"sentence-transformers library not available: {e}. Install with: pip install sentence-transformers"
    )

logger = logging.getLogger(__name__)


class SummarizationMetrics:
    """Comprehensive metrics for summarization evaluation.

    Provides ROUGE, BLEU, METEOR, and semantic similarity metrics
    for evaluating summarization quality.

    :param rouge_types: List of ROUGE types to compute
    :type rouge_types: List[str], optional
    :param use_stemmer: Whether to use stemming for ROUGE
    :type use_stemmer: bool, optional
    :param sentence_model: Sentence transformer model for semantic similarity
    :type sentence_model: str, optional
    """

    def __init__(
        self,
        rouge_types: List[str] = None,
        use_stemmer: bool = True,
        sentence_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize summarization metrics.

        :param rouge_types: List of ROUGE types to compute
        :type rouge_types: List[str], optional
        :param use_stemmer: Whether to use stemming for ROUGE
        :type use_stemmer: bool, optional
        :param sentence_model: Sentence transformer model for semantic similarity
        :type sentence_model: str, optional
        """
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
        self.use_stemmer = use_stemmer

        # Initialize ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                self.rouge_types, use_stemmer=use_stemmer
            )
        else:
            self.rouge_scorer = None
            logger.warning("ROUGE metrics will not be available")

        # Initialize sentence transformer for semantic similarity
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(sentence_model)
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None
            logger.warning("Semantic similarity metrics will not be available")

        # Download required NLTK data
        if NLTK_AVAILABLE:
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)  # New punkt tokenizer
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK data: {e}")
                # Note: We can't modify the global NLTK_AVAILABLE here
                # The import check at module level will handle this

    def compute_rouge(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE scores.

        :param predictions: List of predicted summaries
        :type predictions: List[str]
        :param references: List of reference summaries
        :type references: List[str]
        :return: Dictionary of ROUGE scores
        :rtype: Dict[str, float]
        """
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            logger.warning("ROUGE not available, returning empty scores")
            return {rouge_type: 0.0 for rouge_type in self.rouge_types}

        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")

        rouge_scores = {rouge_type: [] for rouge_type in self.rouge_types}

        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                rouge_scores[rouge_type].append(scores[rouge_type].fmeasure)

        # Compute averages
        avg_scores = {}
        for rouge_type in self.rouge_types:
            avg_scores[rouge_type] = sum(rouge_scores[rouge_type]) / len(
                rouge_scores[rouge_type]
            )

        return avg_scores

    def compute_bleu(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute BLEU scores.

        :param predictions: List of predicted summaries
        :type predictions: List[str]
        :param references: List of reference summaries
        :type references: List[str]
        :return: Dictionary of BLEU scores
        :rtype: Dict[str, float]
        """
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, returning empty BLEU scores")
            return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")

        smoothing = SmoothingFunction().method1
        bleu_scores = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": []}

        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())

            # Compute BLEU scores for different n-grams
            for n in range(1, 5):
                bleu_score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=[1 / n] * n,
                    smoothing_function=smoothing,
                )
                bleu_scores[f"bleu{n}"].append(bleu_score)

        # Compute averages
        avg_scores = {}
        for bleu_type in bleu_scores:
            avg_scores[bleu_type] = sum(bleu_scores[bleu_type]) / len(
                bleu_scores[bleu_type]
            )

        return avg_scores

    def compute_meteor(self, predictions: List[str], references: List[str]) -> float:
        """Compute METEOR score.

        :param predictions: List of predicted summaries
        :type predictions: List[str]
        :param references: List of reference summaries
        :type references: List[str]
        :return: METEOR score
        :rtype: float
        """
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, returning 0.0 for METEOR")
            return 0.0

        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")

        meteor_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())

            score = meteor_score([ref_tokens], pred_tokens)
            meteor_scores.append(score)

        return sum(meteor_scores) / len(meteor_scores)

    def compute_semantic_similarity(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Compute semantic similarity using sentence transformers.

        :param predictions: List of predicted summaries
        :type predictions: List[str]
        :param references: List of reference summaries
        :type references: List[str]
        :return: Average semantic similarity score
        :rtype: float
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.sentence_model is None:
            logger.warning(
                "Sentence transformers not available, returning 0.0 for semantic similarity"
            )
            return 0.0

        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")

        # Encode all texts
        all_texts = predictions + references
        embeddings = self.sentence_model.encode(all_texts)

        # Split embeddings
        pred_embeddings = embeddings[: len(predictions)]
        ref_embeddings = embeddings[len(predictions) :]

        # Compute cosine similarities
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            similarity = cosine_similarity([pred_emb], [ref_emb])[0][0]
            similarities.append(similarity)

        return sum(similarities) / len(similarities)

    def compute_all_metrics(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute all available metrics.

        :param predictions: List of predicted summaries
        :type predictions: List[str]
        :param references: List of reference summaries
        :type references: List[str]
        :return: Dictionary of all computed metrics
        :rtype: Dict[str, float]
        """
        metrics = {}

        # ROUGE scores
        rouge_scores = self.compute_rouge(predictions, references)
        metrics.update(rouge_scores)

        # BLEU scores
        bleu_scores = self.compute_bleu(predictions, references)
        metrics.update(bleu_scores)

        # METEOR score
        meteor_score = self.compute_meteor(predictions, references)
        metrics["meteor"] = meteor_score

        # Semantic similarity
        semantic_sim = self.compute_semantic_similarity(predictions, references)
        metrics["semantic_similarity"] = semantic_sim

        return metrics

    def format_results(self, metrics: Dict[str, float], precision: int = 4) -> str:
        """Format metrics results for display.

        :param metrics: Dictionary of metrics
        :type metrics: Dict[str, float]
        :param precision: Number of decimal places
        :type precision: int
        :return: Formatted results string
        :rtype: str
        """
        lines = ["Summarization Metrics Results:"]
        lines.append("=" * 40)

        # Group metrics by type
        rouge_metrics = {k: v for k, v in metrics.items() if k.startswith("rouge")}
        bleu_metrics = {k: v for k, v in metrics.items() if k.startswith("bleu")}
        other_metrics = {
            k: v
            for k, v in metrics.items()
            if not k.startswith("rouge") and not k.startswith("bleu")
        }

        if rouge_metrics:
            lines.append("ROUGE Scores:")
            for metric, score in rouge_metrics.items():
                lines.append(f"  {metric.upper()}: {score:.{precision}f}")

        if bleu_metrics:
            lines.append("\nBLEU Scores:")
            for metric, score in bleu_metrics.items():
                lines.append(f"  {metric.upper()}: {score:.{precision}f}")

        if other_metrics:
            lines.append("\nOther Metrics:")
            for metric, score in other_metrics.items():
                lines.append(
                    f"  {metric.replace('_', ' ').title()}: {score:.{precision}f}"
                )

        return "\n".join(lines)


def main():
    """Example usage of SummarizationMetrics."""
    # Example predictions and references
    predictions = [
        "A person is walking in the park with a dog.",
        "A man is cooking food in the kitchen.",
        "Children are playing soccer in the field.",
    ]

    references = [
        "A person walks through the park with their dog.",
        "A man prepares food in the kitchen.",
        "Kids are playing football on the field.",
    ]

    # Initialize metrics
    metrics = SummarizationMetrics()

    # Compute all metrics
    results = metrics.compute_all_metrics(predictions, references)

    # Print formatted results
    print(metrics.format_results(results))


if __name__ == "__main__":
    main()
