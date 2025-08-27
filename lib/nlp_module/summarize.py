import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from lib.nlp_module.summarization_wrapper import (
    PegasusCustomConfig,
    PegasusSeparateLoader,
    PegasusSummarizationWrapper,
    T5SummarizationWrapper,
)


def linearize_triples(triples):
    sentences = []

    # Define relationship patterns for natural language conversion
    relationship_patterns = {
        # Visual attention relationships
        "lookingat": "is looking at",
        "notlookingat": "is not looking at",
        "unsure": "may be looking at",
        # Spatial relationships
        "above": "is above",
        "beneath": "is beneath",
        "infrontof": "is in front of",
        "behind": "is behind",
        "onthesideof": "is on the side of",
        "in": "is in",
        # Physical interactions
        "carrying": "is carrying",
        "coveredby": "is covered by",
        "drinkingfrom": "is drinking from",
        "eating": "is eating",
        "haveitontheback": "has it on the back",
        "holding": "is holding",
        "leaningon": "is leaning on",
        "lyingon": "is lying on",
        "notcontacting": "is not contacting",
        "otherrelationship": "has some relationship with",
        "sittingon": "is sitting on",
        "standingon": "is standing on",
        "touching": "is touching",
        "twisting": "is twisting",
        "wearing": "is wearing",
        "wiping": "is wiping",
        "writingon": "is writing on",
    }

    for triple in triples:
        subject, predicate, object_item = triple

        # Get the natural language pattern for this relationship
        if predicate in relationship_patterns:
            pattern = relationship_patterns[predicate]
            sentence = f"The {subject} {pattern} the {object_item}."
        else:
            # Fallback for unknown relationships
            sentence = f"The {subject} is {predicate} the {object_item}."

        sentences.append(sentence)

    return sentences


def summarize_sentences(sentences, model_name="google-t5/t5-base", model_type="t5"):
    combined_text = " ".join(sentences)
    if model_type.lower() == "t5":
        wrapper = T5SummarizationWrapper(model_name)
        summary = wrapper.summarize(combined_text)
    elif model_type.lower() == "pegasus":
        wrapper = PegasusSummarizationWrapper(model_name)
        summary = wrapper.summarize(combined_text)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return summary


def summarize_with_pegasus_separate(sentences, model_name="google/pegasus-xsum"):
    combined_text = " ".join(sentences)
    separate_loader = PegasusSeparateLoader(model_name)
    separate_loader.load_tokenizer()
    separate_loader.load_model()
    summary = separate_loader.summarize(combined_text)
    return summary


def summarize_with_pegasus_custom(
    sentences, model_name="google/pegasus-xsum", **kwargs
):
    combined_text = " ".join(sentences)
    custom_config = PegasusCustomConfig(model_name)
    custom_config.load_with_config()
    summary = custom_config.summarize(combined_text, **kwargs)
    return summary


def main():
    triples = [
        ("man", "riding", "bicycle"),
        ("woman", "holding", "umbrella"),
        ("child", "playing with", "dog"),
        ("girl", "wearing", "red dress"),
        ("boy", "sitting on", "bench"),
    ]
    sentences = linearize_triples(triples)

    print("Original sentences:")
    for sentence in sentences:
        print(f"  - {sentence}")
    print()

    # Test different summarization approaches
    print("=" * 60)
    print("T5 Summarization:")
    print("=" * 60)
    summary_t5 = summarize_sentences(sentences, model_type="t5")
    print(f"Summary: {summary_t5}")
    print()

    print("=" * 60)
    print("Pegasus Summarization:")
    print("=" * 60)
    summary_pegasus = summarize_sentences(sentences, model_type="pegasus")
    print(f"Summary: {summary_pegasus}")
    print()

    print("=" * 60)
    print("Pegasus Separate Loader:")
    print("=" * 60)
    summary_separate = summarize_with_pegasus_separate(sentences)
    print(f"Summary: {summary_separate}")
    print()

    summary_custom = summarize_with_pegasus_custom(
        sentences, max_length=80, min_length=15, length_penalty=1.5, num_beams=6
    )
    print(f"Summary: {summary_custom}")


if __name__ == "__main__":
    main()
