from m3sgg.language.summarization.wrappers import (
    PegasusCustomConfig,
    PegasusSeparateLoader,
    PegasusSummarizationWrapper,
    T5SummarizationWrapper,
)

def linearize_triples(triples, mode="flat"):
    """Convert scene graph triples into natural language sentences.

    Transforms subject-predicate-object triples into human-readable sentences
    using predefined relationship patterns for visual attention, spatial
    relationships, and physical interactions.

    :param triples: List of (subject, predicate, object) tuples
    :type triples: list
    :param mode: Linearization mode (flat, majority, time)
    :type mode: str
    :return: List of natural language sentences
    :rtype: list
    """
    sentences = []
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

    if mode == "flat":
        # Simple flat linearization - all triples as separate sentences
        for triple in triples:
            subject, predicate, object_item = triple
            if predicate in relationship_patterns:
                pattern = relationship_patterns[predicate]
                sentence = f"The {subject} {pattern} the {object_item}."
            else:
                # Fallback for unknown relationships
                sentence = f"The {subject} is {predicate} the {object_item}."
            sentences.append(sentence)
    
    elif mode == "majority":
        # Majority voting - group by object and use most common relationship
        from collections import defaultdict, Counter
        
        object_relationships = defaultdict(list)
        for triple in triples:
            subject, predicate, object_item = triple
            object_relationships[object_item].append((subject, predicate))
        
        for object_item, rels in object_relationships.items():
            # Count relationships for this object
            rel_counts = Counter([rel for _, rel in rels])
            most_common_rel = rel_counts.most_common(1)[0][0]
            
            # Get all subjects for this relationship
            subjects = [subj for subj, rel in rels if rel == most_common_rel]
            subject_text = " and ".join(subjects) if len(subjects) > 1 else subjects[0]
            
            if most_common_rel in relationship_patterns:
                pattern = relationship_patterns[most_common_rel]
                sentence = f"The {subject_text} {pattern} the {object_item}."
            else:
                sentence = f"The {subject_text} is {most_common_rel} the {object_item}."
            sentences.append(sentence)
    
    elif mode == "time":
        # Time-aware linearization - group by temporal patterns
        # For now, just use flat linearization with temporal markers
        for i, triple in enumerate(triples):
            subject, predicate, object_item = triple
            if predicate in relationship_patterns:
                pattern = relationship_patterns[predicate]
                sentence = f"At time {i+1}: The {subject} {pattern} the {object_item}."
            else:
                sentence = f"At time {i+1}: The {subject} is {predicate} the {object_item}."
            sentences.append(sentence)
    
    else:
        # Default to flat mode
        for triple in triples:
            subject, predicate, object_item = triple
            if predicate in relationship_patterns:
                pattern = relationship_patterns[predicate]
                sentence = f"The {subject} {pattern} the {object_item}."
            else:
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

def summarize_with_pegasus_custom(sentences, model_name="google/pegasus-xsum", **kwargs):
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
