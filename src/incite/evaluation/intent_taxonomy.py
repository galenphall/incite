"""Citation intent and failure mode taxonomies for diagnosis."""

# Citation intent categories — why a paper is being cited
CITATION_INTENTS = [
    "background",  # Foundational/classic work in the field
    "method",  # Methodology, tools, techniques adopted
    "support",  # Evidence supporting a claim
    "contrast",  # Contradicting/opposing findings
    "data_source",  # Dataset, corpus, or data origin
    "prior_art",  # Direct predecessor work being extended
    "definition",  # Concept definition or terminology
]

# Failure modes — why the recommendation system got it wrong
FAILURE_MODES = [
    "semantic_mismatch",  # Query and GT paper are about genuinely different subtopics
    "specificity_gap",  # Query is specific, GT is general (or vice versa)
    "metadata_dependent",  # Citation is about who/when/where, not semantic content
    "multi_citation",  # Context cites N papers; hard to isolate one
    "ambiguous_context",  # Not enough info in context to identify the paper
    "lexical_gap",  # Same concept, different terminology
    "correct_competitor",  # Model's top prediction is a reasonable alternative
]

# Difficulty levels for theoretical ceiling analysis
DIFFICULTY_LEVELS = ["easy", "hard", "ambiguous"]

# Valid values sets for validation
VALID_INTENTS = set(CITATION_INTENTS)
VALID_FAILURE_MODES = set(FAILURE_MODES) | {""}  # empty string for successes
VALID_DIFFICULTIES = set(DIFFICULTY_LEVELS)
