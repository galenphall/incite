"""Passage-level training data generation using LLM.

This module has been moved to incite.finetuning._passage.generation.
Re-exports are provided here for backward compatibility.
"""

# Backward compat re-exports
from incite.finetuning._passage.generation import (  # noqa: F401
    PASSAGE_PROMPT,
    SKIP_SECTIONS,
    VALID_TYPES,
    contexts_to_training_examples,
    generate_passage_contexts_batch,
    parse_passage_response,
    select_passages,
    split_passage_data,
)
