"""Generate passage-level training data from full-text corpora (unarXiv / S2ORC).

This module has been moved to incite.finetuning._passage.fulltext.
Re-exports are provided here for backward compatibility.
"""

# Backward compat re-exports
# Also re-export internal helpers used by s2orc_citation_mining.py
from incite.finetuning._passage.fulltext import (  # noqa: F401  # noqa: F401
    _clean_fulltext,
    _parse_s2orc_annotation,
    _safe_span,
    generate_passage_data,
    iter_fulltext_papers,
    parse_fulltext_record,
)
