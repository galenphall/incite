"""Prompt templates and response parsers for synthetic citation context generation.

Contains the LLM prompts used to generate synthetic citation contexts
(background, methods, results, comparison, motivation) and the validation
logic that parses and quality-checks LLM responses.

Quality checks include: JSON structure validation, [CITE] marker presence,
title repetition detection, minimum length requirements, and type validation.

Related modules:
    - incite.corpus.synthetic_contexts: Generation orchestration (consumer).
    - incite.corpus.synthetic_db: SQLite storage for generated contexts.
"""

import json
import logging
import re

from incite.models import Paper

logger = logging.getLogger(__name__)

VALID_TYPES = {"background", "methods", "results", "comparison", "motivation"}

SYNTHETIC_PROMPT = """\
You are an expert academic writer. Given a paper's title and abstract, write 5 \
citation contexts — passages where a researcher might cite this paper. Each \
passage should be 2-4 sentences of flowing academic prose.

Rules:
- Mark the citation point with [CITE]
- Do NOT restate the paper title verbatim
- Do NOT use author names from this paper
- Each passage must be a DIFFERENT citation type:
  1. background — establishing prior work or field context
  2. methods — citing a methodology, technique, or tool
  3. results — referencing specific findings or data
  4. comparison — comparing/contrasting with another approach
  5. motivation — using the paper to motivate new research

Return JSON only:
{{"contexts": [
  {{"type": "<type>", "text": "<passage with [CITE]>", "section": "<likely section heading>"}},
  ...5 total
]}}

Title: {title}
Abstract: {abstract}"""

MODERATE_VALID_TYPES = {"tangential", "field_reference", "methodological_detail"}

MODERATE_DIFFICULTY_PROMPT = """\
You are an expert academic writer. Given a paper's title and abstract, write 3 \
citation contexts that would be CHALLENGING for a retrieval system to match back \
to this paper. Each passage should be 2-4 sentences of flowing academic prose.

The goal: write passages where someone cites this paper, but in a way that does \
NOT closely mirror the paper's specific language or main finding. Think about how \
papers are actually cited in practice — often tangentially, for minor points, or \
as one of many related works.

Rules:
- Mark the citation point with [CITE]
- Do NOT restate the paper title verbatim
- Do NOT use author names from this paper
- AVOID vocabulary that appears in the abstract — paraphrase heavily, use \
synonyms, describe broader concepts rather than paper-specific terms
- Each passage must be a DIFFERENT citation type:
  1. tangential — cite for a peripheral finding, secondary contribution, or \
a detail mentioned only in passing, NOT the paper's main result
  2. field_reference — a broad statement about a research area where this paper \
is one of many relevant works; the passage should NOT single out what makes \
this paper unique
  3. methodological_detail — cite for a specific technical choice (a dataset, \
preprocessing step, evaluation metric, or parameter setting) rather than \
the core method or finding

Return JSON only:
{{"contexts": [
  {{"type": "<type>", "text": "<passage with [CITE]>", "section": "<likely section heading>"}},
  ...3 total
]}}

Title: {title}
Abstract: {abstract}"""


def parse_response(paper: Paper, response_text: str) -> list[dict]:
    """Parse and validate LLM response into context dicts.

    Args:
        paper: The target paper
        response_text: Raw LLM response text

    Returns:
        List of validated context dicts with keys:
        id, paper_id, citation_type, text, section_hint
    """
    # Strip markdown code fences — some models wrap JSON in ```json ... ``` blocks
    # even when instructed to return JSON only.
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Reject entirely if response is not valid JSON — log a warning with a
    # snippet so failures can be diagnosed without flooding the log.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        snippet = text[:200] if len(text) > 200 else text
        logger.warning(
            "Malformed LLM JSON for paper %s (%s): %s",
            paper.id,
            paper.title[:60],
            snippet,
        )
        return []

    contexts_raw = data.get("contexts", [])
    # Guard against malformed schema where "contexts" is not a list.
    if not isinstance(contexts_raw, list):
        return []

    title_lower = paper.title.lower()
    seen_types = set()
    results = []

    for ctx in contexts_raw:
        # Skip non-dict entries — LLM occasionally emits nulls or strings.
        if not isinstance(ctx, dict):
            continue

        ctype = ctx.get("type", "").strip().lower()
        ctext = ctx.get("text", "").strip()
        section = ctx.get("section", "").strip()

        # Reject unknown citation types — keeps the type distribution controlled
        # and prevents garbage values from polluting the training set.
        if ctype not in VALID_TYPES:
            continue

        # Reject duplicate types — each type should appear exactly once per paper
        # so the training set is balanced across citation styles.
        if ctype in seen_types:
            continue

        # Reject if [CITE] marker is missing — the marker is the signal used at
        # retrieval time to locate the citation span; without it the context is useless.
        if "[CITE]" not in ctext:
            continue

        # Reject very short passages — fewer than 30 chars is almost certainly
        # a truncation or placeholder, not a real citation context.
        if len(ctext) < 30:
            continue

        # Reject if title appears verbatim — the model is copying metadata
        # instead of writing a natural citation context. This would make the
        # training example trivially easy and misleadingly high-confidence.
        if title_lower in ctext.lower():
            continue

        seen_types.add(ctype)
        results.append(
            {
                "id": f"synth_{paper.id}_{ctype}",
                "paper_id": paper.id,
                "citation_type": ctype,
                "text": ctext,
                "section_hint": section or None,
            }
        )

    return results


def parse_moderate_response(paper: Paper, response_text: str) -> list[dict]:
    """Parse and validate LLM response for moderate-difficulty contexts.

    Same validation as parse_response but uses moderate types and
    generates IDs like synth_{paper_id}_mod_{type}.
    """
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        snippet = text[:200] if len(text) > 200 else text
        logger.warning(
            "Malformed LLM JSON for paper %s (%s): %s",
            paper.id,
            paper.title[:60],
            snippet,
        )
        return []

    contexts_raw = data.get("contexts", [])
    if not isinstance(contexts_raw, list):
        return []

    title_lower = paper.title.lower()
    seen_types = set()
    results = []

    for ctx in contexts_raw:
        if not isinstance(ctx, dict):
            continue

        ctype = ctx.get("type", "").strip().lower()
        ctext = ctx.get("text", "").strip()
        section = ctx.get("section", "").strip()

        if ctype not in MODERATE_VALID_TYPES:
            continue
        if ctype in seen_types:
            continue
        if "[CITE]" not in ctext:
            continue
        if len(ctext) < 30:
            continue
        if title_lower in ctext.lower():
            continue

        seen_types.add(ctype)
        results.append(
            {
                "id": f"synth_{paper.id}_mod_{ctype}",
                "paper_id": paper.id,
                "citation_type": ctype,
                "text": ctext,
                "section_hint": section or None,
                "difficulty": "moderate",
            }
        )

    return results
