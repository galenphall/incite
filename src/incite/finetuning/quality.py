"""Training data quality: normalization, filtering, deduplication, hard negative mining.

All normalization and quality functions live here. Used by merge_training_data.py,
augment_hard_negatives.py, and the training pipeline.
"""

import json
import re
from pathlib import Path

from incite.finetuning.types import load_training_data

# --- Text cleaning ---

_HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")
_SEP_RE = re.compile(r"\s*\[SEP\]\s*")

REFUSAL_PATTERNS = [
    "unable to generate",
    "i appreciate your request",
    "i'm unable",
    "i am unable",
    "cannot provide a meaningful",
    "insufficient information",
    "not enough information to",
]

_REFUSAL_RE = re.compile(
    "|".join(re.escape(p) for p in REFUSAL_PATTERNS),
    re.IGNORECASE,
)

# V2 pipeline source name remapping
V2_SOURCE_REMAP = {
    "s2orc": "s2orc_abstract",
    "synthetic": "synthetic_zotero",
    "existing": "paper_level_v2",
}


def strip_html(text: str) -> str:
    """Strip HTML tags, collapse resulting whitespace."""
    text = _HTML_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def fix_encoding(text: str) -> str:
    """Remove Unicode replacement characters."""
    return text.replace("\ufffd", "")


def has_mojibake(text: str, threshold: float = 0.15) -> bool:
    """Detect mojibake (e.g. Cyrillic decoded as Latin-1).

    Returns True if more than *threshold* of characters fall in the
    Latin-1 Supplement range (U+00C0-00FF).
    """
    if not text:
        return False
    count = sum(1 for c in text if "\u00c0" <= c <= "\u00ff")
    return (count / len(text)) > threshold


def fix_sep_format(text: str) -> str:
    """Replace old [SEP] separator with canonical '. ' format."""
    if "[SEP]" in text:
        return _SEP_RE.sub(". ", text)
    return text


def clean_text(text: str) -> str:
    """Strip HTML tags, fix encoding, fix [SEP] separators, normalize whitespace."""
    return fix_sep_format(fix_encoding(strip_html(text)))


def has_refusal(text: str) -> bool:
    """Check if text contains LLM refusal patterns."""
    return bool(_REFUSAL_RE.search(text))


def clean_hard_negatives(negatives: list[str], positive: str = "") -> list[str]:
    """Remove refusals, short/mojibake entries, HTML, and positive-duplicates.

    Uses format-aware comparison via strip_metadata_prefix.
    """
    from incite.models import strip_metadata_prefix

    positive_core = strip_metadata_prefix(positive) if positive else ""
    cleaned = []
    for n in negatives:
        if has_refusal(n):
            continue
        n = clean_text(n.strip())
        if len(n) < 30:
            continue
        if has_mojibake(n):
            continue
        if positive and (n == positive or strip_metadata_prefix(n) == positive_core):
            continue
        cleaned.append(n)
    return cleaned


# --- Normalization ---


def normalize_training_example(raw: dict, source_tag: str) -> dict | None:
    """Normalize a raw JSONL record to the canonical training schema.

    Returns None if the example should be filtered out.
    """
    query = clean_text(raw.get("query", "").strip())
    positive = clean_text(raw.get("positive", "").strip())

    if len(query) < 30 or len(positive) < 20:
        return None

    if has_refusal(query) or has_refusal(positive):
        return None

    if has_mojibake(query) or has_mojibake(positive):
        return None

    passage_positive = clean_text(raw.get("passage_positive", "").strip())
    if passage_positive and (has_refusal(passage_positive) or has_mojibake(passage_positive)):
        return None

    hard_negatives = clean_hard_negatives(raw.get("hard_negatives", []), positive=positive)
    passage_hard_negatives = clean_hard_negatives(raw.get("passage_hard_negatives", []))

    source = raw.get("source", "") or source_tag

    if source_tag == "v2_pipeline" and source in V2_SOURCE_REMAP:
        source = V2_SOURCE_REMAP[source]

    out: dict = {
        "query": query,
        "positive": positive,
        "hard_negatives": hard_negatives,
        "source_paper_id": raw.get("source_paper_id", ""),
        "cited_paper_id": raw.get("cited_paper_id", ""),
        "source": source,
    }

    if passage_positive:
        out["passage_positive"] = passage_positive
    passage_score = raw.get("passage_score", 0.0)
    if passage_score:
        out["passage_score"] = passage_score
    passage_validation = raw.get("passage_validation", 0)
    if passage_validation:
        out["passage_validation"] = passage_validation
    intent = raw.get("intent", "")
    if intent:
        out["intent"] = intent
    passage_section = raw.get("passage_section", "")
    if passage_section:
        out["passage_section"] = passage_section
    if passage_hard_negatives:
        out["passage_hard_negatives"] = passage_hard_negatives
    scale = raw.get("scale", "")
    if scale:
        out["scale"] = scale

    return out


def normalize_eval_example(raw: dict, source_tag: str) -> dict | None:
    """Normalize a raw eval JSONL record to the canonical PassageTestExample schema."""
    context = clean_text(raw.get("citation_context", "").strip())
    gold = clean_text(raw.get("gold_passage", "").strip())

    if len(context) < 30 or len(gold) < 20:
        return None

    if has_refusal(context) or has_refusal(gold):
        return None

    if has_mojibake(context) or has_mojibake(gold):
        return None

    return {
        "id": raw.get("id", ""),
        "citation_context": context,
        "cited_paper_id": raw.get("cited_paper_id", ""),
        "gold_passage": gold,
        "gold_passage_section": raw.get("gold_passage_section", ""),
        "intent": raw.get("intent", ""),
        "source_paper_id": raw.get("source_paper_id", ""),
        "reference_set_ids": raw.get("reference_set_ids", []),
        "source": source_tag,
    }


# --- Dedup / leakage ---


def dedup_by_query(examples: list[dict]) -> tuple[list[dict], int]:
    """Remove exact-duplicate queries, keeping the first occurrence."""
    seen: set[str] = set()
    deduped = []
    dupes = 0
    for ex in examples:
        key = ex["query"].strip().lower()
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        deduped.append(ex)
    return deduped, dupes


def remove_train_dev_leakage(train: list[dict], dev: list[dict]) -> tuple[list[dict], int]:
    """Remove train examples whose query also appears in dev."""
    dev_queries = {ex["query"].strip().lower() for ex in dev}
    filtered = [ex for ex in train if ex["query"].strip().lower() not in dev_queries]
    removed = len(train) - len(filtered)
    return filtered, removed


def dedup_eval(examples: list[dict]) -> tuple[list[dict], int]:
    """Remove exact-duplicate eval examples by citation_context."""
    seen: set[str] = set()
    deduped = []
    dupes = 0
    for ex in examples:
        key = ex["citation_context"].strip().lower()
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        deduped.append(ex)
    return deduped, dupes


def filter_low_similarity(
    examples: list[dict],
    min_sim: float = 0.15,
    skip_sources: frozenset[str] = frozenset({"s2orc_citation", "s2_contexts", "failure_mining"}),
) -> tuple[list[dict], int]:
    """Remove paper-level examples where query-positive similarity is too low.

    Only checks examples WITHOUT passage_positive (paper-level data where the
    positive is an abstract that may be a wrong OpenAlex match). Skips sources
    with verified query-positive pairs (real citation contexts, failure mining).
    """
    to_check = []
    passed = []

    for ex in examples:
        if ex.get("passage_positive") or ex.get("source") in skip_sources:
            passed.append(ex)
        else:
            to_check.append(ex)

    if not to_check:
        return examples, 0

    print(f"\n  Similarity filter: checking {len(to_check):,} paper-level examples...")

    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    queries = [ex["query"] for ex in to_check]
    positives = [ex["positive"] for ex in to_check]

    q_embs = model.encode(queries, show_progress_bar=False, batch_size=128)
    p_embs = model.encode(positives, show_progress_bar=False, batch_size=128)

    sims = np.array(
        [np.dot(q, p) / (np.linalg.norm(q) * np.linalg.norm(p)) for q, p in zip(q_embs, p_embs)]
    )

    kept = [ex for ex, sim in zip(to_check, sims) if sim >= min_sim]
    removed = len(to_check) - len(kept)

    below_thresholds = {
        0.0: int((sims < 0.0).sum()),
        0.1: int((sims < 0.1).sum()),
        0.15: int((sims < 0.15).sum()),
        0.2: int((sims < 0.2).sum()),
    }
    print(f"    Checked:  {len(to_check):,}")
    print(f"    Removed:  {removed:,} (sim < {min_sim})")
    print(
        f"    Distribution: <0.0={below_thresholds[0.0]}, "
        f"<0.1={below_thresholds[0.1]}, "
        f"<0.15={below_thresholds[0.15]}, "
        f"<0.2={below_thresholds[0.2]}"
    )

    return passed + kept, removed


# --- Hard negative mining ---


def mine_hard_negatives(
    input_path: Path,
    output_path: Path,
    model_path: str = "models/granite-citation-v6/final",
    top_k: int = 10,
    num_negatives: int = 5,
    batch_size: int = 64,
    show_progress: bool = True,
    query_prefix: str = "",
    passage_prefix: str = "",
    device: str | None = None,
    search_batch_size: int = 4096,
) -> dict:
    """Mine hard negatives using a trained model's embeddings.

    For each training example, embeds the query, finds the top-k most similar
    positives in the corpus (excluding the actual positive), and uses the top
    num_negatives as hard negatives.

    Args:
        batch_size: Encode batch size. Keep low (64) to avoid OOM on unified
            memory (MPS) or constrained machines.
        query_prefix: Prefix to prepend to queries (e.g. "query: " for Granite).
        passage_prefix: Prefix to prepend to positives (e.g. "passage: " for Granite).
        device: Torch device for encoding. None = auto-detect, "cpu" recommended
            on laptops to avoid MPS unified memory pressure.
        search_batch_size: Number of queries per FAISS search batch.
    """
    import gc

    import numpy as np

    from incite.models import strip_metadata_prefix

    examples = load_training_data(input_path)
    if not examples:
        return {"error": "No examples to process"}

    print(f"Loaded {len(examples):,} examples from {input_path}")

    # Build corpus of unique positives
    positive_texts: list[str] = []
    positive_indices: dict[str, int] = {}
    for ex in examples:
        key = ex.positive.strip()
        if key not in positive_indices:
            positive_indices[key] = len(positive_texts)
            positive_texts.append(key)

    print(f"Unique positives: {len(positive_texts):,}")
    if query_prefix or passage_prefix:
        print(f"Prefixes: query='{query_prefix}', passage='{passage_prefix}'")

    from sentence_transformers import SentenceTransformer

    if device is None:
        from incite.utils import get_best_device

        device = get_best_device()

    print(f"Loading model from {model_path} on {device}...")
    model = SentenceTransformer(model_path, device=device)

    print("Encoding positives...")
    encode_positives = (
        [f"{passage_prefix}{t}" for t in positive_texts] if passage_prefix else positive_texts
    )
    positive_embs = model.encode(
        encode_positives,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("Encoding queries...")
    queries = [ex.query for ex in examples]
    encode_queries = [f"{query_prefix}{q}" for q in queries] if query_prefix else queries
    query_embs = model.encode(
        encode_queries,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Free model memory before FAISS search — encoding is done
    del model
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("Model freed, building FAISS index...")

    import faiss

    # Explicitly set FAISS OpenMP threads to avoid deadlock on macOS.
    # torch's libomp can conflict with FAISS's OpenMP barrier when both
    # are loaded in the same process.
    faiss.omp_set_num_threads(1)

    dim = positive_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(positive_embs.astype(np.float32)))
    print(f"Built FAISS index: {index.ntotal} vectors, {dim}-dim")

    # Batch the FAISS search to avoid huge temporary distance matrices
    # (43K queries × 30K docs = 5.2 GB temp allocation in one shot)
    print(f"Mining hard negatives (search batches of {search_batch_size})...")
    query_embs = np.ascontiguousarray(query_embs.astype(np.float32))
    n_queries = query_embs.shape[0]
    all_scores = []
    all_indices = []
    for start in range(0, n_queries, search_batch_size):
        end = min(start + search_batch_size, n_queries)
        batch_scores, batch_indices = index.search(query_embs[start:end], top_k + 1)
        all_scores.append(batch_scores)
        all_indices.append(batch_indices)
    scores = np.concatenate(all_scores, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    del all_scores, all_indices

    stats = {
        "total_examples": len(examples),
        "examples_augmented": 0,
        "negatives_added": 0,
        "already_had_negatives": 0,
        "unique_positives": len(positive_texts),
    }

    augmented = []
    for i, ex in enumerate(examples):
        actual_positive_key = ex.positive.strip()
        actual_positive_idx = positive_indices.get(actual_positive_key, -1)

        existing_neg_set = set(n.strip() for n in ex.hard_negatives)
        existing_neg_cores = {strip_metadata_prefix(n.strip()) for n in ex.hard_negatives}
        actual_positive_core = strip_metadata_prefix(actual_positive_key)
        if existing_neg_set:
            stats["already_had_negatives"] += 1

        mined_negs: list[str] = []
        for j in range(top_k + 1):
            candidate_idx = int(indices[i][j])
            if candidate_idx < 0:
                continue
            if candidate_idx == actual_positive_idx:
                continue
            candidate_text = positive_texts[candidate_idx]
            candidate_stripped = candidate_text.strip()
            candidate_core = strip_metadata_prefix(candidate_stripped)
            if candidate_core == actual_positive_core:
                continue
            if candidate_stripped in existing_neg_set or candidate_core in existing_neg_cores:
                continue
            mined_negs.append(candidate_text)
            if len(mined_negs) >= num_negatives:
                break

        if mined_negs:
            ex.hard_negatives = list(ex.hard_negatives) + mined_negs
            stats["examples_augmented"] += 1
            stats["negatives_added"] += len(mined_negs)

        augmented.append(ex)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in augmented:
            f.write(json.dumps(ex.to_dict()) + "\n")

    print("\nMining complete:")
    print(f"  Examples processed:    {stats['total_examples']:,}")
    print(f"  Already had negatives: {stats['already_had_negatives']:,}")
    print(f"  Examples augmented:    {stats['examples_augmented']:,}")
    print(f"  Negatives added:       {stats['negatives_added']:,}")
    print(f"  Output: {output_path}")

    return stats
