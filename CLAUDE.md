# CLAUDE.md

## Working Directory

The working directory is ALREADY set to this project folder. **NEVER use `cd` before commands.** Run everything directly (e.g., `incite evaluate --method hybrid`, `pytest tests/`).

## Project Overview

inCite is a local-first citation recommendation system. Users write text and the system suggests relevant papers from their corpus. Best results: MRR=0.550, R@1=38.5%, R@10=88.5%, C-index=0.851, Skill MRR=0.460 (Granite-FT v6b neural-only) on primary benchmark (3420 queries after cleaning, 100 source papers).

**Product vision**: Paper-level ranking for discovery + paragraph-level evidence snippets when a passage is compelling enough. See `docs/PROJECT_PLAN.md` for roadmap.

**Repository**: [github.com/galenphall/incite](https://github.com/galenphall/incite)

## Current Priorities

0. **Training data expansion complete.** v6b (MRR=0.550) confirmed as ceiling for Granite-small-R2. See `docs/retrieval/training_experiment_log.md`.
1. **Phase 10 — Optimized embedding training. Ceiling reached.** Recommendation pipeline frozen.
2. **Tune evidence threshold**: Current threshold is 0.35. May want higher for paper mode.

### Someday/Maybe
- **Larger model or different training objective**: Next step for retrieval quality beyond Granite-small-R2 ceiling.
- **Intent classifier**: Route broad vs. specific citation contexts to appropriate retrieval strategy.

## Commands

```bash
# Dev setup, tests, linting
pip install -e ".[dev]"
pytest
ruff check src/incite && ruff format src/incite

# Core workflow
incite index --corpus data/processed/corpus.jsonl --output data/processed/index
incite recommend "your text here" -k 10 --method hybrid
incite evaluate --method hybrid --name "experiment notes"
incite evaluate --macro-average                  # correct for source-paper skew
incite evaluate --sweep scales --no-log          # compare all context scales (incl. cursor-weighted)
incite evaluate --two-stage --alpha 0.6          # two-stage paper+paragraph retrieval
incite serve --embedder minilm-ft                # REST API at localhost:8230 (must match FAISS index embedder)
incite webapp                                    # Streamlit UI at localhost:8501

# Fine-tuning (master files: 64K train / 12.5K dev / 3K eval)
incite finetune train --train master_train.jsonl --dev master_dev.jsonl

# Experiments
incite experiments list
incite experiments diff ID1 ID2                  # per-query diff with significance test
```

Run `incite --help` or `incite <subcommand> --help` for full CLI reference.

## Architecture

### Retrieval Pipeline

1. **Neural**: Embeds papers as `title. authors. year. journal. abstract` (via `format_paper_embedding_text()`), searches FAISS index (flat, HNSW, or SQ8)
2. **BM25**: Lexical keyword matching with English stemming (`tokenize_with_stemming` in `bm25.py`)
3. **Hybrid**: RRF fusion (k=5, stemmed BM25). Adds +1.9pp MRR / +1.5pp R@1 over neural-only with MiniLM-FT v4.
4. **Evidence lookup** (dual-path): After paper-level ranking, searches chunk FAISS index. Best matching paragraph per paper attached as `matched_paragraph` if chunk score >= 0.35. ~4ms overhead.
5. **Two-stage**: Stage 1 paper-level hybrid -> Stage 2 scoped paragraph search within top-50 -> `final = alpha * paper_score + (1-alpha) * chunk_score`, default alpha=0.6.

### Embedders

**MiniLM-FT v4** (best): Fine-tuned on 170K pairs with Matryoshka training (128/256/384-dim), CachedMNRL, early stopping.
**Granite-FT v3**: Fine-tuned Granite-small-R2, 384-dim, 8K context, asymmetric query/passage prefixes.
Also available: MiniLM, E5-small, SPECTER2, Nomic v1.5, Voyage AI, ModernBERT, SciNCL, MiniLM-FT ONNX.
Factory keys: `minilm`, `e5`, `specter`, `nomic`, `voyage`, `minilm-ft`, `minilm-ft-onnx`, `modernbert`, `scincl`, `granite-ft` (registered in `retrieval/factory.py:EMBEDDERS`).
**Model path configuration**: Fine-tuned embedders load from the `"model"` key in their `EMBEDDERS` entry. `get_embedder()` passes this as `model_path` kwarg. To swap models: change one line in `factory.py`.

### Core Abstractions

- `BaseEmbedder` ABC (`embeddings/base.py`): `embed(texts)`, `embed_query(query)`, abstract `dimension` property
- `VectorStore` protocol (`interfaces.py`): `add/search/save/load`
- `CorpusSource` protocol (`interfaces.py`): `load_papers()`, `needs_refresh()`, `cache_key()`
- `Retriever` ABC (`interfaces.py`): `retrieve(query, k)` returns `list[RetrievalResult]`
- `Reranker` ABC (`interfaces.py`): ABANDONED — cross-encoder reranking catastrophically degrades results. See `docs/retrieval/training_experiment_log.md`.
- `TwoStageRetriever` (`retrieval/two_stage.py`): Alpha-blends paper and chunk scores

### Data Flow

```
corpus.jsonl -> Embedder -> FAISSStore (index/)
                                |
test_set.jsonl -> Retriever -> EvaluationResult -> ExperimentLogger (experiments.jsonl)
```

### Module Map

```
src/incite/
  agent.py          # InCiteAgent — SDK entry point (from_zotero, from_corpus, recommend)
  api.py            # FastAPI REST server (incite serve)
  models.py         # Dataclasses + canonical embedding format functions + ReferenceItem Protocol
  interfaces.py     # ABCs/Protocols: Retriever, VectorStore, CorpusSource, Reranker
  utils.py          # get_best_device(), deduplication, shared helpers
  corpus/           # Data ingestion: Zotero reader, PDF extraction, chunking, enrichment, APIs
  embeddings/       # BaseEmbedder impls + FAISSStore, ChunkStore
  retrieval/        # Neural, BM25, hybrid, paragraph, two_stage retrievers + factory.py
  evaluation/       # Metrics, experiment logging, passage-level evaluation
  finetuning/       # Training data generation + training pipeline
  cli/              # Subcommands: core, data, llm, finetune, experiments, agent, serve, acquire
  webapp/           # Streamlit app + state.py (caching/wiring layer)
  acquire/          # PDF acquisition via library proxy + Unpaywall
  export/           # Citation export: BibTeX + RIS formatters, ExportFormat Protocol
scripts/            # Utilities: merge_training_data, download_s2orc, alpha_sweep, etc.
editor-plugins/     # Obsidian, VS Code, Google Docs, Chrome hotkey + shared TS package
```

### Cache Directory (`~/.incite/`)

```
config.json                       # Persisted settings
zotero_corpus.jsonl               # Cached papers from Zotero
zotero_index_{embedder}/          # Paper-level FAISS index per embedder type
zotero_chunks_{strategy}.jsonl    # Cached chunks per chunking strategy
zotero_chunks_{strategy}.version  # Version file for cache invalidation (currently v3)
zotero_chunks_{embedder}/         # Chunk FAISS index for evidence lookup
grobid_output/                    # GROBID TEI-XML results
grobid_chunks.jsonl               # Converted chunks from GROBID output (source of truth)
last_recommendations.json         # Last recommend output (for `acquire --from-last`)
```

Auto-invalidated when Zotero DB mtime changes. Delete `~/.incite/` to force full rebuild.

### CRITICAL: Cache Safety Rules

**The chunk cache (`zotero_chunks_*.jsonl`) is expensive to rebuild.** The primary source is GROBID processing. Recovery source: `grobid_chunks.jsonl` + `grobid_output/tei_xml/`. **NEVER:**

1. **Bump `CHUNK_CACHE_VERSION` for filter-only changes.** The version bump triggers a full rebuild, but `from_zotero()` loads papers without PDF text, producing only abstract chunks (~3K) instead of PDF chunks (~120K). Use `_refilter_cached_chunks()` instead — it loads existing chunks, applies new filters, and saves back.
2. **Delete chunk cache files without a backup.** `extract_and_save_pdfs()` now creates `.bak` files before clearing. Any code that deletes chunk caches must do the same.
3. **Overwrite a large cache with a smaller one.** `load_zotero_chunks()` refuses to overwrite >1000 chunks with <50% of the original count. This catches accidental rebuilds from metadata-only papers.
4. **Assume papers have PDF text.** Papers loaded via `load_zotero_direct()` only have metadata. PDF text requires explicit extraction (`extract_and_save_pdfs`) or loading from a corpus file that already has it.

**To add new chunk filters:** Add the filter to `chunk_paper()` in `chunking.py` AND to `_refilter_cached_chunks()` in `state.py`. Bump `CHUNK_CACHE_VERSION` — the re-filter path handles the upgrade safely.

**To recover chunks from GROBID:** `python scripts/cloud_grobid/convert_to_chunks.py` (reads `~/.incite/grobid_output/tei_xml/`, writes `grobid_chunks.jsonl`).

### Key Design Decisions

- **Per-paper reference sets**: Eval searches only within each source paper's cited works
- **Title dedup**: `deduplicate=True` removes duplicate titles (enabled in agent/webapp, disabled in eval)
- **Confidence scoring**: `RetrievalResult.confidence` (0-1) from cosine similarity. Green/yellow/red at 0.55/0.35.
- **Multi-scale queries**: local, narrow (3 sent), broad (6 sent), section, global, cursor (weighted)
- **Two-stage alpha blending**: `final = alpha * paper_score + (1-alpha) * chunk_score`, default alpha=0.6.
- **Evaluation**: Per-query tracking, bootstrap CIs, macro-averaging, paired significance tests
- **Primary metrics: MRR and R@1.** R@10/R@20 inflated by small ref sets (median=19, random R@20=86%).
- **Corpus-size-adjusted metrics** (in `metrics.py`): C-index (concordance, 0-1) and Skill MRR (normalized against random baseline).

### Canonical Embedding Text Format

**All code that formats text for embedding MUST use these four functions in `models.py`.** This prevents train/eval distribution shift from inconsistent formatting — which caused the v5 regression.

| Function | Purpose | Example output |
|----------|---------|---------------|
| `format_author_string(lastnames)` | Author formatting | `""`, `"Smith"`, `"Smith and Jones"`, `"Smith et al."` |
| `format_paper_embedding_text(...)` | Paper embedding text | `"Title. Smith and Jones. 2023. Nature. Abstract text"` |
| `format_paper_metadata_prefix(...)` | Chunk context prefix | `"Title. Smith and Jones. 2023. Nature"` |
| `format_passage_embedding_text(chunk, prefix)` | Passage embedding text | `"Title. Smith. 2023\n\nParagraph text..."` |
| `strip_metadata_prefix(text)` | Inverse: extract abstract/passage from formatted text | `"Abstract text"` from `"Title. Smith. 2023. Abstract text"` |

**Delegation chain:**
- `Paper.to_embedding_text()` -> `format_paper_embedding_text()`
- `Chunk.to_embedding_text()` -> `format_passage_embedding_text()`
- `_build_paper_metadata_prefix()` (chunking.py) -> `format_paper_metadata_prefix()`
- `_format_positive()` (data_sources.py) -> `format_paper_embedding_text()`
- `contexts_to_training_examples()` wraps passages with `format_passage_embedding_text()`
- `parse_fulltext_record()` (fulltext_passages.py) uses `format_paper_metadata_prefix()` for chunk context

**NEVER:**
1. Use `[SEP]` as separator — always `. ` (period-space) via canonical functions
2. Format embedding text with raw string concatenation — call the canonical function
3. Use bare chunk text as training positive — wrap with `format_passage_embedding_text()`
4. Use different author formatting in different paths
5. Compare hard negatives to positives with exact string match only — use `strip_metadata_prefix()` for format-aware dedup

**Format-aware hard negative dedup:** All 5 dedup points (`merge_training_data.py`, `augment_hard_negatives.py` x2, `data_preparation.py` x2) compare stripped core text via `strip_metadata_prefix()`.

**Regression tests:** `tests/test_embedding_format_consistency.py` verifies all paths produce identical output.

## Data Formats

**corpus.jsonl**: `{id, title, abstract, authors, year, doi, bibtex_key, journal, llm_description}`
**test_set.jsonl**: `{id, source_paper_id, local_context, narrow_context, broad_context, section_context, global_context, ground_truth_ids, reference_set_ids}`
**Training data**: `data/finetuning/master_{train,dev,eval}.jsonl` — 64K/12.5K/3K examples. Rebuilt by `scripts/merge_training_data.py [--force]`.

## Environment Variables (`.env`)

- `SEMANTIC_SCHOLAR_API_KEY` — S2 API calls
- `OPENALEX_EMAIL` — OpenAlex/Unpaywall polite pool
- `ANTHROPIC_API_KEY` — LLM enrichment (Claude Haiku)

## Design Principles

- **NEVER** hard-coded filter lists; **RARELY** regexes; prefer ML/NLP
- Data cleaning (removing markup) is fine; content filtering is not
- Author matching at retrieval time via `author_boost` param (NER was too noisy)
- MPS auto-detected via `get_best_device()`. Falls back to CUDA > CPU.

## API & Plugins

**REST API** (`api.py`): `incite serve --embedder minilm-ft` at localhost:8230. The `--embedder` flag **must match** the FAISS index embedder. Mismatched embedders cause full re-embedding (~18 min). Endpoints: /health, /stats, /config, /recommend, /batch. Docs at /docs.

**Webapp**: `pip install -e ".[webapp]" && incite webapp` (localhost:8501)

**Editor plugins** (`editor-plugins/`): All share `@incite/shared` (types, API client, context extraction). **Obsidian**: `cd editor-plugins/obsidian-incite && npm run build`. **VS Code**: Sidebar + `Cmd+Shift+C`. **Google Docs**: Apps Script sidebar (`clasp push --force`). Known issue: `PERMISSION_DENIED` with multiple Google accounts — use Incognito. **Chrome hotkey**: Option+Shift+M for Google Docs sidebar.

## Training Data

Master files: `data/finetuning/master_{train,dev,eval}.jsonl` — 64K/12.5K/3K examples across 7 sources. Rebuilt by `python scripts/merge_training_data.py [--force]`. Quality filtering: s2orc_abstract dropped, `[SEP]` auto-repaired, LLM refusals removed, deduped, short/low-similarity filtered, train/dev leakage removed.

**TrainingExample schema**: `positive` formatted via canonical functions (`format_paper_embedding_text()` for paper-level, `format_passage_embedding_text()` for passage-level). `hard_negatives` uses the same formatting. Optional passage metadata: `passage_positive`, `passage_score`, `passage_validation`, `intent`, `passage_section`, `passage_hard_negatives`.

## Chunking System

Two strategies: **Paragraph-based** (default, PyMuPDF + font-size heuristics) and **GROBID-based** (ML, requires Docker, ~90% accuracy). Start GROBID: `docker compose up -d grobid`. Files: `corpus/grobid.py`, `corpus/grobid_chunking.py`. Strategies registered in `retrieval/factory.py:CHUNKING_STRATEGIES`.

## RunPod GPU Training

Full details in `docs/RUNPOD.md`. SSH pattern, pod creation, known issues, and training performance lessons.
