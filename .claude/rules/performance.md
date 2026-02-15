# Performance

## Embedding & FAISS
- Use MPS (Apple Silicon) when available via `get_best_device()`
- HNSW+SQ8 indexes for production; flat indexes for evaluation
- Two-stage Matryoshka search: coarse at 128-dim, rerank at 384-dim
- ONNX runtime for fast CPU inference (minilm-ft-onnx)

## Batch Processing
- Process embeddings in batches (default 64-128)
- Use vectorized TF-IDF for BM25 hard negatives (not per-query BM25)
- Use Anthropic Batch API for LLM-heavy data generation

## Context Window Management
- Keep CLAUDE.md focused and up-to-date (it loads every session)
- Avoid loading full corpus files into context; use streaming/sampling
- For large data operations, use scripts/ rather than inline Python

## Evaluation
- Primary metrics: MRR and R@1 (R@10/R@20 inflated by small ref sets)
- Corpus-size-adjusted metrics: C-index and Skill MRR normalize for varying reference set sizes
- Macro-averaging corrects for source-paper skew
- Bootstrap CIs and paired significance tests for comparing methods
