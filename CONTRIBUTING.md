# Contributing to inCite

## 1. Welcome

Thank you for your interest in contributing to inCite, a local-first citation recommendation system. We appreciate contributions of all kinds: bug fixes, new features, documentation improvements, and issue reports.

inCite is licensed under the Apache 2.0 License. By contributing, you agree that your contributions will be licensed under the same terms.

## 2. Development Setup

```bash
git clone https://github.com/galenphall/incite.git
cd incite
pip install -e ".[dev]"
pytest
```

This installs inCite in editable mode with all development dependencies and runs the test suite to verify everything works.

## 3. Code Style

- Follow PEP 8 conventions.
- Use ruff for linting and formatting: `ruff check src/incite && ruff format src/incite`
- Use type annotations on all public function signatures.
- Line length: 100 characters.
- Keep files under 800 lines. Extract utilities when a file grows beyond that.
- Keep functions under 50 lines. Avoid deep nesting (more than 4 levels).
- Prefer immutable data structures where practical (`@dataclass(frozen=True)`, `NamedTuple`).
- No hardcoded values -- use constants or configuration.

## 4. Testing

- Run the full test suite: `pytest`
- Mark slow tests (embedding, FAISS, API calls) with `@pytest.mark.slow`.
- When fixing a bug, write a test that reproduces it first, then fix the code.
- Mock external APIs (Semantic Scholar, OpenAlex, Voyage AI) in tests.
- Use small fixture data, not full corpus files.
- For ML/evaluation changes, also run: `incite evaluate --method hybrid`

## 5. How to Add a New Embedder

1. Create a new class extending `BaseEmbedder` (defined in `src/incite/embeddings/base.py`) in a file under `src/incite/embeddings/`.
2. Implement the required interface:
   - `embed(texts: list[str]) -> np.ndarray` -- batch embedding
   - `embed_query(query: str) -> np.ndarray` -- single query embedding
   - `dimension` property -- returns the embedding dimensionality
3. Register the embedder in the `EMBEDDERS` dict in `src/incite/retrieval/factory.py`.
4. Add tests for the new embedder.
5. Benchmark against existing embedders: `incite evaluate --method hybrid`

## 6. Architecture Overview

The codebase is organized by feature/domain:

| Path | Purpose |
|------|---------|
| `src/incite/models.py` | Data models: `Paper`, `Chunk`, `CitationContext`, `RetrievalResult`, `TrainingExample` |
| `src/incite/interfaces.py` | ABCs and Protocols: `Retriever`, `VectorStore`, `CorpusSource`, `BaseEmbedder` |
| `src/incite/embeddings/` | Embedder implementations and FAISS vector stores |
| `src/incite/retrieval/` | Neural, BM25, and hybrid retrievers, plus the factory for creating them |
| `src/incite/corpus/` | Data ingestion: Zotero reader, PDF extraction, chunking, enrichment |
| `src/incite/evaluation/` | Metrics (MRR, R@k, C-index), experiment logging, significance tests |
| `src/incite/cli/` | CLI subcommands (Click-based) |
| `src/incite/finetuning/` | Training data generation and model training pipeline |
| `editor-plugins/` | Editor integrations (Obsidian, VS Code, Google Docs, Chrome) |
| `cloud/` | Cloud service: multi-user web tier and GROBID processing |

## 7. Canonical Embedding Text Format

All code that formats text for embedding **must** use these functions defined in `src/incite/models.py`:

| Function | Purpose |
|----------|---------|
| `format_paper_embedding_text()` | Paper-level embedding text |
| `format_passage_embedding_text()` | Passage-level embedding text |
| `format_paper_metadata_prefix()` | Chunk context prefix |
| `format_author_string()` | Consistent author formatting |

Never use raw string concatenation to build embedding text. Never use `[SEP]` as a separator -- always use period-space (`. `) via the canonical functions. Inconsistent formatting causes train/eval distribution shift and degrades retrieval quality.

## 8. Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Write tests for new features and bug fixes.
3. Ensure `pytest` passes and `ruff check src/incite` reports no issues.
4. Keep PRs focused: one feature or fix per PR.
5. Write a clear PR description explaining **why** the change is needed, not just what changed.
6. If your change affects retrieval quality, include evaluation results (`incite evaluate --method hybrid`).

## 9. Reporting Issues

Use [GitHub Issues](https://github.com/galenphall/incite/issues) to report bugs or request features.

For bug reports, please include:

- What you expected to happen.
- What actually happened (include error messages or tracebacks).
- Steps to reproduce the issue.
- Your Python version and OS.

For performance issues, also include:

- Which embedder you are using (e.g., `minilm-ft`, `granite-ft`).
- Your corpus size (number of papers).
- Whether you are using hybrid or neural-only retrieval.

## 10. Code of Conduct

We are committed to providing a welcoming and respectful environment for everyone. Contributors are expected to:

- Be respectful and constructive in all interactions.
- Accept constructive criticism gracefully.
- Focus on what is best for the project and its users.
- Show empathy toward other contributors.

Harassment, trolling, and deliberately exclusionary behavior will not be tolerated. Maintainers reserve the right to remove, edit, or reject contributions that do not align with these standards.
