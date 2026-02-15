# Testing

## Framework
Use pytest. Run with: `pytest` (from project root).

## Test Organization
- Unit tests in `tests/` mirroring `src/incite/` structure
- Use `pytest.mark` for categorization (unit, integration, slow)
- Mark slow tests (embedding, FAISS, API calls) with `@pytest.mark.slow`

## Workflow
1. Write test first when fixing bugs (reproduce the bug)
2. Run existing tests before and after changes: `pytest`
3. For ML/evaluation changes, also run: `incite evaluate --method hybrid`

## Coverage
```bash
pytest --cov=src/incite --cov-report=term-missing
```

## Key Testing Patterns for This Project
- Mock external APIs (Semantic Scholar, OpenAlex, Voyage AI) in tests
- Use small fixture corpora, not full corpus.jsonl
- Evaluation metrics (MRR, R@k) are the ground truth for retrieval changes
- Cache-related tests must respect the chunk cache safety rules in CLAUDE.md
