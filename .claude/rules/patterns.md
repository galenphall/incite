# Design Patterns

## Protocol / ABC Pattern
This project uses ABCs and Protocols extensively (interfaces.py):
- `BaseEmbedder` ABC for all embedder implementations
- `VectorStore` Protocol for FAISS stores
- `CorpusSource` Protocol for data sources
- `Retriever` ABC for retrieval strategies
- Follow these patterns when adding new components

## Dataclasses as Models
All data models live in `models.py`:
- `Paper`, `Chunk`, `CitationContext`, `RetrievalResult`, `TrainingExample`
- Use `@dataclass` with type annotations
- Include `from_dict()` / `to_dict()` for serialization

## Factory Pattern
- `retrieval/factory.py` is the entry point for creating retrievers and embedders
- `EMBEDDERS` dict maps string keys to embedder constructors
- New embedders must be registered here

## Context Managers
- Use `with` for file I/O, HTTP sessions, temp directories
- The acquire module uses context-managed sessions with retry logic

## Configuration
- CLI args via Click (cli/ subcommands)
- Runtime config in `~/.incite/config.json`
- Env vars for secrets (.env with python-dotenv)
- Never hardcode API keys or file paths
