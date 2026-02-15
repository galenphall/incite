# Python Coding Style

## Standards
- Follow PEP 8 conventions
- Use type annotations on public function signatures
- Use ruff for linting and formatting (not black/isort separately)

## Immutability
Prefer immutable data structures where practical:
- Use `@dataclass(frozen=True)` for value objects
- Use `NamedTuple` for lightweight records
- Avoid mutating function arguments; return new objects instead

## File Organization
- Keep files under 800 lines; extract utilities when larger
- Organize by feature/domain (corpus/, embeddings/, retrieval/), not by type
- This project already follows this pattern — maintain it

## Error Handling
- Handle errors explicitly; never silently swallow exceptions
- Use structured logging over print statements
- Validate at system boundaries (CLI args, API input, external data)
- Internal code can trust its own types and invariants

## Code Quality
- Functions under 50 lines
- No deep nesting (>4 levels)
- No hardcoded values — use constants or config
- Avoid over-engineering: don't add abstractions for one-time operations
