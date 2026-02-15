# Security

## Secrets
- API keys in `.env` only (SEMANTIC_SCHOLAR_API_KEY, OPENALEX_EMAIL, ANTHROPIC_API_KEY)
- Never hardcode secrets in source code
- Use `os.environ` or `python-dotenv` for access
- Validate required secrets exist at startup with clear error messages

## Before Commits
- No hardcoded secrets, tokens, or API keys
- No `.env` files or credentials in version control
- No database connection strings in code

## Data Safety
- The chunk cache is expensive to rebuild (~$0.30-0.50 cloud processing)
- Always create .bak files before overwriting large caches
- Never bump CHUNK_CACHE_VERSION for filter-only changes (see CLAUDE.md)
- Refuse to overwrite >1000 chunks with <50% of original count

## External APIs
- Validate and sanitize all external API responses
- Use rate limiting and retries for S2, OpenAlex, Unpaywall
- PDF acquisition must go through proxy module (not direct downloads)
