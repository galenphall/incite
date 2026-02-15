"""Corpus loading and processing."""

from incite.corpus.enrichment import (
    BibTeXParser,
    MetadataEnricher,
    enrich_bibtex_to_corpus,
)
from incite.corpus.loader import load_corpus, save_corpus
from incite.corpus.openalex import OpenAlexClient
from incite.corpus.paperpile_source import PaperpileSource, find_paperpile_pdfs
from incite.corpus.semantic_scholar import SemanticScholarClient
from incite.corpus.zotero_reader import (
    find_zotero_data_dir,
    get_library_stats,
    read_zotero_library,
)

__all__ = [
    "load_corpus",
    "save_corpus",
    "SemanticScholarClient",
    "OpenAlexClient",
    "BibTeXParser",
    "MetadataEnricher",
    "enrich_bibtex_to_corpus",
    "find_zotero_data_dir",
    "read_zotero_library",
    "get_library_stats",
    "PaperpileSource",
    "find_paperpile_pdfs",
]
