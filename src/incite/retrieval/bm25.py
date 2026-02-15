"""BM25 lexical retrieval."""

import re
import time
from typing import Union

from rank_bm25 import BM25Okapi

from incite.interfaces import Retriever
from incite.models import Paper, RetrievalResult

# Standard English stopwords plus common academic terms that don't help retrieval
STOPWORDS = frozenset(
    [
        # Standard English stopwords
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "also",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "would",
        "could",
        "ought",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "of",
        # Common academic filler words
        "paper",
        "study",
        "research",
        "work",
        "approach",
        "method",
        "methods",
        "propose",
        "proposed",
        "present",
        "presented",
        "show",
        "shows",
        "shown",
        "use",
        "used",
        "using",
        "based",
        "results",
        "result",
        "however",
        "therefore",
        "thus",
        "hence",
        "although",
        "while",
        "since",
        "because",
        "furthermore",
        "moreover",
        "additionally",
        "specifically",
        "particularly",
        "generally",
        "typically",
        "often",
        "usually",
        "recently",
        "previously",
        "first",
        "second",
        "third",
        "finally",
        "example",
        "examples",
        "e.g",
        "i.e",
        "et",
        "al",
        "etc",
        "fig",
        "figure",
        "table",
        "section",
    ]
)


def tokenize_with_stopwords(text: str) -> list[str]:
    """Tokenize with lowercasing and stopword removal.

    Preserves:
    - Scientific abbreviations and acronyms
    - Numbers (years, metrics)
    - Author names and technical terms
    """
    text = text.lower()
    # Replace punctuation with space, but preserve hyphens in compound terms
    text = re.sub(r"[^\w\s\-]", " ", text)
    # Split hyphenated words for better matching (e.g., "self-attention" -> ["self", "attention"])
    text = re.sub(r"-", " ", text)
    tokens = text.split()
    # Remove stopwords and very short tokens (likely noise)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


_snowball_stemmer = None


def _get_stemmer():
    """Get or create a cached SnowballStemmer instance."""
    global _snowball_stemmer
    if _snowball_stemmer is None:
        from nltk.stem import SnowballStemmer

        _snowball_stemmer = SnowballStemmer("english")
    return _snowball_stemmer


def tokenize_with_stemming(text: str) -> list[str]:
    """Tokenize with stopword removal + English stemming."""
    stemmer = _get_stemmer()
    tokens = tokenize_with_stopwords(text)
    return [stemmer.stem(t) for t in tokens]


class BM25Retriever(Retriever):
    """BM25 lexical retriever."""

    def __init__(
        self,
        papers: dict[str, Paper],
        bm25: BM25Okapi,
        paper_ids: list[str],
        tokenizer=None,
    ):
        """Initialize BM25 retriever.

        Args:
            papers: Dict mapping paper_id to Paper objects
            bm25: Initialized BM25 index
            paper_ids: List of paper IDs in same order as BM25 corpus
            tokenizer: Tokenization function (default: simple_tokenize)
        """
        self.papers = papers
        self.bm25 = bm25
        self.paper_ids = paper_ids
        self.tokenizer = tokenizer or tokenize_with_stopwords

    def retrieve(
        self,
        query: str,
        k: int = 10,
        return_timing: bool = False,
        **kwargs,
    ) -> Union[list[RetrievalResult], tuple[list[RetrievalResult], dict]]:
        """Retrieve top-k papers for a query.

        Args:
            query: Query text
            k: Number of results to return
            return_timing: If True, return (results, timing_dict)
            **kwargs: Ignored (for API compatibility with HybridRetriever)

        Returns:
            List of RetrievalResult objects sorted by score.
            If return_timing=True, returns (results, timing_dict).
        """
        timing = {}

        search_start = time.perf_counter()
        query_tokens = self.tokenizer(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = scores.argsort()[-k:][::-1]
        timing["bm25_search_ms"] = (time.perf_counter() - search_start) * 1000

        results = []
        for rank, idx in enumerate(top_indices):
            paper_id = self.paper_ids[idx]
            score = float(scores[idx])

            if score > 0:  # Only include non-zero scores
                results.append(
                    RetrievalResult(
                        paper_id=paper_id,
                        score=score,
                        rank=rank + 1,
                        score_breakdown={"bm25": score},
                    )
                )

        if return_timing:
            return results, timing
        return results

    @classmethod
    def from_papers(
        cls,
        papers: list[Paper],
        tokenizer=None,
        include_metadata: bool = True,
    ) -> "BM25Retriever":
        """Build retriever from a list of papers.

        Args:
            papers: List of Paper objects
            tokenizer: Tokenization function (default: tokenize_with_stopwords)
            include_metadata: Include author/year/journal in document text

        Returns:
            Initialized BM25Retriever
        """
        tokenizer = tokenizer or tokenize_with_stopwords
        paper_dict = {p.id: p for p in papers}
        paper_ids = [p.id for p in papers]

        # Create corpus
        corpus = []
        for paper in papers:
            text = paper.to_embedding_text(include_metadata=include_metadata)
            corpus.append(tokenizer(text))

        bm25 = BM25Okapi(corpus)

        return cls(
            papers=paper_dict,
            bm25=bm25,
            paper_ids=paper_ids,
            tokenizer=tokenizer,
        )
