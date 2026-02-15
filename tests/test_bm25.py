"""Tests for BM25 retrieval."""

import pytest

from incite.models import Paper, RetrievalResult
from incite.retrieval.bm25 import BM25Retriever, tokenize_with_stopwords, STOPWORDS


class TestTokenization:
    def test_basic_tokenization(self):
        tokens = tokenize_with_stopwords("Climate change impacts on sea level rise")
        assert "climate" in tokens
        assert "change" in tokens
        assert "sea" in tokens
        assert "level" in tokens
        assert "rise" in tokens
        # "on" is a stopword
        assert "on" not in tokens

    def test_stopword_removal(self):
        tokens = tokenize_with_stopwords("the quick brown fox and the lazy dog")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_academic_stopwords(self):
        tokens = tokenize_with_stopwords("This paper propose a method for analysis")
        assert "paper" not in tokens
        assert "propose" not in tokens
        assert "method" not in tokens
        assert "analysis" in tokens

    def test_lowercasing(self):
        tokens = tokenize_with_stopwords("MACHINE Learning Models")
        assert "machine" in tokens
        assert "learning" in tokens
        assert "models" in tokens

    def test_punctuation_removal(self):
        tokens = tokenize_with_stopwords("Smith et al. (2020) found that CO2 levels...")
        assert "smith" in tokens
        assert "2020" in tokens
        assert "co2" in tokens
        assert "levels" in tokens
        # Punctuation artifacts should not appear
        assert "." not in tokens
        assert "(" not in tokens

    def test_hyphenated_words_split(self):
        tokens = tokenize_with_stopwords("self-attention mechanism for state-of-the-art results")
        assert "self" in tokens
        assert "attention" in tokens
        assert "mechanism" in tokens
        assert "state" in tokens
        assert "art" in tokens

    def test_short_tokens_filtered(self):
        tokens = tokenize_with_stopwords("A B C deep learning")
        # Single-char tokens should be removed
        assert "a" not in tokens  # also a stopword
        assert "deep" in tokens
        assert "learning" in tokens

    def test_empty_input(self):
        assert tokenize_with_stopwords("") == []
        assert tokenize_with_stopwords("   ") == []


class TestBM25Retriever:
    @pytest.fixture
    def papers(self):
        return [
            Paper(id="p1", title="Sea level rise projections", abstract="Global sea levels are rising due to climate change"),
            Paper(id="p2", title="Deep learning for NLP", abstract="Neural networks for natural language processing"),
            Paper(id="p3", title="Ocean temperature trends", abstract="Sea surface temperatures and oceanic heat content"),
        ]

    @pytest.fixture
    def retriever(self, papers):
        return BM25Retriever.from_papers(papers)

    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve("sea level rise", k=3)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_relevance(self, retriever):
        results = retriever.retrieve("sea level rise", k=3)
        # "sea level rise" should match p1 best
        paper_ids = [r.paper_id for r in results]
        assert "p1" in paper_ids

    def test_retrieve_scores_descending(self, retriever):
        results = retriever.retrieve("climate change ocean", k=3)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_retrieve_ranks_assigned(self, retriever):
        results = retriever.retrieve("neural networks", k=3)
        for i, result in enumerate(results):
            assert result.rank == i + 1

    def test_retrieve_with_timing(self, retriever):
        results, timing = retriever.retrieve("sea level", k=3, return_timing=True)
        assert isinstance(results, list)
        assert isinstance(timing, dict)
        assert "bm25_search_ms" in timing
        assert timing["bm25_search_ms"] >= 0

    def test_retrieve_k_limits_results(self, retriever):
        results = retriever.retrieve("ocean temperature climate", k=1)
        assert len(results) <= 1

    def test_score_breakdown_has_bm25(self, retriever):
        results = retriever.retrieve("sea level", k=3)
        for result in results:
            assert "bm25" in result.score_breakdown

    def test_from_papers_class_method(self, papers):
        retriever = BM25Retriever.from_papers(papers)
        assert len(retriever.paper_ids) == 3
        assert "p1" in retriever.papers

    def test_zero_score_results_excluded(self, retriever):
        # A very specific query unlikely to match all docs
        results = retriever.retrieve("xyznonexistentterm", k=3)
        for result in results:
            assert result.score > 0
