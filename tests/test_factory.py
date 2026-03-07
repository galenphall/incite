"""Tests for factory validation (registry structure, create_retriever)."""

import pytest

from incite.models import Paper
from incite.retrieval.factory import (
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNKING,
    DEFAULT_EMBEDDER,
    EMBEDDERS,
    create_retriever,
    get_chunker,
    get_embedder,
)


class TestEmbedderRegistry:
    def test_all_have_required_keys(self):
        for key, config in EMBEDDERS.items():
            assert "name" in config, f"{key} missing 'name'"
            assert "class" in config, f"{key} missing 'class'"
            assert "module" in config, f"{key} missing 'module'"

    def test_default_embedder_exists(self):
        assert DEFAULT_EMBEDDER in EMBEDDERS

    def test_all_keys_lowercase(self):
        for key in EMBEDDERS:
            assert key == key.lower(), f"Key '{key}' is not lowercase"


class TestChunkingRegistry:
    def test_all_have_required_keys(self):
        for key, config in CHUNKING_STRATEGIES.items():
            assert "name" in config, f"{key} missing 'name'"
            assert "function" in config, f"{key} missing 'function'"
            assert "module" in config, f"{key} missing 'module'"

    def test_default_chunking_exists(self):
        assert DEFAULT_CHUNKING in CHUNKING_STRATEGIES


class TestGetEmbedder:
    def test_unknown_raises_valueerror(self):
        with pytest.raises(ValueError):
            get_embedder("nonexistent_embedder")

    def test_error_message_lists_available(self):
        with pytest.raises(ValueError, match="Available:"):
            get_embedder("nonexistent_embedder")


class TestGetChunker:
    def test_unknown_raises_valueerror(self):
        with pytest.raises(ValueError):
            get_chunker("nonexistent_strategy")

    def test_paragraph_returns_callable(self):
        chunker = get_chunker("paragraph")
        assert callable(chunker)


class TestCreateRetriever:
    @pytest.fixture
    def papers(self):
        return [
            Paper(id="p1", title="Sea level rise", abstract="Sea levels are rising"),
            Paper(id="p2", title="Deep learning", abstract="Neural networks for NLP"),
        ]

    def test_bm25_method(self, papers):
        from incite.retrieval.bm25 import BM25Retriever

        retriever = create_retriever(papers, method="bm25")
        assert isinstance(retriever, BM25Retriever)

    def test_neural_with_mock(self, monkeypatch, papers, mock_embedder):
        from incite.retrieval import factory
        from incite.retrieval.neural import NeuralRetriever

        monkeypatch.setattr(factory, "get_embedder", lambda *a, **kw: mock_embedder)
        retriever = create_retriever(papers, method="neural", show_progress=False)
        assert isinstance(retriever, NeuralRetriever)

    def test_hybrid_with_mock(self, monkeypatch, papers, mock_embedder):
        from incite.retrieval import factory
        from incite.retrieval.hybrid import HybridRetriever

        monkeypatch.setattr(factory, "get_embedder", lambda *a, **kw: mock_embedder)
        retriever = create_retriever(papers, method="hybrid", show_progress=False)
        assert isinstance(retriever, HybridRetriever)
        assert retriever.rrf_k == 5

    def test_hybrid_uses_stemmed_bm25(self, monkeypatch, papers, mock_embedder):
        from incite.retrieval import factory
        from incite.retrieval.bm25 import tokenize_with_stemming
        from incite.retrieval.hybrid import HybridRetriever

        monkeypatch.setattr(factory, "get_embedder", lambda *a, **kw: mock_embedder)
        retriever = create_retriever(papers, method="hybrid", show_progress=False)
        assert isinstance(retriever, HybridRetriever)
        # The second retriever in the list should be BM25 with stemming tokenizer
        _, bm25_weight = retriever.retrievers[1]
        bm25 = retriever.retrievers[1][0]
        assert bm25.tokenizer is tokenize_with_stemming

    def test_loads_existing_index(self, monkeypatch, papers, mock_embedder, tmp_path):
        from incite.retrieval import factory
        from incite.retrieval.neural import NeuralRetriever

        monkeypatch.setattr(factory, "get_embedder", lambda *a, **kw: mock_embedder)

        # Build and save an index first
        neural = NeuralRetriever.from_papers(papers, mock_embedder, show_progress=False)
        neural.store.save(str(tmp_path / "test_index"))

        # create_retriever should load from the saved path
        retriever = create_retriever(
            papers, method="neural", index_path=tmp_path / "test_index", show_progress=False
        )
        assert isinstance(retriever, NeuralRetriever)
        assert retriever.store.size == len(papers)

    def test_multi_scale_retriever(self, monkeypatch, mock_embedder, tmp_path):
        """Test building and creating multi-scale retriever."""
        from incite.retrieval import factory
        from incite.retrieval.factory import (
            build_multi_scale_index,
            create_multi_scale_retriever,
        )
        from incite.retrieval.multi_scale import MultiScaleRetriever

        # Create papers with enough text to generate chunks
        papers_long = [
            Paper(
                id="p1", 
                title="Long Paper 1", 
                abstract="This is an abstract that is reasonably long but maybe not long enough for a chunk unless we pad it.",
                full_text=(
                    "This is the first sentence of the paper and it is quite long to ensure it passes length filters. "
                    "Here is a second sentence that adds more content to the paragraph. "
                    "We need to make sure there are enough sentences to be split correctly. "
                    "Finally, a fourth sentence concludes this paragraph with sufficient detail."
                )
            ),
            Paper(
                id="p2",
                title="Long Paper 2",
                abstract="Abstract 2.",
                full_text=(
                    "Another paper with sufficient length for testing purposes. "
                    "It also contains multiple sentences to verify the sentence chunking logic. "
                    "The quick brown fox jumps over the lazy dog in this standard test sentence. "
                    "Testing the multi-scale retriever requires robust test data."
                )
            )
        ]

        monkeypatch.setattr(factory, "get_embedder", lambda *a, **kw: mock_embedder)

        # 1. Build index
        build_multi_scale_index(
            papers=papers_long,
            output_dir=tmp_path,
            embedder_type="minilm",
            show_progress=False,
        )

        # Check files exist
        assert (tmp_path / "paper/index.faiss").exists()
        assert (tmp_path / "paragraph/index.faiss").exists()
        assert (tmp_path / "sentence/index.faiss").exists()
        assert (tmp_path / "chunks_paragraph.jsonl").exists()
        assert (tmp_path / "chunks_sentence.jsonl").exists()

        # 2. Create retriever
        retriever = create_multi_scale_retriever(
            papers=papers_long,
            base_dir=tmp_path,
            embedder_type="minilm",
            show_progress=False,
        )

        assert isinstance(retriever, MultiScaleRetriever)
        assert retriever.weights == {"paper": 1.0, "paragraph": 1.0, "sentence": 1.0}
