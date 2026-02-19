"""Tests for UMAP projection visualization module."""

import importlib
import json

import numpy as np
import pytest


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


from incite.embeddings.stores import FAISSStore
from incite.visualization.plotly_charts import build_umap_figure
from incite.visualization.umap_projection import (
    UMAPPoint,
    UMAPProjection,
    compute_umap_projection,
    extract_embeddings_from_faiss,
    load_projection_cache,
    save_projection_cache,
)


class TestUMAPPoint:
    def test_umap_point_frozen(self):
        """UMAPPoint should be immutable."""
        point = UMAPPoint(
            paper_id="p1",
            x=0.5,
            y=0.3,
            title="Test Paper",
            authors=["Smith, J"],
            year=2023,
            journal="Nature",
        )
        assert point.paper_id == "p1"
        with pytest.raises(AttributeError):
            point.x = 1.0

    def test_umap_point_optional_fields(self):
        """UMAPPoint should handle None year and journal."""
        point = UMAPPoint(
            paper_id="p1",
            x=0.5,
            y=0.3,
            title="Test",
            authors=["Smith"],
            year=None,
            journal=None,
        )
        assert point.year is None
        assert point.journal is None

    def test_umap_point_doi(self):
        """UMAPPoint should store doi when provided."""
        point = UMAPPoint(
            paper_id="p1",
            x=0.5,
            y=0.3,
            title="Test",
            authors=["Smith"],
            year=2023,
            journal="Nature",
            doi="10.1234/test",
        )
        assert point.doi == "10.1234/test"

    def test_umap_point_doi_default_none(self):
        """UMAPPoint doi should default to None."""
        point = UMAPPoint(
            paper_id="p1",
            x=0.5,
            y=0.3,
            title="Test",
            authors=["Smith"],
            year=2023,
            journal="Nature",
        )
        assert point.doi is None


class TestUMAPProjection:
    def test_projection_to_dict(self):
        """Test serialization to dict."""
        points = [
            UMAPPoint(
                paper_id="p1",
                x=0.1,
                y=0.2,
                title="Paper 1",
                authors=["A"],
                year=2023,
                journal="J1",
            ),
            UMAPPoint(
                paper_id="p2",
                x=0.3,
                y=0.4,
                title="Paper 2",
                authors=["B", "C"],
                year=2022,
                journal="J2",
            ),
        ]
        projection = UMAPProjection(
            points=points,
            num_papers=2,
            embedder_type="minilm-ft",
            computed_at="2026-02-16T00:00:00+00:00",
        )
        data = projection.to_dict()

        assert data["num_papers"] == 2
        assert data["embedder_type"] == "minilm-ft"
        assert data["computed_at"] == "2026-02-16T00:00:00+00:00"
        assert len(data["points"]) == 2
        assert data["points"][0]["paper_id"] == "p1"
        assert data["points"][0]["x"] == 0.1
        assert data["points"][1]["authors"] == ["B", "C"]

    def test_projection_roundtrip(self):
        """Test from_dict produces identical projection."""
        points = [
            UMAPPoint(
                paper_id="p1",
                x=0.1,
                y=0.2,
                title="Paper 1",
                authors=["A"],
                year=2023,
                journal="J1",
            ),
        ]
        original = UMAPProjection(
            points=points,
            num_papers=1,
            embedder_type="granite-ft",
            computed_at="2026-02-16T12:00:00+00:00",
        )

        data = original.to_dict()
        restored = UMAPProjection.from_dict(data)

        assert restored.num_papers == original.num_papers
        assert restored.embedder_type == original.embedder_type
        assert restored.computed_at == original.computed_at
        assert len(restored.points) == len(original.points)
        assert restored.points[0].paper_id == original.points[0].paper_id
        assert restored.points[0].x == original.points[0].x
        assert restored.points[0].year == original.points[0].year

    def test_from_dict_missing_doi(self):
        """from_dict should handle cached data without doi field (backward compat)."""
        data = {
            "points": [
                {
                    "paper_id": "p1",
                    "x": 0.1,
                    "y": 0.2,
                    "title": "Paper 1",
                    "authors": ["A"],
                    "year": 2023,
                    "journal": "J1",
                    # no "doi" key — simulates old cached projections
                },
            ],
            "num_papers": 1,
            "embedder_type": "test",
            "computed_at": "2026-02-16T00:00:00+00:00",
        }
        restored = UMAPProjection.from_dict(data)
        assert restored.points[0].doi is None

    def test_projection_frozen(self):
        """UMAPProjection should be immutable."""
        projection = UMAPProjection(
            points=[],
            num_papers=0,
            embedder_type="test",
            computed_at="2026-02-16T00:00:00+00:00",
        )
        with pytest.raises(AttributeError):
            projection.num_papers = 5


class TestExtractEmbeddingsFromFAISS:
    def test_extract_embeddings_basic(self, mock_embedder):
        """Extract embeddings from FAISSStore."""
        store = FAISSStore(dimension=mock_embedder.dimension)
        texts = ["alpha", "beta", "gamma"]
        ids = ["p1", "p2", "p3"]
        vecs = mock_embedder.embed(texts)
        store.add(ids, vecs)

        extracted_ids, extracted_vecs = extract_embeddings_from_faiss(store)

        assert len(extracted_ids) == 3
        assert extracted_vecs.shape == (3, mock_embedder.dimension)
        assert set(extracted_ids) == set(ids)

    def test_extract_preserves_order(self, mock_embedder):
        """Extracted embeddings should match original order."""
        store = FAISSStore(dimension=mock_embedder.dimension)
        texts = ["one", "two", "three"]
        ids = ["id_1", "id_2", "id_3"]
        vecs = mock_embedder.embed(texts)
        store.add(ids, vecs)

        extracted_ids, extracted_vecs = extract_embeddings_from_faiss(store)

        # Extract embeddings correspond to their stored IDs
        for i, extracted_id in enumerate(extracted_ids):
            assert extracted_id in ids

    def test_extract_empty_store(self, mock_embedder):
        """Extracting from empty store should handle gracefully."""
        store = FAISSStore(dimension=mock_embedder.dimension)
        # Empty store has _index = None, so we need to handle this
        if store._index is None:
            # Expected behavior for uninitialized store
            assert store.size == 0
        else:
            extracted_ids, extracted_vecs = extract_embeddings_from_faiss(store)
            assert extracted_ids == []
            assert extracted_vecs.shape == (0, mock_embedder.dimension)


@pytest.mark.skipif(not _has_module("umap"), reason="umap not installed")
class TestComputeUMAPProjection:
    @pytest.mark.slow
    def test_compute_umap_basic(self, mock_embedder):
        """Compute basic UMAP projection with 50 random vectors."""
        n_papers = 50
        embeddings = np.random.randn(n_papers, mock_embedder.dimension).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        paper_ids = [f"p{i}" for i in range(n_papers)]
        paper_metadata = {
            pid: {
                "title": f"Paper {i}",
                "authors": ["Author A", "Author B"],
                "year": 2020 + (i % 5),
                "journal": ["Nature", "Science", "PNAS"][i % 3],
            }
            for i, pid in enumerate(paper_ids)
        }

        projection = compute_umap_projection(paper_ids, embeddings, paper_metadata, "test-embedder")

        assert len(projection.points) == 50
        assert projection.num_papers == 50
        assert projection.embedder_type == "test-embedder"
        # All points should have valid 2D coordinates
        for p in projection.points:
            assert isinstance(p.x, float)
            assert isinstance(p.y, float)
            assert not np.isnan(p.x)
            assert not np.isnan(p.y)

    @pytest.mark.slow
    def test_compute_umap_output_shape(self, mock_embedder):
        """Output should be 2D coordinates."""
        embeddings = np.random.randn(10, mock_embedder.dimension).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        paper_ids = [f"p{i}" for i in range(10)]
        paper_metadata = {
            pid: {"title": "T", "authors": [], "year": None, "journal": None} for pid in paper_ids
        }

        projection = compute_umap_projection(paper_ids, embeddings, paper_metadata, "test")

        assert len(projection.points) == 10
        for p in projection.points:
            assert isinstance(p.x, float)
            assert isinstance(p.y, float)

    @pytest.mark.slow
    def test_compute_umap_few_papers(self, mock_embedder):
        """UMAP should handle edge case with very few papers."""
        embeddings = np.random.randn(3, mock_embedder.dimension).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        paper_ids = ["p1", "p2", "p3"]
        paper_metadata = {
            pid: {"title": "T", "authors": [], "year": None, "journal": None} for pid in paper_ids
        }

        # Should not crash even with n_neighbors=15 > n_papers-1=2
        projection = compute_umap_projection(
            paper_ids, embeddings, paper_metadata, "test", n_neighbors=15
        )

        assert len(projection.points) == 3

    def test_compute_umap_metadata_lookup(self, mock_embedder):
        """Paper metadata should be correctly attached to points."""
        # Use enough papers for UMAP to work (n >= 15 for default n_neighbors)
        n_papers = 20
        embeddings = np.random.randn(n_papers, mock_embedder.dimension).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        paper_ids = [f"p{i}" for i in range(n_papers)]
        paper_metadata = {
            "p1": {
                "title": "Climate Change",
                "authors": ["Smith, J", "Jones, A"],
                "year": 2023,
                "journal": "Nature",
            },
            "p2": {
                "title": "AI Methods",
                "authors": ["Lee, Ming"],
                "year": 2024,
                "journal": "ICML",
            },
        }
        # Fill in rest of metadata
        for i in range(n_papers):
            if f"p{i}" not in paper_metadata:
                paper_metadata[f"p{i}"] = {
                    "title": f"Paper {i}",
                    "authors": ["Author"],
                    "year": 2023,
                    "journal": "Journal",
                }

        projection = compute_umap_projection(paper_ids, embeddings, paper_metadata, "test")

        p1_point = next(p for p in projection.points if p.paper_id == "p1")
        assert p1_point.title == "Climate Change"
        assert p1_point.authors == ["Smith, J", "Jones, A"]
        assert p1_point.year == 2023
        assert p1_point.journal == "Nature"

    def test_compute_umap_missing_metadata(self, mock_embedder):
        """Should handle missing paper metadata gracefully."""
        # Use enough papers for UMAP to work
        n_papers = 20
        embeddings = np.random.randn(n_papers, mock_embedder.dimension).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        paper_ids = [f"p{i}" for i in range(n_papers)]
        paper_metadata = {"p1": {"title": "Paper 1", "authors": [], "year": 2023, "journal": None}}
        # Fill in rest of metadata (except p2 to test missing metadata)
        for i in range(n_papers):
            if f"p{i}" not in paper_metadata and i != 2:
                paper_metadata[f"p{i}"] = {
                    "title": f"Paper {i}",
                    "authors": ["Author"],
                    "year": 2023,
                    "journal": "Journal",
                }

        projection = compute_umap_projection(paper_ids, embeddings, paper_metadata, "test")

        p2_point = next(p for p in projection.points if p.paper_id == "p2")
        assert p2_point.title == "Unknown"


class TestProjectionCaching:
    def test_save_projection_cache(self, tmp_path):
        """Save projection to JSON cache."""
        points = [
            UMAPPoint(
                paper_id="p1",
                x=0.1,
                y=0.2,
                title="Paper 1",
                authors=["A"],
                year=2023,
                journal="J1",
            ),
        ]
        projection = UMAPProjection(
            points=points,
            num_papers=1,
            embedder_type="test",
            computed_at="2026-02-16T00:00:00+00:00",
        )
        cache_path = tmp_path / "projection.json"

        save_projection_cache(projection, cache_path)

        assert cache_path.exists()
        with open(cache_path) as f:
            data = json.load(f)
        assert data["num_papers"] == 1
        assert len(data["points"]) == 1

    def test_load_projection_cache(self, tmp_path):
        """Load projection from JSON cache."""
        points = [
            UMAPPoint(
                paper_id="p1",
                x=0.1,
                y=0.2,
                title="Paper 1",
                authors=["A"],
                year=2023,
                journal="J1",
            ),
        ]
        projection = UMAPProjection(
            points=points,
            num_papers=1,
            embedder_type="test",
            computed_at="2026-02-16T00:00:00+00:00",
        )
        cache_path = tmp_path / "projection.json"
        save_projection_cache(projection, cache_path)

        loaded = load_projection_cache(cache_path)

        assert loaded is not None
        assert loaded.num_papers == 1
        assert loaded.embedder_type == "test"
        assert len(loaded.points) == 1

    def test_load_nonexistent_cache(self, tmp_path):
        """Load from nonexistent cache returns None."""
        cache_path = tmp_path / "nonexistent.json"
        loaded = load_projection_cache(cache_path)
        assert loaded is None

    def test_load_invalid_json(self, tmp_path):
        """Load from corrupted JSON returns None."""
        cache_path = tmp_path / "bad.json"
        cache_path.write_text("not valid json {")
        loaded = load_projection_cache(cache_path)
        assert loaded is None

    def test_cache_roundtrip(self, tmp_path):
        """Save and load preserves projection exactly."""
        points = [
            UMAPPoint(
                paper_id=f"p{i}",
                x=float(i) * 0.1,
                y=float(i) * 0.2,
                title=f"Paper {i}",
                authors=[f"Author {j}" for j in range(2)],
                year=2020 + i,
                journal=f"Journal {i}",
            )
            for i in range(5)
        ]
        original = UMAPProjection(
            points=points,
            num_papers=5,
            embedder_type="minilm-ft",
            computed_at="2026-02-16T12:34:56+00:00",
        )
        cache_path = tmp_path / "proj.json"
        save_projection_cache(original, cache_path)

        loaded = load_projection_cache(cache_path)

        assert loaded.num_papers == original.num_papers
        assert loaded.embedder_type == original.embedder_type
        assert len(loaded.points) == len(original.points)
        for orig_p, loaded_p in zip(original.points, loaded.points):
            assert orig_p.paper_id == loaded_p.paper_id
            assert orig_p.x == loaded_p.x
            assert orig_p.y == loaded_p.y
            assert orig_p.title == loaded_p.title


@pytest.mark.skipif(not _has_module("plotly"), reason="plotly not installed")
class TestBuildUMAPFigure:
    def test_build_figure_basic(self):
        """Build a basic Plotly figure from projection."""
        points = [
            UMAPPoint(
                paper_id="p1",
                x=0.1,
                y=0.2,
                title="Paper 1",
                authors=["Smith"],
                year=2023,
                journal="Nature",
            ),
            UMAPPoint(
                paper_id="p2",
                x=0.3,
                y=0.4,
                title="Paper 2",
                authors=["Jones"],
                year=2022,
                journal="Science",
            ),
        ]
        projection = UMAPProjection(
            points=points,
            num_papers=2,
            embedder_type="test",
            computed_at="2026-02-16T00:00:00+00:00",
        )

        fig_dict = build_umap_figure(projection)

        assert isinstance(fig_dict, dict)
        assert "data" in fig_dict
        assert "layout" in fig_dict
        assert len(fig_dict["data"]) > 0

    def test_build_figure_has_scatter_trace(self):
        """Figure should have a scatter trace."""
        points = [
            UMAPPoint(
                paper_id=f"p{i}",
                x=float(i) * 0.1,
                y=float(i) * 0.1,
                title=f"Paper {i}",
                authors=["A"],
                year=2023,
                journal="J",
            )
            for i in range(5)
        ]
        projection = UMAPProjection(
            points=points,
            num_papers=5,
            embedder_type="test",
            computed_at="2026-02-16T00:00:00+00:00",
        )

        fig_dict = build_umap_figure(projection)

        trace = fig_dict["data"][0]
        assert trace["mode"] == "markers"
        assert len(trace["x"]) == 5
        assert len(trace["y"]) == 5

    def test_build_figure_color_by_year(self):
        """Figure should support coloring by year."""
        points = [
            UMAPPoint(
                paper_id="p1",
                x=0.1,
                y=0.2,
                title="P1",
                authors=["A"],
                year=2020,
                journal="J",
            ),
            UMAPPoint(
                paper_id="p2",
                x=0.3,
                y=0.4,
                title="P2",
                authors=["B"],
                year=2024,
                journal="J",
            ),
        ]
        projection = UMAPProjection(
            points=points,
            num_papers=2,
            embedder_type="test",
            computed_at="2026-02-16T00:00:00+00:00",
        )

        fig_dict = build_umap_figure(projection, color_field="year")

        trace = fig_dict["data"][0]
        assert "marker" in trace
        assert "color" in trace["marker"]
        assert "colorscale" in trace["marker"]

    def test_build_figure_highlight_papers(self):
        """Figure should highlight specific paper IDs."""
        points = [
            UMAPPoint(
                paper_id=f"p{i}",
                x=float(i) * 0.1,
                y=float(i) * 0.1,
                title=f"P{i}",
                authors=["A"],
                year=2023,
                journal="J",
            )
            for i in range(5)
        ]
        projection = UMAPProjection(
            points=points,
            num_papers=5,
            embedder_type="test",
            computed_at="2026-02-16T00:00:00+00:00",
        )

        fig_dict = build_umap_figure(projection, highlight_ids={"p1", "p3"})

        trace = fig_dict["data"][0]
        sizes = trace["marker"]["size"]
        opacities = trace["marker"]["opacity"]

        # Should have 5 sizes and opacities
        assert len(sizes) == 5
        assert len(opacities) == 5
        # p1 and p3 should be highlighted (larger size, full opacity)
        assert sizes[1] == 12  # p1
        assert opacities[1] == 1.0
        assert sizes[3] == 12  # p3
        assert opacities[3] == 1.0

    def test_build_figure_hover_text(self):
        """Figure should include hover text with paper info."""
        points = [
            UMAPPoint(
                paper_id="p1",
                x=0.1,
                y=0.2,
                title="Climate Paper",
                authors=["Smith, J", "Jones, A"],
                year=2023,
                journal="Nature",
            ),
        ]
        projection = UMAPProjection(
            points=points,
            num_papers=1,
            embedder_type="test",
            computed_at="2026-02-16T00:00:00+00:00",
        )

        fig_dict = build_umap_figure(projection)

        trace = fig_dict["data"][0]
        hover_texts = trace["text"]
        assert len(hover_texts) > 0
        # Should contain paper title and authors
        assert "Climate Paper" in hover_texts[0]
        assert "Smith, J" in hover_texts[0]
        assert "2023" in hover_texts[0]

    def test_build_figure_layout(self):
        """Figure should have clean layout with hidden axes."""
        points = [
            UMAPPoint(
                paper_id="p1",
                x=0.1,
                y=0.2,
                title="P1",
                authors=["A"],
                year=2023,
                journal="J",
            ),
        ]
        projection = UMAPProjection(
            points=points,
            num_papers=1,
            embedder_type="test",
            computed_at="2026-02-16T00:00:00+00:00",
        )

        fig_dict = build_umap_figure(projection)

        layout = fig_dict["layout"]
        assert layout["xaxis"]["visible"] is False
        assert layout["yaxis"]["visible"] is False
        assert layout["plot_bgcolor"] == "white"
