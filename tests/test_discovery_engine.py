"""Tests for the discovery engine and models."""

from __future__ import annotations

from unittest.mock import MagicMock

from incite.discovery.models import DiscoveryCandidate


class TestDiscoveryCandidate:
    """Test the DiscoveryCandidate dataclass."""

    def test_signal_count_no_signals(self):
        c = DiscoveryCandidate(
            s2_id="abc", title="Test", authors=[], year=2023, doi=None, abstract=""
        )
        assert c.signal_count == 0

    def test_signal_count_all_signals(self):
        c = DiscoveryCandidate(
            s2_id="abc",
            title="Test",
            authors=["A"],
            year=2023,
            doi="10.1/x",
            abstract="text",
            citation_overlap=2,
            bib_coupling_score=0.5,
            semantic_score=0.8,
            author_overlap=1,
        )
        assert c.signal_count == 4

    def test_signal_count_thresholds(self):
        """Bib coupling < 0.1 and semantic < 0.4 don't count."""
        c = DiscoveryCandidate(
            s2_id="abc",
            title="Test",
            authors=[],
            year=2023,
            doi=None,
            abstract="",
            bib_coupling_score=0.05,
            semantic_score=0.3,
        )
        assert c.signal_count == 0

    def test_discovery_score_range(self):
        c = DiscoveryCandidate(
            s2_id="abc",
            title="Test",
            authors=[],
            year=2023,
            doi=None,
            abstract="",
            citation_overlap=5,
            bib_coupling_score=1.0,
            semantic_score=1.0,
            author_overlap=3,
        )
        assert 0.0 <= c.discovery_score <= 1.0

    def test_discovery_score_zero(self):
        c = DiscoveryCandidate(
            s2_id="abc", title="Test", authors=[], year=2023, doi=None, abstract=""
        )
        assert c.discovery_score == 0.0

    def test_to_dict_from_dict_roundtrip(self):
        c = DiscoveryCandidate(
            s2_id="abc",
            title="Test Paper",
            authors=["Alice", "Bob"],
            year=2023,
            doi="10.1/x",
            abstract="Some text",
            citation_overlap=2,
            citing_library_ids=["s2_1", "s2_2"],
            bib_coupling_score=0.5,
            bib_coupling_refs=3,
            semantic_score=0.7,
            semantic_source_title="Source",
            author_overlap=1,
            overlapping_authors=["Alice"],
        )
        d = c.to_dict()
        c2 = DiscoveryCandidate.from_dict(d)

        assert c2.s2_id == c.s2_id
        assert c2.title == c.title
        assert c2.authors == c.authors
        assert c2.citation_overlap == c.citation_overlap
        assert c2.bib_coupling_score == c.bib_coupling_score
        assert c2.semantic_score == c.semantic_score
        assert c2.author_overlap == c.author_overlap

    def test_to_dict_includes_computed_fields(self):
        c = DiscoveryCandidate(
            s2_id="abc",
            title="Test",
            authors=[],
            year=2023,
            doi=None,
            abstract="",
            citation_overlap=1,
        )
        d = c.to_dict()
        assert "discovery_score" in d
        assert "signal_count" in d
        assert d["signal_count"] == 1


class TestDiscoveryEngine:
    """Test the DiscoveryEngine with mocked API clients."""

    def _make_paper(self, id, title, authors=None, year=2023, doi=None, abstract=""):
        from incite.models import Paper

        return Paper(
            id=id,
            title=title,
            authors=authors or [],
            year=year,
            doi=doi,
            abstract=abstract,
        )

    def test_run_basic(self):
        """Engine runs end-to-end with mocked clients."""
        from incite.discovery.engine import DiscoveryEngine

        s2 = MagicMock()
        oa = MagicMock()

        # Mock batch lookup
        p1 = self._make_paper("s2_1", "Library Paper 1", doi="10.1/lib1")
        s2.get_papers_batch.return_value = {"DOI:10.1/lib1": p1}

        # Mock references/citations
        ref = self._make_paper("s2_ref", "Referenced Paper", abstract="text")
        cite = self._make_paper("s2_cite", "Citing Paper", abstract="text")
        s2.get_paper_references.return_value = [ref]
        s2.get_paper_citations.return_value = [cite]

        # Mock recommendations
        rec = self._make_paper("s2_rec", "Recommended Paper", abstract="text")
        s2.get_recommendations.return_value = [rec]

        library_papers = [
            {"doi": "10.1/lib1", "title": "Library Paper 1", "authors": ["Smith"], "year": 2023}
        ]

        engine = DiscoveryEngine(s2, oa)
        results = engine.run(
            library_papers,
            config={"n_seed": 10, "skip_bibcoupling": True, "skip_authors": True},
        )

        assert len(results) > 0
        assert all(isinstance(c, DiscoveryCandidate) for c in results)

    def test_progress_callback_called(self):
        """Progress callback is invoked during the run."""
        from incite.discovery.engine import DiscoveryEngine

        s2 = MagicMock()
        oa = MagicMock()
        s2.get_papers_batch.return_value = {}

        callback = MagicMock()
        engine = DiscoveryEngine(s2, oa, progress_callback=callback)
        engine.run(
            [{"doi": "10.1/x", "title": "Test", "authors": [], "year": 2023}],
            config={"skip_bibcoupling": True, "skip_recs": True, "skip_authors": True},
        )

        assert callback.call_count >= 1

    def test_excludes_library_papers(self):
        """Papers already in the library are excluded from candidates."""
        from incite.discovery.engine import DiscoveryEngine

        s2 = MagicMock()
        oa = MagicMock()

        p1 = self._make_paper("s2_1", "Library Paper", doi="10.1/lib")
        s2.get_papers_batch.return_value = {"DOI:10.1/lib": p1}

        # Return the library paper itself as a citation
        s2.get_paper_references.return_value = [p1]
        s2.get_paper_citations.return_value = []
        s2.get_recommendations.return_value = []

        engine = DiscoveryEngine(s2, oa)
        results = engine.run(
            [{"doi": "10.1/lib", "title": "Library Paper", "authors": [], "year": 2023}],
            config={"skip_bibcoupling": True, "skip_authors": True},
        )

        # The library paper itself should not appear in results
        assert all(c.s2_id != "s2_1" for c in results)
