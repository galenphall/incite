"""Tests for S2 citation context mining pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from incite.finetuning.s2_context_mining import (
    CitationContextHarvester,
    _load_state,
    _save_state,
    backfill_hard_negatives,
    load_seeds,
    merge_to_splits,
    save_seeds,
)
from incite.finetuning.types import TrainingExample, save_training_data
from incite.models import Paper, format_paper_embedding_text

# --- Fixtures ---


def _make_seed(pid: str, title: str = "Test Paper", year: int = 2020) -> Paper:
    return Paper(
        id=pid,
        title=title,
        abstract=f"Abstract for {title}.",
        authors=["Alice Smith", "Bob Jones"],
        year=year,
        journal="Nature",
    )


def _seed_embedding_text(seed: Paper) -> str:
    return format_paper_embedding_text(
        title=seed.title,
        abstract=seed.abstract,
        author_lastnames=seed.author_lastnames,
        year=seed.year,
        journal=seed.journal,
        include_abstract=True,
        include_metadata=True,
    )


def _make_citation_response(citing_id: str, contexts: list[str]) -> dict:
    """Build a single citation item matching S2 API format."""
    return {
        "citingPaper": {"paperId": citing_id},
        "contexts": contexts,
    }


# --- Context cleaning ---


class TestContextCleaning:
    """Test that citation contexts are cleaned properly."""

    def test_citation_markers_removed(self):
        """[CITE] markers and {{cite:...}} patterns should be stripped."""
        harvester = CitationContextHarvester(min_context_length=10)

        seed = _make_seed("seed1", "Deep Learning for NLP")
        seed_texts = {"seed1": _seed_embedding_text(seed)}
        co_citation_map: dict[str, set[str]] = {}

        # Mock the API to return a context with citation markers
        raw_context = "This approach [CITE] significantly improved results in the field of study."
        with patch.object(
            harvester,
            "_fetch_all_citation_contexts",
            return_value=[("citing1", raw_context)],
        ):
            examples = harvester.harvest_seed(seed, seed_texts, co_citation_map)

        assert len(examples) == 1
        assert "[CITE]" not in examples[0].query
        assert "significantly improved" in examples[0].query

    def test_short_contexts_filtered(self):
        """Contexts shorter than min_context_length should be excluded."""
        harvester = CitationContextHarvester(min_context_length=50)

        seed = _make_seed("seed1")
        seed_texts = {"seed1": _seed_embedding_text(seed)}
        co_citation_map: dict[str, set[str]] = {}

        with patch.object(
            harvester,
            "_fetch_all_citation_contexts",
            return_value=[
                ("citing1", "Too short"),
                ("citing2", "This is a sufficiently long citation context that passes the filter."),
            ],
        ):
            examples = harvester.harvest_seed(seed, seed_texts, co_citation_map)

        assert len(examples) == 1
        assert "sufficiently long" in examples[0].query

    def test_html_and_whitespace_cleaned(self):
        """HTML tags and excessive whitespace should be normalized."""
        harvester = CitationContextHarvester(min_context_length=10)

        seed = _make_seed("seed1")
        seed_texts = {"seed1": _seed_embedding_text(seed)}
        co_citation_map: dict[str, set[str]] = {}

        raw = "This   <b>study</b>   demonstrated    the effectiveness of the method."
        with patch.object(
            harvester,
            "_fetch_all_citation_contexts",
            return_value=[("citing1", raw)],
        ):
            examples = harvester.harvest_seed(seed, seed_texts, co_citation_map)

        assert len(examples) == 1
        assert "<b>" not in examples[0].query
        assert "  " not in examples[0].query


# --- Co-citation negatives ---


class TestCoCitationNegatives:
    """Test co-citation hard negative construction."""

    def test_builds_negatives_from_co_citation(self):
        """When a citing paper cites multiple seeds, others become negatives."""
        harvester = CitationContextHarvester(min_context_length=10, max_hard_negatives=5)

        seed_a = _make_seed("seedA", "Paper A")
        seed_b = _make_seed("seedB", "Paper B")
        seed_c = _make_seed("seedC", "Paper C")

        seed_texts = {
            "seedA": _seed_embedding_text(seed_a),
            "seedB": _seed_embedding_text(seed_b),
            "seedC": _seed_embedding_text(seed_c),
        }

        # citing1 already cites seedB and seedC
        co_citation_map: dict[str, set[str]] = {
            "citing1": {"seedB", "seedC"},
        }

        context = "This important study demonstrated significant results in the analysis."
        with patch.object(
            harvester,
            "_fetch_all_citation_contexts",
            return_value=[("citing1", context)],
        ):
            examples = harvester.harvest_seed(seed_a, seed_texts, co_citation_map)

        assert len(examples) == 1
        ex = examples[0]
        # Hard negatives should come from co-cited seeds
        assert len(ex.hard_negatives) > 0
        assert all(neg in [seed_texts["seedB"], seed_texts["seedC"]] for neg in ex.hard_negatives)

    def test_excludes_positive_from_negatives(self):
        """The target seed should never appear as its own hard negative."""
        harvester = CitationContextHarvester(min_context_length=10, max_hard_negatives=5)

        seed = _make_seed("seed1", "Target Paper")
        seed_texts = {"seed1": _seed_embedding_text(seed)}

        # Simulate co-citation map that includes seed1 itself
        co_citation_map: dict[str, set[str]] = {
            "citing1": {"seed1"},
        }

        context = "This important study demonstrated significant results in the analysis."
        with patch.object(
            harvester,
            "_fetch_all_citation_contexts",
            return_value=[("citing1", context)],
        ):
            examples = harvester.harvest_seed(seed, seed_texts, co_citation_map)

        assert len(examples) == 1
        assert len(examples[0].hard_negatives) == 0  # No other seeds to use

    def test_co_citation_map_updated(self):
        """harvest_seed should update the co-citation map with new citations."""
        harvester = CitationContextHarvester(min_context_length=10)

        seed = _make_seed("seed1")
        seed_texts = {"seed1": _seed_embedding_text(seed)}
        co_citation_map: dict[str, set[str]] = {}

        context = "This study found important results in the research literature and field."
        with patch.object(
            harvester,
            "_fetch_all_citation_contexts",
            return_value=[("citingX", context)],
        ):
            harvester.harvest_seed(seed, seed_texts, co_citation_map)

        assert "citingX" in co_citation_map
        assert "seed1" in co_citation_map["citingX"]


# --- Positive formatting ---


class TestPositiveFormatting:
    """Test that positives use canonical embedding format."""

    def test_uses_canonical_format(self):
        """Positive should match format_paper_embedding_text() output."""
        harvester = CitationContextHarvester(min_context_length=10)

        seed = Paper(
            id="s1",
            title="Attention Is All You Need",
            abstract="We propose a new architecture.",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            year=2017,
            journal="NeurIPS",
        )
        expected = format_paper_embedding_text(
            title="Attention Is All You Need",
            abstract="We propose a new architecture.",
            author_lastnames=["Vaswani", "Shazeer"],
            year=2017,
            journal="NeurIPS",
            include_abstract=True,
            include_metadata=True,
        )

        seed_texts = {"s1": expected}
        co_citation_map: dict[str, set[str]] = {}

        context = "The transformer architecture has been highly influential in recent years."
        with patch.object(
            harvester,
            "_fetch_all_citation_contexts",
            return_value=[("c1", context)],
        ):
            examples = harvester.harvest_seed(seed, seed_texts, co_citation_map)

        assert len(examples) == 1
        assert examples[0].positive == expected
        assert "Attention Is All You Need" in examples[0].positive
        assert "Vaswani and Shazeer" in examples[0].positive
        assert "2017" in examples[0].positive
        assert "NeurIPS" in examples[0].positive

    def test_source_tag_is_s2_contexts(self):
        """Source should be tagged as 's2_contexts'."""
        harvester = CitationContextHarvester(min_context_length=10)
        seed = _make_seed("s1")
        seed_texts = {"s1": _seed_embedding_text(seed)}
        co_citation_map: dict[str, set[str]] = {}

        context = "This study provides substantial evidence for the theoretical framework."
        with patch.object(
            harvester,
            "_fetch_all_citation_contexts",
            return_value=[("c1", context)],
        ):
            examples = harvester.harvest_seed(seed, seed_texts, co_citation_map)

        assert examples[0].source == "s2_contexts"
        assert examples[0].scale == "narrow"


# --- State persistence ---


class TestStatePersistence:
    """Test state save/load and resumability."""

    def test_save_and_load_roundtrip(self):
        """State should round-trip through JSON correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"

            state = {
                "completed_seeds": ["a", "b"],
                "total_examples": 100,
                "co_citation_map": {"citing1": ["a", "b"]},
                "seed_stats": {"a": {"contexts_harvested": 50}},
            }

            _save_state(state_path, state)
            loaded = _load_state(state_path)

            assert loaded["completed_seeds"] == ["a", "b"]
            assert loaded["total_examples"] == 100
            assert loaded["co_citation_map"]["citing1"] == ["a", "b"]

    def test_load_missing_file(self):
        """Loading non-existent state should return empty defaults."""
        state = _load_state(Path("/nonexistent/state.json"))

        assert state["completed_seeds"] == []
        assert state["total_examples"] == 0
        assert state["co_citation_map"] == {}

    def test_resume_skips_completed_seeds(self):
        """harvest_all should skip seeds already in completed_seeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            seed_a = _make_seed("seedA", "Paper A")
            seed_b = _make_seed("seedB", "Paper B")

            # Pre-populate state: seedA is already done
            state = {
                "completed_seeds": ["seedA"],
                "total_examples": 5,
                "co_citation_map": {},
                "seed_stats": {"seedA": {"contexts_harvested": 5, "seed_index": 0}},
            }
            _save_state(output_dir / "state.json", state)

            harvester = CitationContextHarvester(min_context_length=10)

            context = "This study provides substantial evidence for the theoretical framework."
            with patch.object(
                harvester,
                "_fetch_all_citation_contexts",
                return_value=[("citing1", context)],
            ) as mock_fetch:
                harvester.harvest_all(
                    seeds=[seed_a, seed_b],
                    output_dir=output_dir,
                    target=100,
                    show_progress=False,
                )

            # Should only have called fetch for seedB (seedA was skipped)
            assert mock_fetch.call_count == 1
            mock_fetch.assert_called_once_with(seed_b.id)

    def test_seeds_file_persistence(self):
        """Seeds should be saved and loadable for backfill."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "seeds.jsonl"
            seeds = [
                _make_seed("s1", "Paper One", 2020),
                _make_seed("s2", "Paper Two", 2021),
            ]

            save_seeds(seeds, path)
            loaded = load_seeds(path)

            assert len(loaded) == 2
            assert loaded[0].id == "s1"
            assert loaded[0].title == "Paper One"
            assert loaded[1].id == "s2"
            assert loaded[1].year == 2021


# --- Train/dev split ---


class TestMergeToSplits:
    """Test merging per-seed files into train/dev splits."""

    def test_no_leakage_by_seed(self):
        """Examples from the same seed should all be in train or all in dev."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create examples for 5 seeds
            for i in range(5):
                examples = [
                    TrainingExample(
                        query=f"Context {j} for seed {i} with enough text to pass filters.",
                        positive=f"Positive text for seed {i}",
                        cited_paper_id=f"seed_{i}",
                        source_paper_id=f"citing_{i}_{j}",
                        source="s2_contexts",
                    )
                    for j in range(10)
                ]
                save_training_data(examples, output_dir / f"seed_{i:04d}.jsonl")

            stats = merge_to_splits(output_dir, dev_fraction=0.4, seed=42)

            assert stats["train_count"] + stats["dev_count"] == 50
            assert stats["seeds"] == 5

            # Verify no leakage: load both splits and check
            train_seeds = set()
            with open(output_dir / "train.jsonl") as f:
                for line in f:
                    ex = json.loads(line)
                    train_seeds.add(ex["cited_paper_id"])

            dev_seeds = set()
            with open(output_dir / "dev.jsonl") as f:
                for line in f:
                    ex = json.loads(line)
                    dev_seeds.add(ex["cited_paper_id"])

            # No seed should appear in both splits
            assert train_seeds.isdisjoint(dev_seeds)

    def test_handles_empty_dir(self):
        """Should return zeros for an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = merge_to_splits(Path(tmpdir))
            assert stats["train_count"] == 0
            assert stats["dev_count"] == 0

    def test_reproducible_split(self):
        """Same seed should produce the same split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            for i in range(10):
                examples = [
                    TrainingExample(
                        query=f"Context for seed {i} with enough text to pass length filters.",
                        positive=f"Positive {i}",
                        cited_paper_id=f"seed_{i}",
                        source="s2_contexts",
                    )
                ]
                save_training_data(examples, output_dir / f"seed_{i:04d}.jsonl")

            stats1 = merge_to_splits(output_dir, seed=42)

            with open(output_dir / "train.jsonl") as f:
                train1 = [json.loads(line) for line in f]

            # Re-merge with same seed
            stats2 = merge_to_splits(output_dir, seed=42)

            with open(output_dir / "train.jsonl") as f:
                train2 = [json.loads(line) for line in f]

            assert stats1 == stats2
            assert [e["cited_paper_id"] for e in train1] == [e["cited_paper_id"] for e in train2]


# --- Backfill ---


class TestBackfillHardNegatives:
    """Test the hard negative backfill pass."""

    def test_enriches_early_seeds(self):
        """Examples with few negatives should gain more from the complete map."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            seed_a = _make_seed("seedA", "Paper A")
            seed_b = _make_seed("seedB", "Paper B")
            seed_c = _make_seed("seedC", "Paper C")

            # Example from seedA with no hard negatives (early seed)
            examples = [
                TrainingExample(
                    query="A citation context that references paper A in the literature.",
                    positive=_seed_embedding_text(seed_a),
                    hard_negatives=[],
                    source_paper_id="citing1",
                    cited_paper_id="seedA",
                    source="s2_contexts",
                )
            ]
            save_training_data(examples, output_dir / "seed_0000.jsonl")

            # State with complete co-citation map showing citing1 cites A, B, and C
            state = {
                "completed_seeds": ["seedA", "seedB", "seedC"],
                "total_examples": 10,
                "co_citation_map": {
                    "citing1": ["seedA", "seedB", "seedC"],
                },
                "seed_stats": {},
            }
            _save_state(output_dir / "state.json", state)

            stats = backfill_hard_negatives(
                output_dir=output_dir,
                seeds=[seed_a, seed_b, seed_c],
                show_progress=False,
            )

            assert stats["enriched_count"] >= 1
            assert stats["negatives_added"] >= 1

            # Verify the file was updated
            with open(output_dir / "seed_0000.jsonl") as f:
                updated = TrainingExample.from_dict(json.loads(f.readline()))

            assert len(updated.hard_negatives) > 0


# --- API pagination ---


class TestFetchCitationsPage:
    """Test the paginated citation fetching."""

    def test_retries_on_429(self):
        """Should retry with backoff on rate limit errors."""
        harvester = CitationContextHarvester(delay=0.0)

        mock_responses = [
            MagicMock(status_code=429),
            MagicMock(
                status_code=200,
                json=MagicMock(
                    return_value={
                        "data": [_make_citation_response("c1", ["A long context string."])],
                    }
                ),
            ),
        ]
        mock_responses[0].raise_for_status = MagicMock()
        mock_responses[1].raise_for_status = MagicMock()

        with patch("incite.finetuning.s2_context_mining.requests.get") as mock_get:
            with patch("incite.finetuning.s2_context_mining.time.sleep"):
                mock_get.side_effect = mock_responses
                result = harvester._fetch_citations_page("paper1")

        assert len(result["data"]) == 1

    def test_returns_empty_on_all_failures(self):
        """Should return empty data after exhausting retries."""
        harvester = CitationContextHarvester(delay=0.0)

        mock_resp = MagicMock(status_code=500)
        with patch("incite.finetuning.s2_context_mining.requests.get") as mock_get:
            with patch("incite.finetuning.s2_context_mining.time.sleep"):
                mock_get.return_value = mock_resp
                result = harvester._fetch_citations_page("paper1")

        assert result["data"] == []

    def test_max_contexts_per_seed_caps_results(self):
        """Should stop fetching once max_contexts_per_seed is reached."""
        harvester = CitationContextHarvester(
            delay=0.0, min_context_length=5, max_contexts_per_seed=10
        )

        # Page has 500 items, each with a valid context — way more than the cap
        items = [
            _make_citation_response(f"c{i}", [f"Context text number {i}."]) for i in range(500)
        ]
        page1 = {"data": items, "next": 500}
        page2 = {"data": items[:100]}  # Should never be fetched

        with patch.object(
            harvester,
            "_fetch_citations_page",
            side_effect=[page1, page2],
        ) as mock_fetch:
            results = harvester._fetch_all_citation_contexts("paper1")

        # Should be capped at max_contexts_per_seed
        assert len(results) <= 10
        # Should NOT have fetched page 2 — early exit saves API calls
        assert mock_fetch.call_count == 1

    def test_pagination(self):
        """Should paginate through multiple pages of results."""
        harvester = CitationContextHarvester(
            delay=0.0, min_context_length=5, max_contexts_per_seed=10000
        )

        page1 = {
            "data": [_make_citation_response("c1", ["Context one text."])] * 500,
            "next": 500,
        }
        page2 = {
            "data": [_make_citation_response("c2", ["Context two text."])] * 100,
        }

        with patch.object(
            harvester,
            "_fetch_citations_page",
            side_effect=[page1, page2],
        ):
            results = harvester._fetch_all_citation_contexts("paper1")

        # 500 contexts from page1 + 100 from page2
        assert len(results) == 600
