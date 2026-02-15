"""Tests for passage-level training data generation."""

import json

from incite.finetuning.passage_generation import (
    contexts_to_training_examples,
    parse_passage_response,
    select_passages,
    split_passage_data,
)
from incite.models import Chunk, Paper


def _make_chunk(paper_id: str, idx: int, text: str, section: str = "") -> Chunk:
    return Chunk(
        id=f"{paper_id}::chunk_{idx}",
        paper_id=paper_id,
        text=text,
        section=section,
    )


def _make_paper(paper_id: str = "p1", title: str = "Test Paper") -> Paper:
    return Paper(
        id=paper_id,
        title=title,
        abstract="This is a test abstract about machine learning methods.",
        year=2023,
    )


class TestSelectPassages:
    def test_filters_by_length(self):
        """Short and long chunks should be excluded."""
        chunks = [
            _make_chunk("p1", 0, "Too short", "Intro"),
            _make_chunk("p1", 1, "A" * 200, "Methods"),
            _make_chunk("p1", 2, "B" * 2001, "Results"),
        ]
        selected = select_passages(chunks, min_length=150, max_length=2000)
        assert len(selected) == 1
        assert selected[0].id == "p1::chunk_1"

    def test_caps_per_paper(self):
        """max_per_paper should be respected."""
        chunks = [
            _make_chunk("p1", i, f"Passage text number {i} " * 20, f"Section {i}")
            for i in range(15)
        ]
        selected = select_passages(chunks, max_per_paper=5)
        assert len(selected) == 5

    def test_section_diversity(self):
        """No more than 3 from any single section."""
        chunks = [
            _make_chunk("p1", i, f"Passage about methods topic {i} " * 20, "Methods")
            for i in range(10)
        ]
        selected = select_passages(chunks, max_per_paper=20)
        assert len(selected) == 3

    def test_empty_chunks(self):
        """No chunks should return empty."""
        assert select_passages([]) == []

    def test_skips_boilerplate_sections(self):
        """Acknowledgments and similar sections should be skipped."""
        chunks = [
            _make_chunk("p1", 0, "A" * 200, "Acknowledgments"),
            _make_chunk("p1", 1, "B" * 200, "Author Contributions"),
            _make_chunk("p1", 2, "C" * 200, "Methods"),
        ]
        selected = select_passages(chunks)
        assert len(selected) == 1
        assert selected[0].section == "Methods"


class TestParsePassageResponse:
    def _paper(self):
        return _make_paper()

    def _chunk(self):
        return _make_chunk("p1", 3, "The gradient descent method converges quickly.", "Methods")

    def test_valid_response(self):
        """Two valid contexts should be parsed correctly."""
        response = json.dumps(
            {
                "contexts": [
                    {
                        "type": "background",
                        "text": "Previous work has shown that optimization methods converge under certain conditions [CITE].",
                    },
                    {
                        "type": "methods",
                        "text": "We adopt the training approach described in [CITE] for our experiments.",
                    },
                ]
            }
        )
        results = parse_passage_response(self._paper(), self._chunk(), response)
        assert len(results) == 2
        assert results[0]["citation_type"] == "background"
        assert results[1]["citation_type"] == "methods"
        assert results[0]["paper_id"] == "p1"
        assert results[0]["chunk_id"] == "p1::chunk_3"
        assert results[0]["id"] == "passage_p1_3_background"

    def test_markdown_wrapped(self):
        """Should handle JSON wrapped in markdown code fences."""
        response = '```json\n{"contexts": [{"type": "results", "text": "The study found significant improvements in convergence rates [CITE]."}]}\n```'
        results = parse_passage_response(self._paper(), self._chunk(), response)
        assert len(results) == 1

    def test_missing_cite(self):
        """Contexts without [CITE] marker should be filtered out."""
        response = json.dumps(
            {
                "contexts": [
                    {
                        "type": "background",
                        "text": "This is a passage without a cite marker but it is long enough.",
                    },
                ]
            }
        )
        results = parse_passage_response(self._paper(), self._chunk(), response)
        assert len(results) == 0

    def test_invalid_type(self):
        """Contexts with invalid types should be filtered out."""
        response = json.dumps(
            {
                "contexts": [
                    {
                        "type": "invalid_type",
                        "text": "Previous work has shown improvements [CITE].",
                    },
                ]
            }
        )
        results = parse_passage_response(self._paper(), self._chunk(), response)
        assert len(results) == 0

    def test_malformed_json(self):
        """Malformed JSON should return empty list."""
        results = parse_passage_response(self._paper(), self._chunk(), "not valid json {{{")
        assert results == []

    def test_title_verbatim_filtered(self):
        """Contexts containing the paper title verbatim should be filtered."""
        response = json.dumps(
            {
                "contexts": [
                    {
                        "type": "background",
                        "text": "As shown in Test Paper, the method works well [CITE].",
                    },
                ]
            }
        )
        results = parse_passage_response(self._paper(), self._chunk(), response)
        assert len(results) == 0

    def test_duplicate_types_filtered(self):
        """Only the first of each type should be kept."""
        response = json.dumps(
            {
                "contexts": [
                    {
                        "type": "background",
                        "text": "First background context about optimization [CITE].",
                    },
                    {
                        "type": "background",
                        "text": "Second background context about convergence [CITE].",
                    },
                ]
            }
        )
        results = parse_passage_response(self._paper(), self._chunk(), response)
        assert len(results) == 1


class TestContextsToTrainingExamples:
    def test_fields_set_correctly(self):
        """Training example fields should match expected values."""
        paper = _make_paper()
        chunk = _make_chunk("p1", 0, "Passage text about methods.", "Methods")
        contexts = [
            {
                "id": "passage_p1_0_background",
                "paper_id": "p1",
                "chunk_id": "p1::chunk_0",
                "passage_text": "Passage text about methods.",
                "citation_type": "background",
                "text": "Prior work has demonstrated [CITE] these methods.",
                "section": "Methods",
            }
        ]

        examples = contexts_to_training_examples(contexts, {"p1": paper}, {"p1": [chunk]})

        assert len(examples) == 1
        ex = examples[0]
        # Passage positive now includes metadata prefix (matches Chunk.to_embedding_text())
        assert ex.passage_positive == "Test Paper. 2023\n\nPassage text about methods."
        assert ex.passage_score == 1.0
        assert ex.passage_validation == 5
        assert ex.intent == "background"
        assert ex.passage_section == "Methods"
        assert ex.source == "passage_gen"
        assert "[CITE]" not in ex.query

    def test_hard_negatives_from_same_paper(self):
        """Hard negatives should come from other chunks of the same paper."""
        paper = _make_paper()
        chunks = [
            _make_chunk("p1", 0, "Target passage text.", "Methods"),
            _make_chunk("p1", 1, "Other passage one.", "Results"),
            _make_chunk("p1", 2, "Other passage two.", "Discussion"),
        ]
        contexts = [
            {
                "id": "passage_p1_0_methods",
                "paper_id": "p1",
                "chunk_id": "p1::chunk_0",
                "passage_text": "Target passage text.",
                "citation_type": "methods",
                "text": "We follow the approach described in [CITE].",
                "section": "Methods",
            }
        ]

        examples = contexts_to_training_examples(
            contexts, {"p1": paper}, {"p1": chunks}, max_hard_negatives=3
        )

        assert len(examples) == 1
        # Should have 2 hard negatives (the other 2 chunks, not the target)
        assert len(examples[0].hard_negatives) == 2
        # Hard negatives now include metadata prefix, so check raw text isn't present
        for neg in examples[0].hard_negatives:
            assert "Target passage text." not in neg


class TestSplitPassageData:
    def test_split_by_paper_no_leakage(self):
        """No paper should appear in both train and eval splits."""
        paper1 = _make_paper("p1")
        paper2 = _make_paper("p2")
        paper3 = _make_paper("p3")

        from incite.finetuning.data_preparation import TrainingExample

        examples = []
        for pid in ["p1", "p2", "p3"]:
            for intent in ["background", "methods"]:
                examples.append(
                    TrainingExample(
                        query=f"Query for {pid} {intent}",
                        positive=f"Positive for {pid}",
                        cited_paper_id=pid,
                        source="passage_gen",
                        passage_positive=f"Passage for {pid}",
                        passage_score=1.0,
                        passage_validation=5,
                        intent=intent,
                        passage_section="Methods",
                    )
                )

        train, dev, eval_set = split_passage_data(examples, dev_fraction=0.34, eval_fraction=0.34)

        train_papers = {ex.cited_paper_id for ex in train}
        dev_papers = {ex.cited_paper_id for ex in dev}
        eval_papers = {ex.cited_paper_id for ex in eval_set}

        # No overlap between any splits
        assert train_papers & eval_papers == set()
        assert train_papers & dev_papers == set()
        assert dev_papers & eval_papers == set()

        # All papers accounted for
        assert train_papers | dev_papers | eval_papers == {"p1", "p2", "p3"}
