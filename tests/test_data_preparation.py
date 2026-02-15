"""Tests for TrainingExample data preparation, including passage field extensions."""

import json
import tempfile
from pathlib import Path

import pytest

from incite.finetuning.data_preparation import TrainingExample, load_training_data


class TestTrainingExampleBackwardCompat:
    def test_old_jsonl_roundtrip(self):
        """Old JSONL without passage fields should load and round-trip correctly."""
        old_data = {
            "query": "We use gradient descent.",
            "positive": "Paper about gradient descent [SEP] Abstract text",
            "hard_negatives": ["Negative 1", "Negative 2"],
            "source_paper_id": "src1",
            "cited_paper_id": "cite1",
        }

        ex = TrainingExample.from_dict(old_data)
        assert ex.query == "We use gradient descent."
        assert ex.positive == "Paper about gradient descent [SEP] Abstract text"
        assert len(ex.hard_negatives) == 2
        assert ex.passage_positive == ""
        assert ex.passage_score == 0.0
        assert ex.passage_validation == 0
        assert ex.intent == ""
        assert ex.passage_section == ""
        assert ex.passage_hard_negatives == []

        # Round-trip: serialized dict should not contain empty passage fields
        d = ex.to_dict()
        assert "passage_positive" not in d
        assert "passage_score" not in d
        assert "passage_validation" not in d
        assert "intent" not in d
        assert "passage_section" not in d
        assert "passage_hard_negatives" not in d

    def test_new_fields_serialize(self):
        """New passage fields should serialize when non-empty."""
        ex = TrainingExample(
            query="query",
            positive="positive",
            passage_positive="A passage from the paper.",
            passage_score=0.85,
            passage_validation=4,
            intent="method",
            passage_section="Methods",
            passage_hard_negatives=["wrong passage 1"],
        )

        d = ex.to_dict()
        assert d["passage_positive"] == "A passage from the paper."
        assert d["passage_score"] == 0.85
        assert d["passage_validation"] == 4
        assert d["intent"] == "method"
        assert d["passage_section"] == "Methods"
        assert d["passage_hard_negatives"] == ["wrong passage 1"]

    def test_new_fields_deserialize(self):
        """New passage fields should deserialize correctly."""
        data = {
            "query": "query",
            "positive": "positive",
            "passage_positive": "passage text",
            "passage_score": 0.75,
            "passage_validation": 5,
            "intent": "result",
            "passage_section": "Results",
            "passage_hard_negatives": ["neg1", "neg2"],
        }

        ex = TrainingExample.from_dict(data)
        assert ex.passage_positive == "passage text"
        assert ex.passage_score == 0.75
        assert ex.passage_validation == 5
        assert ex.intent == "result"
        assert ex.passage_section == "Results"
        assert ex.passage_hard_negatives == ["neg1", "neg2"]

    def test_mixed_jsonl_file(self):
        """File with both old and new format examples should load correctly."""
        old_example = {"query": "old query", "positive": "old positive"}
        new_example = {
            "query": "new query",
            "positive": "new positive",
            "passage_positive": "a passage",
            "passage_score": 0.9,
            "intent": "background",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(old_example) + "\n")
            f.write(json.dumps(new_example) + "\n")
            tmp_path = Path(f.name)

        try:
            examples = load_training_data(tmp_path)
            assert len(examples) == 2

            # Old example
            assert examples[0].passage_positive == ""
            assert examples[0].passage_score == 0.0

            # New example
            assert examples[1].passage_positive == "a passage"
            assert examples[1].passage_score == 0.9
            assert examples[1].intent == "background"
        finally:
            tmp_path.unlink()

    def test_source_field_preserved(self):
        """Source field should still work correctly."""
        ex = TrainingExample(
            query="q", positive="p", source="s2orc"
        )
        d = ex.to_dict()
        assert d["source"] == "s2orc"

        ex2 = TrainingExample(query="q", positive="p")
        d2 = ex2.to_dict()
        assert "source" not in d2
