"""Tests for dataset validation: quality checks, split integrity, JSONL format, dedup."""

import json
from pathlib import Path

from incite.finetuning.data_preparation import TrainingExample, load_training_data
from incite.finetuning.validation import (
    MIN_POSITIVE_LENGTH,
    MIN_QUERY_LENGTH,
    validate_dataset,
    validate_example,
    validate_split_integrity,
)


def _make_example(**overrides) -> TrainingExample:
    """Create a valid TrainingExample with optional overrides."""
    defaults = {
        "query": "We apply gradient descent optimization to train the neural network " * 2,
        "positive": "Gradient Descent Methods [SEP] This paper surveys gradient-based optimization",
        "hard_negatives": ["Negative 1 text here", "Negative 2 text here"],
        "source_paper_id": "src_001",
        "cited_paper_id": "cite_001",
        "source": "test",
    }
    defaults.update(overrides)
    return TrainingExample(**defaults)


def _write_jsonl(path: Path, examples: list[TrainingExample]):
    """Write examples to JSONL file."""
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")


class TestTrainingExampleQuality:
    def test_query_minimum_length(self):
        ex = _make_example(query="too short")
        issues = validate_example(ex)
        assert any("query too short" in i for i in issues)

    def test_query_at_minimum_length_passes(self):
        ex = _make_example(query="x" * MIN_QUERY_LENGTH)
        issues = validate_example(ex)
        assert not any("query too short" in i for i in issues)

    def test_positive_minimum_length(self):
        ex = _make_example(positive="short")
        issues = validate_example(ex)
        assert any("positive too short" in i for i in issues)

    def test_positive_at_minimum_length_passes(self):
        ex = _make_example(positive="x" * MIN_POSITIVE_LENGTH)
        issues = validate_example(ex)
        assert not any("positive too short" in i for i in issues)

    def test_query_not_empty_after_stripping(self):
        ex = _make_example(query="   " * 20)
        issues = validate_example(ex)
        assert any("empty after stripping" in i for i in issues)

    def test_hard_negatives_not_duplicated(self):
        ex = _make_example(hard_negatives=["same text", "same text"])
        issues = validate_example(ex)
        assert any("duplicate hard_negatives" in i for i in issues)

    def test_hard_negatives_differ_from_positive(self):
        pos = "Gradient Descent Methods [SEP] This paper surveys gradient-based optimization"
        ex = _make_example(positive=pos, hard_negatives=[pos, "other text"])
        issues = validate_example(ex)
        assert any("positive appears in hard_negatives" in i for i in issues)

    def test_passage_positive_without_intent_is_not_an_error(self):
        """After normalization, passage_positive without intent is OK."""
        ex = _make_example(passage_positive="A passage from the paper.", intent="")
        issues = validate_example(ex)
        assert not any("passage_positive" in i for i in issues)

    def test_passage_fields_with_intent_passes(self):
        ex = _make_example(passage_positive="A passage from the paper.", intent="method")
        issues = validate_example(ex)
        assert not any("passage_positive" in i for i in issues)

    def test_valid_example_passes(self):
        ex = _make_example()
        issues = validate_example(ex)
        assert issues == []


class TestTrainDevSplitIntegrity:
    def test_no_source_paper_overlap(self, tmp_path):
        train = [_make_example(source_paper_id="paper_A")]
        dev = [_make_example(source_paper_id="paper_A",
                             query="Different query text for dev set that is long enough" * 2)]
        _write_jsonl(tmp_path / "train.jsonl", train)
        _write_jsonl(tmp_path / "dev.jsonl", dev)

        issues = validate_split_integrity(
            tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
        )
        assert any("Source paper overlap" in i for i in issues)

    def test_no_query_overlap(self, tmp_path):
        shared_query = "We apply gradient descent optimization to train the neural network " * 2
        train = [_make_example(query=shared_query, source_paper_id="p1")]
        dev = [_make_example(query=shared_query, source_paper_id="p2")]
        _write_jsonl(tmp_path / "train.jsonl", train)
        _write_jsonl(tmp_path / "dev.jsonl", dev)

        issues = validate_split_integrity(
            tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
        )
        assert any("Query overlap" in i for i in issues)

    def test_dev_fraction_reasonable(self, tmp_path):
        # 1 train, 99 dev = 99% dev -> outside 5-25%
        train = [_make_example(source_paper_id="p1")]
        dev = [
            _make_example(
                source_paper_id=f"dev_{i}",
                query=f"Unique query text number {i} that is long enough for validation " * 2,
            )
            for i in range(99)
        ]
        _write_jsonl(tmp_path / "train.jsonl", train)
        _write_jsonl(tmp_path / "dev.jsonl", dev)

        issues = validate_split_integrity(
            tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
        )
        assert any("Dev fraction" in i for i in issues)

    def test_clean_split_passes(self, tmp_path):
        long_q = "that is sufficiently long for validation "
        train = [
            _make_example(
                source_paper_id=f"p{i}",
                query=f"Train query number {i} {long_q}" * 2,
            )
            for i in range(9)
        ]
        dev = [
            _make_example(source_paper_id="p_dev",
                          query="Dev query that is completely different and long enough " * 2)
        ]
        _write_jsonl(tmp_path / "train.jsonl", train)
        _write_jsonl(tmp_path / "dev.jsonl", dev)

        issues = validate_split_integrity(
            tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
        )
        assert issues == []


class TestJSONLFormat:
    def test_valid_jsonl_roundtrip(self, tmp_path):
        examples = [_make_example(), _make_example(source="s2orc")]
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, examples)

        loaded = load_training_data(path)
        assert len(loaded) == 2
        assert loaded[0].query == examples[0].query
        assert loaded[1].source == "s2orc"

    def test_required_fields_present(self, tmp_path):
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"query": "q", "positive": "p"}) + "\n")

        loaded = load_training_data(path)
        assert len(loaded) == 1
        assert loaded[0].query == "q"

    def test_empty_lines_skipped(self, tmp_path):
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"query": "q1", "positive": "p1"}) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps({"query": "q2", "positive": "p2"}) + "\n")

        loaded = load_training_data(path)
        assert len(loaded) == 2

    def test_validate_dataset_on_clean_file(self, tmp_path):
        examples = [_make_example() for _ in range(5)]
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, examples)

        issues = validate_dataset(path)
        assert issues == []

    def test_validate_dataset_catches_short_queries(self, tmp_path):
        examples = [_make_example(), _make_example(query="too short")]
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, examples)

        issues = validate_dataset(path)
        assert len(issues) == 1
        assert "query too short" in issues[0]


class TestDataPipelineDedup:
    def test_duplicates_removed_by_query_hash(self, tmp_path):
        """DataPipeline should deduplicate identical queries."""
        from incite.finetuning.data_pipeline import DataPipeline, PipelineConfig

        shared_query = "Identical query for dedup testing that meets minimum length " * 2

        class FakeSource:
            name = "fake"

            def stream(self, limit=0):
                yield _make_example(query=shared_query, source="a")
                yield _make_example(query=shared_query, source="b")
                unique = "Unique second query that is different " * 2
                yield _make_example(query=unique, source="c")

            def count_available(self):
                return 3

        config = PipelineConfig(output_dir=tmp_path, show_progress=False)
        pipeline = DataPipeline(config)
        stats = pipeline.build([(FakeSource(), 0)])

        assert stats.duplicates_removed == 1
        total = stats.train_examples + stats.dev_examples
        assert total == 2

    def test_quality_filter_removes_short_queries(self, tmp_path):
        from incite.finetuning.data_pipeline import DataPipeline, PipelineConfig

        class FakeSource:
            name = "fake"

            def stream(self, limit=0):
                yield _make_example(query="too short")
                yield _make_example()

            def count_available(self):
                return 2

        config = PipelineConfig(output_dir=tmp_path, show_progress=False)
        pipeline = DataPipeline(config)
        stats = pipeline.build([(FakeSource(), 0)])

        assert stats.quality_filtered == 1
        total = stats.train_examples + stats.dev_examples
        assert total == 1

    def test_quality_filter_removes_short_positives(self, tmp_path):
        from incite.finetuning.data_pipeline import DataPipeline, PipelineConfig

        class FakeSource:
            name = "fake"

            def stream(self, limit=0):
                yield _make_example(positive="x")
                yield _make_example()

            def count_available(self):
                return 2

        config = PipelineConfig(output_dir=tmp_path, show_progress=False)
        pipeline = DataPipeline(config)
        stats = pipeline.build([(FakeSource(), 0)])

        assert stats.quality_filtered == 1
        total = stats.train_examples + stats.dev_examples
        assert total == 1
