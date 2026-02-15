"""Core types for fine-tuning training data.

TrainingExample is the universal data format for contrastive learning pairs.
All training sources produce TrainingExamples; all training scripts consume them.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingExample:
    """A single training example for contrastive learning.

    Core fields (always present):
        query: Citation context text (cleaned, multi-scale)
        positive: Cited paper's embedding text (title. authors. year. journal. abstract)
        hard_negatives: Other papers from the same reference set or mined negatives
        source_paper_id: ID of the paper containing the citation context
        cited_paper_id: ID of the cited paper
        source: Data source tag (e.g. "s2orc_abstract", "unarxiv")
        scale: Context scale used for query (local/narrow/broad/section)

    Passage fields (optional, for passage-level training data):
        passage_positive, passage_score, passage_validation, intent,
        passage_section, passage_hard_negatives
    """

    query: str
    positive: str
    hard_negatives: list[str] = field(default_factory=list)
    source_paper_id: str = ""
    cited_paper_id: str = ""
    source: str = ""
    scale: str = ""

    # Passage-level fields (Phase 7) — kept for backward compat with old JSONL
    passage_positive: str = ""
    passage_score: float = 0.0
    passage_validation: int = 0
    intent: str = ""
    passage_section: str = ""
    passage_hard_negatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "query": self.query,
            "positive": self.positive,
            "hard_negatives": list(dict.fromkeys(self.hard_negatives)),
            "source_paper_id": self.source_paper_id,
            "cited_paper_id": self.cited_paper_id,
        }
        if self.source:
            d["source"] = self.source
        # Only serialize non-empty passage fields for backward compatibility
        if self.passage_positive:
            d["passage_positive"] = self.passage_positive
        if self.passage_score > 0:
            d["passage_score"] = self.passage_score
        if self.passage_validation > 0:
            d["passage_validation"] = self.passage_validation
        if self.intent:
            d["intent"] = self.intent
        if self.passage_section:
            d["passage_section"] = self.passage_section
        if self.passage_hard_negatives:
            d["passage_hard_negatives"] = self.passage_hard_negatives
        if self.scale:
            d["scale"] = self.scale
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingExample":
        positive = data["positive"]
        hard_negatives = list(dict.fromkeys(data.get("hard_negatives", [])))
        passage_positive = data.get("passage_positive", "")
        passage_hard_negatives = data.get("passage_hard_negatives", [])

        # Auto-migrate legacy passage data: if passage_positive exists but
        # hard_negatives is empty and passage_hard_negatives is not, the data
        # was written before normalization. Promote passage fields to primary.
        if passage_positive and not hard_negatives and passage_hard_negatives:
            positive = passage_positive
            hard_negatives = passage_hard_negatives

        return cls(
            query=data["query"],
            positive=positive,
            hard_negatives=hard_negatives,
            source_paper_id=data.get("source_paper_id", ""),
            cited_paper_id=data.get("cited_paper_id", ""),
            source=data.get("source", ""),
            passage_positive=passage_positive,
            passage_score=data.get("passage_score", 0.0),
            passage_validation=data.get("passage_validation", 0),
            intent=data.get("intent", ""),
            passage_section=data.get("passage_section", ""),
            passage_hard_negatives=passage_hard_negatives,
            scale=data.get("scale", ""),
        )


def load_training_data(path: Path) -> list[TrainingExample]:
    """Load training examples from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(TrainingExample.from_dict(json.loads(line)))
    return examples


def save_training_data(examples: list[TrainingExample], path: Path) -> None:
    """Save training examples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
