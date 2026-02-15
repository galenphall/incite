"""Multi-source data pipeline for fine-tuning.

Combines multiple DataSource streams into a single train/dev split
with deduplication and quality filtering.
"""

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

from tqdm import tqdm

from incite.finetuning.data_preparation import TrainingExample


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""

    output_dir: Path = Path("data/finetuning")
    dev_fraction: float = 0.1
    seed: int = 42
    min_query_length: int = 50
    min_positive_length: int = 30
    show_progress: bool = True


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""

    per_source: dict[str, int] = field(default_factory=dict)
    total_streamed: int = 0
    duplicates_removed: int = 0
    quality_filtered: int = 0
    train_examples: int = 0
    dev_examples: int = 0

    def summary(self) -> str:
        lines = ["Data pipeline results:"]
        lines.append("  Sources:")
        for name, count in sorted(self.per_source.items()):
            lines.append(f"    {name}: {count:,}")
        lines.append(f"  Total streamed: {self.total_streamed:,}")
        lines.append(f"  Duplicates removed: {self.duplicates_removed:,}")
        lines.append(f"  Quality filtered: {self.quality_filtered:,}")
        lines.append(f"  Train: {self.train_examples:,}")
        lines.append(f"  Dev: {self.dev_examples:,}")
        lines.append(f"  Total usable: {self.train_examples + self.dev_examples:,}")
        return "\n".join(lines)


class DataPipeline:
    """Combines multiple data sources into train/dev splits."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def build(
        self,
        sources: list[tuple[Any, int]],
    ) -> PipelineStats:
        """Build training data from multiple sources.

        Args:
            sources: List of (source, limit) tuples. Each source must have
                     name, stream(limit), and count_available() attributes.
                     limit=0 means use all available from that source.

        Returns:
            PipelineStats with processing summary
        """
        config = self.config
        config.output_dir.mkdir(parents=True, exist_ok=True)
        stats = PipelineStats()

        # Phase 1: Stream from all sources
        print("\nPhase 1: Streaming from sources...")
        all_examples: list[TrainingExample] = []
        seen_hashes: set[str] = set()

        for source, limit in sources:
            print(f"\n  [{source.name}] Streaming (limit={limit or 'unlimited'})...")
            source_count = 0

            iterator = source.stream(limit=limit)
            if config.show_progress and limit:
                iterator = tqdm(iterator, desc=f"  {source.name}", total=limit)

            for ex in iterator:
                stats.total_streamed += 1

                # Quality filter
                if len(ex.query) < config.min_query_length:
                    stats.quality_filtered += 1
                    continue
                if len(ex.positive) < config.min_positive_length:
                    stats.quality_filtered += 1
                    continue

                # Dedup by query text hash
                query_hash = hashlib.md5(ex.query.encode()).hexdigest()
                if query_hash in seen_hashes:
                    stats.duplicates_removed += 1
                    continue
                seen_hashes.add(query_hash)

                all_examples.append(ex)
                source_count += 1

            stats.per_source[source.name] = source_count
            print(f"  [{source.name}] Got {source_count:,} examples")

        print(f"\nTotal unique examples: {len(all_examples):,}")

        # Phase 2: Split into train/dev
        print("\nPhase 2: Splitting into train/dev...")
        rng = random.Random(config.seed)
        rng.shuffle(all_examples)

        dev_count = max(1, int(len(all_examples) * config.dev_fraction))
        dev_examples = all_examples[:dev_count]
        train_examples = all_examples[dev_count:]

        stats.train_examples = len(train_examples)
        stats.dev_examples = len(dev_examples)

        # Phase 3: Save
        print("\nPhase 3: Saving...")
        train_path = config.output_dir / "train.jsonl"
        dev_path = config.output_dir / "dev.jsonl"
        info_path = config.output_dir / "pipeline_info.json"

        with open(train_path, "w") as f:
            for ex in train_examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

        with open(dev_path, "w") as f:
            for ex in dev_examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

        # Save metadata
        info = {
            "per_source": stats.per_source,
            "total_streamed": stats.total_streamed,
            "duplicates_removed": stats.duplicates_removed,
            "quality_filtered": stats.quality_filtered,
            "train_examples": stats.train_examples,
            "dev_examples": stats.dev_examples,
            "config": {
                "dev_fraction": config.dev_fraction,
                "seed": config.seed,
                "min_query_length": config.min_query_length,
                "min_positive_length": config.min_positive_length,
            },
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        print("\nOutput files:")
        print(f"  {train_path} ({stats.train_examples:,} examples)")
        print(f"  {dev_path} ({stats.dev_examples:,} examples)")
        print(f"  {info_path}")
        print(f"\n{stats.summary()}")

        return stats
