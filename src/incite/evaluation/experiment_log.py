"""Experiment logging for tracking evaluation runs."""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from incite.models import EvaluationResult, QueryResult


def compute_file_hash(path: str) -> str:
    """Compute SHA256 hash of a file for reproducibility tracking.

    Args:
        path: Path to the file

    Returns:
        SHA256 hash as hex string
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    method: str  # "neural", "bm25", "hybrid"
    model_name: str  # e.g. "allenai/specter2_base"
    fusion: Optional[str]  # "rrf", "weighted", or None
    k: int  # retrieval cutoff
    scale: str  # "local", "section", "global"
    dataset_name: str  # name/path of test set
    dataset_hash: str  # SHA256 of test set file
    extra: dict = field(default_factory=dict)  # additional config params

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        return cls(**data)


@dataclass
class ExperimentRun:
    """A single experiment run with config and results."""

    id: str
    timestamp: str  # ISO format
    config: ExperimentConfig
    results: dict  # from EvaluationResult.to_dict()
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "results": self.results,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentRun":
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            config=ExperimentConfig.from_dict(data["config"]),
            results=data["results"],
            notes=data.get("notes"),
        )


class ExperimentLogger:
    """Logger for tracking experiment runs."""

    def __init__(self, log_path: str = "data/experiments/experiments.jsonl"):
        self.log_path = Path(log_path)

    def log(
        self,
        config: ExperimentConfig,
        results: EvaluationResult,
        notes: Optional[str] = None,
    ) -> ExperimentRun:
        """Log an experiment run.

        Args:
            config: Experiment configuration
            results: Evaluation results
            notes: Optional notes about the run

        Returns:
            The logged ExperimentRun
        """
        run = ExperimentRun(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            config=config,
            results=results.to_dict(),
            notes=notes,
        )

        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to JSONL file
        with open(self.log_path, "a") as f:
            f.write(json.dumps(run.to_dict()) + "\n")

        return run

    def list_runs(self, limit: int = 20) -> list[ExperimentRun]:
        """List recent experiment runs.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of ExperimentRun objects (most recent first)
        """
        if not self.log_path.exists():
            return []

        runs = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    runs.append(ExperimentRun.from_dict(json.loads(line)))

        # Return most recent first
        return list(reversed(runs[-limit:]))

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get a specific run by ID.

        Args:
            run_id: The run ID (or prefix)

        Returns:
            ExperimentRun if found, None otherwise
        """
        if not self.log_path.exists():
            return None

        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if data["id"].startswith(run_id):
                        return ExperimentRun.from_dict(data)

        return None

    def delete_run(self, run_id: str) -> bool:
        """Delete a specific run by ID.

        Removes the run from the JSONL log and deletes any per-query data.

        Args:
            run_id: The run ID (or prefix)

        Returns:
            True if a run was deleted, False if not found
        """
        if not self.log_path.exists():
            return False

        lines = self.log_path.read_text().splitlines()
        kept = []
        deleted = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data["id"].startswith(run_id):
                deleted = True
                # Remove per-query data if it exists
                per_query_file = self.log_path.parent / "per_query" / f"{data['id']}.jsonl"
                if per_query_file.exists():
                    per_query_file.unlink()
            else:
                kept.append(line)

        if deleted:
            self.log_path.write_text("\n".join(kept) + "\n" if kept else "")

        return deleted

    def compare(self, run_ids: list[str]) -> str:
        """Compare multiple runs in a formatted table.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Formatted comparison table as string
        """
        runs = []
        for rid in run_ids:
            run = self.get_run(rid)
            if run:
                runs.append(run)
            else:
                return f"Run not found: {rid}"

        if not runs:
            return "No runs to compare"

        # Build comparison table
        lines = []

        # Header
        header = ["Metric"] + [r.id for r in runs]
        col_widths = [max(12, len(h)) for h in header]

        lines.append(" | ".join(h.ljust(w) for h, w in zip(header, col_widths)))
        lines.append("-+-".join("-" * w for w in col_widths))

        # Config rows
        config_fields = ["method", "model_name", "fusion", "scale"]
        for field_name in config_fields:
            row = [field_name]
            for run in runs:
                val = getattr(run.config, field_name)
                row.append(str(val) if val else "-")
            lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

        lines.append("-+-".join("-" * w for w in col_widths))

        # Result rows
        result_fields = [
            "recall@1",
            "recall@5",
            "recall@10",
            "recall@20",
            "recall@50",
            "mrr",
            "ndcg@10",
            "concordance",
            "skill_mrr",
            "num_queries",
        ]
        for field_name in result_fields:
            row = [field_name]
            for run in runs:
                val = run.results.get(field_name, 0)
                if isinstance(val, float):
                    row.append(f"{val:.3f}")
                else:
                    row.append(str(val))
            lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

        return "\n".join(lines)

    def save_per_query(self, run_id: str, query_results: list[QueryResult]) -> Path:
        """Save per-query results for a run.

        Args:
            run_id: The experiment run ID
            query_results: List of QueryResult objects

        Returns:
            Path to the saved file
        """
        per_query_dir = self.log_path.parent / "per_query"
        per_query_dir.mkdir(parents=True, exist_ok=True)
        out_path = per_query_dir / f"{run_id}.jsonl"

        with open(out_path, "w") as f:
            for qr in query_results:
                f.write(json.dumps(qr.to_dict()) + "\n")

        return out_path

    def load_per_query(self, run_id: str) -> list[QueryResult]:
        """Load per-query results for a run.

        Supports prefix-match on run_id.

        Args:
            run_id: The experiment run ID (or prefix)

        Returns:
            List of QueryResult objects
        """
        per_query_dir = self.log_path.parent / "per_query"
        if not per_query_dir.exists():
            return []

        # Try exact match first, then prefix match
        exact = per_query_dir / f"{run_id}.jsonl"
        if exact.exists():
            target = exact
        else:
            matches = sorted(per_query_dir.glob(f"{run_id}*.jsonl"))
            if not matches:
                return []
            target = matches[0]

        results = []
        with open(target) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(QueryResult.from_dict(json.loads(line)))
        return results

    def diff_runs(
        self,
        run_id_a: str,
        run_id_b: str,
        metric: str = "recall@10",
        top_n: int = 10,
    ) -> str:
        """Compare per-query results between two runs.

        Args:
            run_id_a: First run ID (baseline)
            run_id_b: Second run ID (comparison)
            metric: Metric to compare (e.g. "recall@10")
            top_n: Number of top improved/regressed queries to show

        Returns:
            Formatted diff report as string
        """
        from incite.evaluation.metrics import paired_bootstrap_test

        qrs_a = self.load_per_query(run_id_a)
        qrs_b = self.load_per_query(run_id_b)

        if not qrs_a:
            return f"No per-query data found for run {run_id_a}"
        if not qrs_b:
            return f"No per-query data found for run {run_id_b}"

        # Index by context_id
        a_by_id = {qr.context_id: qr for qr in qrs_a}
        b_by_id = {qr.context_id: qr for qr in qrs_b}

        # Match by context_id
        common_ids = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))
        if not common_ids:
            return "No common queries found between runs."

        # Compute per-query deltas
        deltas = []
        scores_a = []
        scores_b = []
        improved = 0
        regressed = 0
        unchanged = 0
        for cid in common_ids:
            sa = a_by_id[cid].scores.get(metric, 0.0)
            sb = b_by_id[cid].scores.get(metric, 0.0)
            scores_a.append(sa)
            scores_b.append(sb)
            d = sb - sa
            deltas.append((cid, sa, sb, d))
            if d > 1e-9:
                improved += 1
            elif d < -1e-9:
                regressed += 1
            else:
                unchanged += 1

        # Sort by delta
        deltas.sort(key=lambda x: x[3], reverse=True)

        lines = [
            f"Diff: {run_id_a} (A) vs {run_id_b} (B)  —  metric: {metric}",
            f"Matched queries: {len(common_ids)}",
            f"  Improved:  {improved}",
            f"  Regressed: {regressed}",
            f"  Unchanged: {unchanged}",
            f"  Mean A: {sum(scores_a) / len(scores_a):.3f}  "
            f"Mean B: {sum(scores_b) / len(scores_b):.3f}",
        ]

        # Significance test
        delta, p_value, effect_size = paired_bootstrap_test(scores_a, scores_b)
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        lines.append(f"  Delta: {delta:+.4f}  p={p_value:.4f} ({sig})  Cohen's d={effect_size:.3f}")

        # Top improved
        lines.append(f"\nTop {top_n} improved (B > A):")
        for cid, sa, sb, d in deltas[:top_n]:
            if d <= 1e-9:
                break
            src = a_by_id[cid].source_paper_id or "?"
            lines.append(f"  {cid}  src={src}  A={sa:.3f} B={sb:.3f} Δ={d:+.3f}")

        # Top regressed
        lines.append(f"\nTop {top_n} regressed (A > B):")
        for cid, sa, sb, d in reversed(deltas[-top_n:]):
            if d >= -1e-9:
                break
            src = a_by_id[cid].source_paper_id or "?"
            lines.append(f"  {cid}  src={src}  A={sa:.3f} B={sb:.3f} Δ={d:+.3f}")

        return "\n".join(lines)
