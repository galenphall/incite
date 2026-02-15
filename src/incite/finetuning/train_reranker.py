"""Training script for fine-tuning a cross-encoder reranker on citation data.

V5 reranker: Full-text input + retriever-aware dev evaluator + random negatives.
V4 reranker: BGE-reranker-base (110M params) on cleaned data with cosine LR.
V3 reranker: listwise loss (LambdaLoss / RankNet) with co-citation hard negatives.

Key improvements over v4:
  - Full-text mode (--use-full-text): passes title/authors/year/journal/abstract
    to the cross-encoder instead of abstract-only. Every successful citation
    reranker in the literature uses full metadata.
  - Retriever-aware dev evaluator: when --corpus is provided, builds a dev
    evaluator using actual Granite-FT v6b top-100 candidates as negatives
    instead of co-citation negatives. Per Gao & Dai 2021, matching the negative
    distribution between training and inference is critical.
  - Random negatives (via augment_hard_negatives.py --random-negatives) prevent
    overspecialization on hard cases.

V4 adds:
  - weight_decay and lr_scheduler_type parameters
  - Support for cosine LR schedule (better for pre-trained models)

When use_full_text=False (default, v3/v4 behavior):
  Training positives/negatives are stripped of metadata prefixes to match
  inference on raw abstracts.

When use_full_text=True (v5):
  Training positives/negatives keep full format_paper_embedding_text() format.
  At inference, pass use_full_text=True to the reranker as well.

Usage:
    # v5 (full-text + retriever-aware):
    incite finetune train-reranker \\
        --train master_train_augmented.jsonl --dev master_dev.jsonl \\
        --output-dir models/reranker-v5 --use-full-text \\
        --corpus data/processed/corpus.jsonl

    # v4/v3 (abstract-only, co-citation dev):
    incite finetune train-reranker \\
        --train master_train.jsonl --dev master_dev.jsonl \\
        --output-dir models/reranker-v4 --loss lambda --max-negatives 15
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from incite.finetuning.data_preparation import TrainingExample, load_training_data
from incite.models import strip_metadata_prefix as _strip_metadata_prefix


def _build_listwise_dataset(
    examples: list[TrainingExample],
    max_negatives: int = 15,
    seed: int = 42,
    use_full_text: bool = False,
):
    """Build HuggingFace Dataset with (query, documents[], label[]) for listwise loss.

    Each sample contains one positive document and up to max_negatives negative
    documents. Self-contradictory negatives (identical to positive after stripping)
    are removed.

    When use_full_text=False (default): documents are stripped of metadata prefixes
    to match abstract-only inference.
    When use_full_text=True: documents keep full format (title/authors/year/abstract)
    for full-text inference.

    Args:
        examples: Training examples with query, positive, hard_negatives.
        max_negatives: Maximum number of negatives per example.
        seed: Random seed for sampling negatives.
        use_full_text: Keep full metadata in documents (True) or strip (False).

    Returns:
        HuggingFace Dataset with columns: query, documents, label.
    """
    from datasets import Dataset

    rng = random.Random(seed)

    queries = []
    documents_list = []
    labels_list = []
    skipped_no_negs = 0
    total_contradictions = 0

    def _format_doc(text: str) -> str:
        return text if use_full_text else _strip_metadata_prefix(text)

    for ex in examples:
        stripped_pos = _strip_metadata_prefix(ex.positive)

        # Filter contradictory negatives (always compare via stripped core text)
        clean_negs = []
        for neg in ex.hard_negatives:
            stripped_neg = _strip_metadata_prefix(neg)
            if stripped_neg != stripped_pos:
                clean_negs.append(neg)
            else:
                total_contradictions += 1

        # Deduplicate negatives by core text
        seen_cores: set[str] = set()
        deduped_negs = []
        for neg in clean_negs:
            core = _strip_metadata_prefix(neg)
            if core not in seen_cores:
                seen_cores.add(core)
                deduped_negs.append(neg)
        clean_negs = deduped_negs

        if not clean_negs:
            skipped_no_negs += 1
            continue

        # Sample negatives if too many
        if len(clean_negs) > max_negatives:
            clean_negs = rng.sample(clean_negs, max_negatives)

        # Build document list: positive first, then negatives
        formatted_pos = _format_doc(ex.positive)
        formatted_negs = [_format_doc(n) for n in clean_negs]
        docs = [formatted_pos] + formatted_negs
        labels = [1.0] + [0.0] * len(formatted_negs)

        queries.append(ex.query)
        documents_list.append(docs)
        labels_list.append(labels)

    print(f"  Built listwise dataset: {len(queries)} examples")
    print(f"  Skipped (no negatives): {skipped_no_negs}")
    print(f"  Contradictions removed: {total_contradictions}")
    print(f"  Full-text mode: {use_full_text}")
    if documents_list:
        avg_docs = sum(len(d) for d in documents_list) / len(documents_list)
        print(f"  Avg documents per query: {avg_docs:.1f}")

    return Dataset.from_dict(
        {
            "query": queries,
            "documents": documents_list,
            "label": labels_list,
        }
    )


def _build_pointwise_dataset(
    examples: list[TrainingExample],
    max_negatives: int = 3,
    seed: int = 42,
    use_full_text: bool = False,
):
    """Build HuggingFace Dataset with (query, text, label) for BCE loss.

    Fallback for pointwise binary cross-entropy training (v2-compatible).

    Args:
        examples: Training examples.
        max_negatives: Maximum negatives per example.
        seed: Random seed.
        use_full_text: Keep full metadata in documents (True) or strip (False).

    Returns:
        HuggingFace Dataset with columns: query, text, label.
    """
    from datasets import Dataset

    def _format_doc(text: str) -> str:
        return text if use_full_text else _strip_metadata_prefix(text)

    rng = random.Random(seed)
    queries = []
    texts = []
    labels = []

    for ex in examples:
        stripped_pos = _strip_metadata_prefix(ex.positive)
        queries.append(ex.query)
        texts.append(_format_doc(ex.positive))
        labels.append(1.0)

        negatives = ex.hard_negatives[:max_negatives]
        if len(ex.hard_negatives) > max_negatives:
            negatives = rng.sample(ex.hard_negatives, max_negatives)

        for neg in negatives:
            stripped_neg = _strip_metadata_prefix(neg)
            if stripped_neg != stripped_pos:
                queries.append(ex.query)
                texts.append(_format_doc(neg))
                labels.append(0.0)

    return Dataset.from_dict(
        {
            "query": queries,
            "text": texts,
            "label": labels,
        }
    )


def _build_realistic_evaluator(
    dev_examples: list[TrainingExample],
    max_samples: int = 2000,
    max_negatives: int = 15,
    seed: int = 42,
    use_full_text: bool = False,
):
    """Build a realistic CERerankingEvaluator from dev examples.

    Groups dev examples by source_paper_id and uses other positives from
    the same source as candidates. This creates a realistic evaluation
    where the model must rank among same-subfield papers.

    For examples without source_paper_id, falls back to hard_negatives.

    Args:
        dev_examples: Dev TrainingExamples.
        max_samples: Maximum number of evaluator samples.
        max_negatives: Maximum negative candidates per sample.
        seed: Random seed.
        use_full_text: Keep full metadata in documents (True) or strip (False).

    Returns:
        CERerankingEvaluator instance.
    """
    from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

    def _format_doc(text: str) -> str:
        return text if use_full_text else _strip_metadata_prefix(text)

    rng = random.Random(seed)
    samples = []

    # Group by source_paper_id for co-citation eval
    by_source: dict[str, list[TrainingExample]] = defaultdict(list)
    no_source = []
    for ex in dev_examples:
        if ex.source_paper_id:
            by_source[ex.source_paper_id].append(ex)
        else:
            no_source.append(ex)

    # Co-citation samples: use other positives from same source as negatives
    for source_id, source_examples in by_source.items():
        if len(source_examples) < 2:
            continue

        # Always deduplicate via stripped core text, format output per mode
        stripped_positives = {id(ex): _strip_metadata_prefix(ex.positive) for ex in source_examples}
        formatted_positives = {id(ex): _format_doc(ex.positive) for ex in source_examples}

        for ex in source_examples:
            my_stripped = stripped_positives[id(ex)]
            my_formatted = formatted_positives[id(ex)]
            candidates = []
            seen_cores: set[str] = set()
            for other in source_examples:
                core = stripped_positives[id(other)]
                if core != my_stripped and core not in seen_cores:
                    seen_cores.add(core)
                    candidates.append(formatted_positives[id(other)])

            if len(candidates) > max_negatives:
                candidates = rng.sample(candidates, max_negatives)

            if candidates:
                samples.append(
                    {
                        "query": ex.query,
                        "positive": [my_formatted],
                        "negative": candidates,
                    }
                )

    # Fallback samples from examples without source_paper_id
    for ex in no_source:
        stripped_pos = _strip_metadata_prefix(ex.positive)
        formatted_pos = _format_doc(ex.positive)
        negs = []
        seen_cores: set[str] = set()
        for n in ex.hard_negatives[:max_negatives]:
            core = _strip_metadata_prefix(n)
            if core != stripped_pos and core not in seen_cores:
                seen_cores.add(core)
                negs.append(_format_doc(n))
        if negs:
            samples.append(
                {
                    "query": ex.query,
                    "positive": [formatted_pos],
                    "negative": negs,
                }
            )

    # Cap and shuffle
    if len(samples) > max_samples:
        rng.shuffle(samples)
        samples = samples[:max_samples]

    avg_negs = sum(len(s["negative"]) for s in samples) / max(1, len(samples))
    print(
        f"  Dev evaluator (co-citation): {len(samples)} samples, "
        f"avg {avg_negs:.1f} negatives/sample"
    )

    return CERerankingEvaluator(samples=samples, name="dev", at_k=10, batch_size=16)


def _build_retriever_aware_evaluator(
    dev_examples: list[TrainingExample],
    corpus_path: Path,
    max_samples: int = 1000,
    top_k: int = 100,
    max_negatives: int = 20,
    seed: int = 42,
    use_full_text: bool = False,
):
    """Build a CERerankingEvaluator using actual retriever candidates as negatives.

    Instead of co-citation negatives (which don't match what the reranker sees
    at inference), this loads the Granite-FT v6b embedder, encodes the corpus,
    and retrieves top-k candidates per dev query. The ground truth positive is
    identified among candidates and up to max_negatives become evaluator negatives.

    Per Gao & Dai 2021 (LCE paper), matching the negative distribution between
    training/evaluation and inference is the #1 factor for reranker quality.

    Args:
        dev_examples: Dev TrainingExamples.
        corpus_path: Path to corpus.jsonl for building the retriever index.
        max_samples: Maximum evaluator samples (limits GPU time).
        top_k: Number of retriever candidates to retrieve per query.
        max_negatives: Maximum negatives per evaluator sample (caps GPU memory
            during eval — 20 is a good balance between realism and memory).
        seed: Random seed.
        use_full_text: Keep full metadata in documents (True) or strip (False).

    Returns:
        CERerankingEvaluator instance with retriever-realistic negatives.
    """
    import gc

    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

    from incite.corpus.loader import load_corpus

    rng = random.Random(seed)

    # Load corpus
    print("  Loading corpus for retriever-aware evaluator...")
    papers = load_corpus(corpus_path)
    print(f"  Corpus: {len(papers):,} papers")

    # Format paper texts for embedding and for evaluator output
    if use_full_text:
        paper_texts = [p.to_embedding_text() for p in papers]
    else:
        paper_texts = [p.abstract if p.abstract else p.title for p in papers]

    # Load Granite-FT v6b for retrieval
    print("  Loading Granite-FT v6b for candidate retrieval...")
    model = SentenceTransformer("models/granite-citation-v6/final")

    # Encode corpus with passage prefix (Granite asymmetric)
    print("  Encoding corpus...")
    corpus_encode_texts = [f"passage: {p.to_embedding_text()}" for p in papers]
    corpus_embs = model.encode(
        corpus_encode_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Build FAISS index
    import faiss

    faiss.omp_set_num_threads(1)
    dim = corpus_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(corpus_embs.astype(np.float32)))
    print(f"  Built FAISS index: {index.ntotal} vectors, {dim}-dim")

    # Subsample dev examples
    dev_subset = list(dev_examples)
    if len(dev_subset) > max_samples:
        rng.shuffle(dev_subset)
        dev_subset = dev_subset[:max_samples]

    # Build positive lookup: cited_paper_id -> paper text
    # For each dev query, find its ground truth positive in the corpus
    print(f"  Retrieving top-{top_k} candidates for {len(dev_subset)} dev queries...")
    queries = [ex.query for ex in dev_subset]
    query_encode_texts = [f"query: {q}" for q in queries]
    query_embs = model.encode(
        query_encode_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Free model memory
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Search
    query_embs = np.ascontiguousarray(query_embs.astype(np.float32))
    _, indices = index.search(query_embs, top_k)

    # Build evaluator samples
    samples = []
    found_positive = 0
    for i, ex in enumerate(dev_subset):
        # Identify ground truth positive
        positive_core = _strip_metadata_prefix(ex.positive)

        # Get candidate paper texts
        candidate_indices = indices[i]
        candidate_texts = []
        positive_text = None

        for idx in candidate_indices:
            if idx < 0:
                continue
            text = paper_texts[idx]
            text_core = _strip_metadata_prefix(text) if use_full_text else text.strip()
            if text_core.strip() == positive_core.strip():
                if use_full_text:
                    positive_text = text
                else:
                    positive_text = text
            else:
                candidate_texts.append(text)

        # If positive not found in top-k, use the training positive
        if positive_text is None:
            if use_full_text:
                positive_text = ex.positive
            else:
                positive_text = _strip_metadata_prefix(ex.positive)
        else:
            found_positive += 1

        # Cap negatives to control GPU memory during eval
        if len(candidate_texts) > max_negatives:
            candidate_texts = rng.sample(candidate_texts, max_negatives)

        if candidate_texts:
            samples.append(
                {
                    "query": ex.query,
                    "positive": [positive_text],
                    "negative": candidate_texts,
                }
            )

    # Free index and embeddings
    del index, corpus_embs, query_embs
    gc.collect()

    avg_negs = sum(len(s["negative"]) for s in samples) / max(1, len(samples))
    print(
        f"  Dev evaluator (retriever-aware): {len(samples)} samples, "
        f"avg {avg_negs:.1f} negatives/sample"
    )
    print(
        f"  Positive found in top-{top_k}: {found_positive}/{len(dev_subset)} "
        f"({100 * found_positive / max(1, len(dev_subset)):.1f}%)"
    )

    return CERerankingEvaluator(samples=samples, name="dev", at_k=10, batch_size=16)


def train_reranker(
    train_path: Path,
    dev_path: Path,
    output_dir: Path = Path("models/reranker-v3"),
    base_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    max_length: int = 512,
    max_negatives: int = 15,
    warmup_ratio: float = 0.1,
    eval_steps: int = 500,
    early_stopping_patience: int = 3,
    loss_type: str = "lambda",
    weight_decay: float = 0.0,
    lr_scheduler_type: str = "linear",
    seed: int = 42,
    corpus_path: Path | None = None,
    use_full_text: bool = False,
    resume_from_checkpoint: str | None = None,
) -> dict:
    """Fine-tune a cross-encoder reranker on citation retrieval data.

    Supports three loss functions:
      - "lambda": LambdaLoss with NDCGLoss2PPScheme (listwise, recommended)
      - "ranknet": RankNetLoss (pairwise)
      - "bce": BinaryCrossEntropyLoss (pointwise, v2-compatible)

    When use_full_text=False (default, v3/v4 behavior):
      All text is stripped of metadata prefixes to match abstract-only inference.
    When use_full_text=True (v5):
      Training data keeps full format_paper_embedding_text() format. At inference,
      pass use_full_text=True to the reranker as well.

    When corpus_path points to a real file and the Granite-FT v6 model exists,
    the dev evaluator uses actual retriever candidates as negatives instead of
    co-citation negatives (retriever-aware evaluator).

    Args:
        train_path: Path to train.jsonl (TrainingExample format).
        dev_path: Path to dev.jsonl (TrainingExample format).
        output_dir: Directory to save fine-tuned model.
        base_model: Base cross-encoder model name.
        epochs: Number of training epochs.
        batch_size: Training batch size (per device).
        learning_rate: Peak learning rate.
        max_length: Maximum sequence length for cross-encoder.
        max_negatives: Maximum negatives per training example.
        warmup_ratio: Fraction of steps for warmup.
        eval_steps: Evaluate every N steps.
        early_stopping_patience: Early stopping patience on dev MRR.
        loss_type: Loss function: "lambda", "ranknet", or "bce".
        weight_decay: Weight decay for regularization (default: 0.0).
        lr_scheduler_type: LR scheduler type: "linear" or "cosine" (default: "linear").
        seed: Random seed.
        corpus_path: Path to corpus.jsonl. When provided and the retriever model
            exists, enables retriever-aware dev evaluator with realistic negatives.
        use_full_text: If True, keep full metadata (title/authors/year/abstract)
            in training documents instead of stripping to abstract-only.
        resume_from_checkpoint: Path to checkpoint directory to resume from.

    Returns:
        Stats dict with training summary.
    """
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
    from transformers import EarlyStoppingCallback

    from incite.utils import get_best_device

    # Load data
    print(f"Loading training data from {train_path}...")
    train_examples = load_training_data(train_path)
    print(f"Loaded {len(train_examples)} training examples")

    print(f"Loading dev data from {dev_path}...")
    dev_examples = load_training_data(dev_path)
    print(f"Loaded {len(dev_examples)} dev examples")

    if not train_examples:
        return {"error": "No training examples found"}

    # Build datasets based on loss type
    is_listwise = loss_type in ("lambda", "ranknet")

    print(f"\nBuilding {'listwise' if is_listwise else 'pointwise'} dataset...")
    if is_listwise:
        train_dataset = _build_listwise_dataset(
            train_examples,
            max_negatives=max_negatives,
            seed=seed,
            use_full_text=use_full_text,
        )
    else:
        train_dataset = _build_pointwise_dataset(
            train_examples,
            max_negatives=max_negatives,
            seed=seed,
            use_full_text=use_full_text,
        )

    # Build dev evaluator — prefer retriever-aware if corpus + model available
    print("Building dev evaluator...")
    use_retriever_eval = (
        corpus_path is not None
        and Path(corpus_path).is_file()
        and Path("models/granite-citation-v6/final").is_dir()
    )
    if use_retriever_eval:
        print("  Using retriever-aware evaluator (Granite-FT v6b candidates)")
        evaluator = _build_retriever_aware_evaluator(
            dev_examples,
            corpus_path=Path(corpus_path),
            seed=seed,
            use_full_text=use_full_text,
        )
    else:
        if corpus_path and not Path(corpus_path).is_file():
            print(f"  WARNING: corpus not found at {corpus_path}, using co-citation evaluator")
        if not Path("models/granite-citation-v6/final").is_dir():
            print("  WARNING: Granite-FT v6 model not found, using co-citation evaluator")
        evaluator = _build_realistic_evaluator(
            dev_examples,
            max_negatives=max_negatives,
            seed=seed,
            use_full_text=use_full_text,
        )

    # Create output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize cross-encoder
    device = get_best_device()
    print(f"\nLoading base model: {base_model} (device: {device})")
    model = CrossEncoder(
        base_model,
        num_labels=1,
        max_length=max_length,
        device=device,
    )

    # Create loss function
    if loss_type == "lambda":
        from sentence_transformers.cross_encoder.losses import LambdaLoss

        loss = LambdaLoss(model=model)
        loss_name = "LambdaLoss (NDCGLoss2PPScheme)"
    elif loss_type == "ranknet":
        from sentence_transformers.cross_encoder.losses import RankNetLoss

        loss = RankNetLoss(model=model)
        loss_name = "RankNetLoss"
    elif loss_type == "bce":
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

        loss = BinaryCrossEntropyLoss(model=model)
        loss_name = "BinaryCrossEntropyLoss"
    else:
        return {"error": f"Unknown loss type: {loss_type}"}

    # Determine training settings
    num_train_steps = (len(train_dataset) // batch_size) * epochs

    print("\nStarting cross-encoder training:")
    print(f"  Model: {base_model}")
    print(f"  Device: {device}")
    print(f"  Loss: {loss_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max length: {max_length}")
    print(f"  Max negatives: {max_negatives}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  LR scheduler: {lr_scheduler_type}")
    print(f"  Full-text mode: {use_full_text}")
    print(f"  Dev evaluator: {'retriever-aware' if use_retriever_eval else 'co-citation'}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Total steps: {num_train_steps}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print()

    # Training arguments
    training_args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dev_mrr@10",
        greater_is_better=True,
        bf16=True,
        dataloader_num_workers=4 if str(device) != "mps" else 0,
        logging_steps=50,
        seed=seed,
    )

    # Callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
    ]

    # Create trainer
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=callbacks,
    )

    # Train
    if resume_from_checkpoint:
        print(f"  Resuming from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    final_dir = output_dir / "final"
    model.save(str(final_dir))

    # Run final evaluation
    print("\nRunning final evaluation...")
    final_score = evaluator(model, str(output_dir))

    stats = {
        "train_samples": len(train_dataset),
        "dev_evaluator_samples": len(evaluator.samples),
        "dev_evaluator_type": "retriever-aware" if use_retriever_eval else "co-citation",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "max_negatives": max_negatives,
        "weight_decay": weight_decay,
        "lr_scheduler_type": lr_scheduler_type,
        "use_full_text": use_full_text,
        "loss_type": loss_type,
        "device": str(device),
        "base_model": base_model,
        "model_dir": str(final_dir),
        "final_score": final_score,
    }

    # Save training stats
    stats_path = output_dir / "training_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str))

    print("\nTraining complete!")
    print(f"  Model saved to: {final_dir}")
    print(f"  Stats saved to: {stats_path}")
    if isinstance(final_score, (int, float)):
        print(f"  Dev MRR@10: {final_score:.4f}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune cross-encoder reranker")
    parser.add_argument(
        "--train", default="data/finetuning/master_train.jsonl", help="Training data path"
    )
    parser.add_argument("--dev", default="data/finetuning/master_dev.jsonl", help="Dev data path")
    parser.add_argument("--output", default="models/reranker-v3", help="Output directory")
    parser.add_argument(
        "--base-model",
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        help="Base cross-encoder model",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--max-negatives", type=int, default=15, help="Max negatives per example")
    parser.add_argument(
        "--loss",
        default="lambda",
        choices=["lambda", "ranknet", "bce"],
        help="Loss function (default: lambda)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for regularization (default: 0.0)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="LR scheduler type (default: linear)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience (default: 3)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup fraction of total steps (default: 0.1)",
    )
    parser.add_argument(
        "--corpus",
        default="data/processed/corpus.jsonl",
        help="Path to corpus.jsonl. Enables retriever-aware dev evaluator "
        "when the Granite-FT v6 model is available.",
    )
    parser.add_argument(
        "--use-full-text",
        action="store_true",
        default=False,
        help="Keep full metadata (title/authors/year/abstract) in training "
        "documents instead of stripping to abstract-only (v5+).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from.",
    )

    args = parser.parse_args()
    train_reranker(
        train_path=Path(args.train),
        dev_path=Path(args.dev),
        output_dir=Path(args.output),
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        max_negatives=args.max_negatives,
        warmup_ratio=args.warmup_ratio,
        loss_type=args.loss,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.scheduler,
        early_stopping_patience=args.early_stopping_patience,
        corpus_path=Path(args.corpus),
        use_full_text=args.use_full_text,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
