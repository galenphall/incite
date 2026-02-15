"""Training script for fine-tuning MiniLM on citation retrieval.

Uses CachedMultipleNegativesRankingLoss with in-batch + explicit hard negatives.
Compatible with Apple Silicon (MPS) via fp32 and no multiprocessing.
"""

import json
from pathlib import Path

from incite.finetuning.data_preparation import TrainingExample, load_training_data


def _build_ir_evaluator(dev_examples: list[TrainingExample]):
    """Build an InformationRetrievalEvaluator from dev examples.

    Groups dev examples by source paper to create per-query evaluations.
    Each query maps to its positive paper, and the corpus is all unique papers.
    """
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}

    for i, ex in enumerate(dev_examples):
        qid = f"q_{i}"
        # Use unique doc ID to avoid overwrites when multiple examples
        # share the same cited_paper_id
        doc_id = f"{ex.cited_paper_id}_{i}" if ex.cited_paper_id else f"doc_{i}"

        queries[qid] = ex.query
        corpus[doc_id] = ex.positive
        relevant_docs[qid] = {doc_id}

        # Add hard negatives to corpus
        for j, neg in enumerate(ex.hard_negatives):
            neg_id = f"neg_{i}_{j}"
            corpus[neg_id] = neg

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="dev",
        show_progress_bar=False,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[10],
    )


def train(
    train_path: Path,
    dev_path: Path,
    output_dir: Path,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    pretrained_model: str | None = None,
    general_data_path: Path | None = None,
    general_data_ratio: float = 0.5,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    eval_steps: int = 200,
    mini_batch_size: int = 32,
    save_total_limit: int = 3,
    use_cached_mnrl: bool | None = None,
    max_hard_negatives: int = 3,
    matryoshka_dims: list[int] | None = None,
    early_stopping_patience: int | None = None,
    seed: int = 42,
    show_progress: bool = True,
) -> dict:
    """Fine-tune MiniLM on citation retrieval pairs.

    Args:
        train_path: Path to train.jsonl
        dev_path: Path to dev.jsonl
        output_dir: Directory to save fine-tuned model
        base_model: HuggingFace model name for base (used if pretrained_model is None)
        pretrained_model: Path to domain-pretrained model (from pretrain.py).
            If set, used instead of base_model. Enables the OpenScholar two-stage
            training: TSDAE pretraining -> supervised contrastive training.
        general_data_path: Path to general-domain training data (NLI/STS-B format).
            If set, interleaved with citation data at general_data_ratio to prevent
            catastrophic forgetting (OpenScholar finding #6).
        general_data_ratio: Fraction of general data in each batch (default: 0.5).
            Only used when general_data_path is set.
        epochs: Number of training epochs
        batch_size: Training batch size (effective)
        learning_rate: Peak learning rate
        warmup_ratio: Fraction of steps for warmup
        weight_decay: Weight decay for AdamW
        eval_steps: Evaluate every N steps
        mini_batch_size: Mini-batch size for gradient caching (memory control)
        use_cached_mnrl: Use CachedMultipleNegativesRankingLoss. None = auto (True on CUDA).
        save_total_limit: Maximum checkpoints to keep
        matryoshka_dims: Dimensions for MatryoshkaLoss (e.g. [128, 256, 384]).
            If set, wraps the base loss to regularize lower-dimensional embeddings.
            V4 used [128, 256, 384] — enables two-stage Matryoshka search.
        early_stopping_patience: Number of eval steps without improvement before
            stopping. If set, adds EarlyStoppingCallback. V4 used patience=3.
        seed: Random seed
        show_progress: Show progress bars

    Returns:
        Stats dict with training summary
    """
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import BatchSamplers

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

    # Mix in general-domain data to prevent catastrophic forgetting
    general_examples: list[TrainingExample] = []
    if general_data_path and general_data_path.exists():
        general_examples = load_training_data(general_data_path)
        if general_examples:
            # Sample to target ratio: general / (general + domain) = ratio
            ratio = general_data_ratio / (1 - general_data_ratio)
            target_general = int(len(train_examples) * ratio)
            if len(general_examples) > target_general:
                import random as _random

                _random.seed(seed)
                general_examples = _random.sample(general_examples, target_general)
            print(
                f"Mixing {len(general_examples)} general-domain examples "
                f"(ratio={general_data_ratio:.0%})"
            )

    # Convert to sentence-transformers format with consistent column count.
    # Cap hard negatives. With CachedMNRL, mini_batch_size controls GPU memory
    # so more negatives are fine (5-6). With regular MNRL, cap at 3 to fit in
    # 24GB at batch_size=128. First negatives are the hardest/most informative.
    max_negs = max_hard_negatives
    print(f"Converting to training format (max_negs={max_negs})...")
    train_dataset = _examples_to_dataset(train_examples, max_negs=max_negs)

    if general_examples:
        general_dataset = _examples_to_dataset(general_examples, max_negs=max_negs)
        from datasets import concatenate_datasets

        train_dataset = concatenate_datasets([train_dataset, general_dataset])
        train_dataset = train_dataset.shuffle(seed=seed)

    # Load model — use pretrained model if available (two-stage training)
    device = get_best_device()
    effective_model = pretrained_model if pretrained_model else base_model
    print(f"Loading model: {effective_model} (device: {device})")
    if pretrained_model:
        print(f"  (domain-pretrained from {base_model})")
    model = SentenceTransformer(effective_model, device=device)

    # Build evaluator
    print("Building dev evaluator...")
    evaluator = _build_ir_evaluator(dev_examples)

    # Configure loss — CachedMNRL is faster and more memory-efficient on CUDA
    # but incompatible with MPS (torch.mps.device not implemented).
    if use_cached_mnrl is None:
        use_cached_mnrl = device == "cuda"

    if use_cached_mnrl:
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

        loss = CachedMultipleNegativesRankingLoss(model=model, mini_batch_size=mini_batch_size)
        loss_name = "CachedMultipleNegativesRankingLoss"
    else:
        loss = MultipleNegativesRankingLoss(model=model)
        loss_name = "MultipleNegativesRankingLoss"

    # Wrap in MatryoshkaLoss if dims specified (V4 used [128, 256, 384])
    if matryoshka_dims:
        from sentence_transformers.losses import MatryoshkaLoss

        loss = MatryoshkaLoss(model=model, loss=loss, matryoshka_dims=matryoshka_dims)
        loss_name = f"Matryoshka({loss_name}, dims={matryoshka_dims})"

    # Output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mixed precision: bf16 preferred on CUDA (native on Ampere+, no loss scaling),
    # fp16 fallback, neither on MPS
    is_mps = device == "mps"
    use_bf16 = device == "cuda"
    use_fp16 = False  # bf16 is strictly better on modern GPUs

    # Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="dev_cosine_ndcg@10",
        seed=seed,
        dataloader_num_workers=0 if is_mps else 4,
        dataloader_pin_memory=not is_mps,
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
        logging_steps=50,
        report_to="none",  # No wandb/tensorboard
    )

    # Early stopping callback (V4 used patience=3)
    callbacks = []
    if early_stopping_patience is not None:
        from transformers import EarlyStoppingCallback

        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=callbacks if callbacks else None,
    )

    # Train
    print("\nStarting training:")
    print(f"  Model: {effective_model}")
    if pretrained_model:
        print(f"  Base: {base_model} (domain-pretrained)")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Dev examples: {len(dev_examples)}")
    if general_data_path and general_data_path.exists():
        print(f"  General data: {general_data_path} (ratio={general_data_ratio:.0%})")
    print(f"  Loss: {loss_name}")
    if matryoshka_dims:
        print(f"  Matryoshka dims: {matryoshka_dims}")
    if early_stopping_patience is not None:
        print(f"  Early stopping: patience={early_stopping_patience}")
    print(f"  FP16: {use_fp16}")
    print()

    trainer.train()

    # Save the best model to final location
    final_model_dir = output_dir / "final"
    print(f"\nSaving best model to {final_model_dir}...")
    model.save(str(final_model_dir))

    # Evaluate final model
    print("Running final evaluation on dev set...")
    final_metrics = evaluator(model)

    stats = {
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device,
        "model_dir": str(final_model_dir),
        "final_metrics": final_metrics,
    }

    # Save training stats
    stats_path = output_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print("\nTraining complete!")
    print(f"  Model saved to: {final_model_dir}")
    print(f"  Stats saved to: {stats_path}")
    if final_metrics:
        for key, value in sorted(final_metrics.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

    return stats


def _examples_to_dataset(examples: list[TrainingExample], max_negs: int | None = None):
    """Convert TrainingExamples to a HuggingFace Dataset for sentence-transformers.

    Uses the format expected by CachedMultipleNegativesRankingLoss:
    columns named "anchor", "positive", and optionally "negative_1", "negative_2", etc.

    Args:
        examples: Training examples to convert.
        max_negs: Number of negative columns to create. If None, computed from
            examples. Pass explicitly when concatenating datasets with different
            numbers of hard negatives to avoid None-filled columns.
    """
    from datasets import Dataset

    if max_negs is None:
        max_negs = max((len(ex.hard_negatives) for ex in examples), default=0)

    records = []
    for ex in examples:
        record = {
            "anchor": ex.query,
            "positive": ex.positive,
        }
        negs = ex.hard_negatives[:max_negs] if ex.hard_negatives else []
        for i in range(max_negs):
            if i < len(negs):
                record[f"negative_{i + 1}"] = negs[i]
            elif negs:
                # Round-robin duplicate existing negatives instead of empty strings.
                # Empty strings are trivially distinguishable and waste training signal.
                record[f"negative_{i + 1}"] = negs[i % len(negs)]
            else:
                record[f"negative_{i + 1}"] = ""  # No negatives at all
        records.append(record)

    return Dataset.from_list(records)
