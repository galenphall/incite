"""Fine-tuning commands: prepare, train, generate-passages, eval-passages, status, validate."""

import os
import sys
from pathlib import Path

from incite.cli._shared import EMBEDDER_CHOICES
from incite.utils import DEFAULT_LLM_MODEL


def register(subparsers):
    """Register the finetune command group."""
    ft_parser = subparsers.add_parser("finetune", help="Fine-tune embedder models on citation data")
    ft_subparsers = ft_parser.add_subparsers(dest="ft_command", help="Finetune subcommands")

    _register_prepare(ft_subparsers)
    _register_train(ft_subparsers)
    _register_train_reranker(ft_subparsers)
    _register_generate_passages(ft_subparsers)
    _register_generate_fulltext_passages(ft_subparsers)
    _register_eval_passages(ft_subparsers)
    _register_status(ft_subparsers)
    _register_validate(ft_subparsers)

    ft_parser.set_defaults(func=cmd_finetune)


def _register_prepare(subparsers):
    p = subparsers.add_parser("prepare", help="Prepare training data for fine-tuning")
    p.add_argument(
        "--from-existing",
        action="store_true",
        default=True,
        help="From existing test set + corpus (fast, default)",
    )
    p.add_argument(
        "--from-unarxiv",
        action="store_true",
        help="Mine new training data from unarXiv (slow, requires API calls)",
    )
    p.add_argument(
        "--from-all",
        action="store_true",
        help="Combine all available sources (S2ORC + synthetic + existing)",
    )
    p.add_argument(
        "--from-s2orc",
        action="store_true",
        help="Stream training data from S2ORC on HuggingFace (fast, no API calls)",
    )
    p.add_argument(
        "--from-synthetic",
        action="store_true",
        help="Use synthetic citation contexts from Zotero library",
    )
    p.add_argument(
        "--from-scicite",
        action="store_true",
        help="Use SciCite dataset (requires S2 API for abstracts)",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/unarxiv",
        help="Directory containing unarXiv JSONL files (for --from-unarxiv)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="data/finetuning",
        help="Directory for train.jsonl and dev.jsonl (default: data/finetuning)",
    )
    p.add_argument(
        "--test-set",
        type=str,
        default="data/processed/test_set.jsonl",
        help="Path to test set JSONL",
    )
    p.add_argument(
        "--corpus",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL (for --from-existing)",
    )
    p.add_argument(
        "--target",
        type=int,
        default=500,
        help="Target source papers to mine (for --from-unarxiv, default: 500)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Per-source example limit (0 = use defaults, e.g., 100K for S2ORC)",
    )
    p.add_argument(
        "--max-negatives", type=int, default=5, help="Max hard negatives per example (default: 5)"
    )
    p.add_argument(
        "--dev-fraction",
        type=float,
        default=0.1,
        help="Fraction of examples for dev set (default: 0.1)",
    )
    p.add_argument("--email", type=str, help="Email for OpenAlex polite pool (for --from-unarxiv)")


def _register_train(subparsers):
    p = subparsers.add_parser("train", help="Train fine-tuned model")
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/finetuning",
        help="Directory with train.jsonl and dev.jsonl",
    )
    p.add_argument(
        "--train",
        type=str,
        default=None,
        dest="train_file",
        help="Training JSONL file (relative to --data-dir, default: train.jsonl)",
    )
    p.add_argument(
        "--dev",
        type=str,
        default=None,
        dest="dev_file",
        help="Dev JSONL file (relative to --data-dir, default: dev.jsonl)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="models/minilm-citation-v1",
        help="Directory to save fine-tuned model (default: models/minilm-citation-v1)",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model to fine-tune",
    )
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    p.add_argument("--batch-size", type=int, default=64, help="Training batch size (default: 64)")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    p.add_argument(
        "--eval-steps", type=int, default=200, help="Evaluate every N steps (default: 200)"
    )
    p.add_argument(
        "--mini-batch-size",
        type=int,
        default=32,
        help="Mini-batch size for gradient caching (default: 32, lower = less memory)",
    )
    p.add_argument(
        "--no-cached-mnrl",
        action="store_true",
        default=False,
        help="Use standard MNRL instead of CachedMNRL (faster for small models)",
    )
    p.add_argument(
        "--max-hard-negatives",
        type=int,
        default=3,
        help="Max hard negatives per example (default: 3, reduces memory/compute)",
    )
    p.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Path to domain-pretrained model (from pretrain.py, enables two-stage training)",
    )
    p.add_argument(
        "--general-data",
        type=str,
        default=None,
        help="Path to general-domain training data JSONL (e.g. AllNLI)",
    )
    p.add_argument(
        "--general-data-ratio",
        type=float,
        default=0.5,
        help="Fraction of general data in training mix (default: 0.5)",
    )
    p.add_argument(
        "--matryoshka-dims",
        type=str,
        default=None,
        help="Comma-separated Matryoshka loss dimensions (e.g. '128,256,384'). "
        "Regularizes lower-dimensional embeddings for multi-scale retrieval.",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Early stopping patience (number of eval steps without improvement). "
        "V4 used patience=3.",
    )


def _register_train_reranker(subparsers):
    p = subparsers.add_parser(
        "train-reranker", help="Train cross-encoder reranker on citation data"
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/finetuning",
        help="Directory with train.jsonl and dev.jsonl",
    )
    p.add_argument(
        "--train",
        type=str,
        default=None,
        dest="train_file",
        help="Training JSONL file (relative to --data-dir, default: master_train.jsonl)",
    )
    p.add_argument(
        "--dev",
        type=str,
        default=None,
        dest="dev_file",
        help="Dev JSONL file (relative to --data-dir, default: master_dev.jsonl)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="models/reranker-v3",
        help="Directory to save fine-tuned reranker (default: models/reranker-v3)",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        help="Base cross-encoder model",
    )
    p.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2)")
    p.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16)")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    p.add_argument("--max-length", type=int, default=512, help="Max sequence length (default: 512)")
    p.add_argument(
        "--max-negatives",
        type=int,
        default=15,
        help="Max hard negatives per example (default: 15)",
    )
    p.add_argument(
        "--eval-steps", type=int, default=500, help="Evaluate every N steps (default: 500)"
    )
    p.add_argument(
        "--loss",
        type=str,
        default="lambda",
        choices=["lambda", "ranknet", "bce"],
        help="Loss: lambda (listwise NDCG), ranknet (pairwise), bce (pointwise)",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for regularization (default: 0.0)",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="LR scheduler type (default: linear)",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience (default: 3)",
    )
    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup fraction of total steps (default: 0.1)",
    )
    p.add_argument(
        "--corpus",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus.jsonl. Enables retriever-aware dev evaluator "
        "when the Granite-FT v6 model is available.",
    )
    p.add_argument(
        "--use-full-text",
        action="store_true",
        default=False,
        help="Keep full metadata (title/authors/year/abstract) in training "
        "documents instead of stripping to abstract-only (v5+).",
    )


def _register_generate_passages(subparsers):
    p = subparsers.add_parser(
        "generate-passages", help="Generate passage-level training data using LLM"
    )
    p.add_argument(
        "--chunks",
        type=str,
        default="data/processed/chunks.jsonl",
        help="Path to chunks JSONL file",
    )
    p.add_argument(
        "--corpus",
        type=str,
        default="data/processed/corpus.jsonl",
        help="Path to corpus JSONL file",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="data/finetuning",
        help="Directory for output files (default: data/finetuning)",
    )
    p.add_argument(
        "--max-per-paper", type=int, default=10, help="Maximum passages per paper (default: 10)"
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"Model to use (default: {DEFAULT_LLM_MODEL})",
    )
    p.add_argument(
        "--api-key", type=str, help="Anthropic API key (or use ANTHROPIC_API_KEY env var)"
    )
    p.add_argument(
        "--eval-fraction",
        type=float,
        default=0.05,
        help="Fraction of papers for eval set (default: 0.05)",
    )
    p.add_argument(
        "--dev-fraction",
        type=float,
        default=0.1,
        help="Fraction of papers for dev set (default: 0.1)",
    )
    p.add_argument(
        "--limit", type=int, default=0, help="Limit to first N papers (0 = unlimited, for testing)"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Show passage selection stats without calling API"
    )


def _register_generate_fulltext_passages(subparsers):
    p = subparsers.add_parser(
        "generate-fulltext-passages",
        help="Generate passage-level training data from full-text corpora (unarXiv/S2ORC)",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/unarxiv",
        help="Directory with JSONL files (unarXiv or S2ORC layout)",
    )
    p.add_argument(
        "--target-papers",
        type=int,
        default=5000,
        help="Number of papers to process (default: 5000)",
    )
    p.add_argument(
        "--max-per-paper", type=int, default=3, help="Maximum passages per paper (default: 3)"
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="data/finetuning",
        help="Directory for output files (default: data/finetuning)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"LLM model to use (default: {DEFAULT_LLM_MODEL})",
    )
    p.add_argument(
        "--api-key", type=str, help="Anthropic API key (or use ANTHROPIC_API_KEY env var)"
    )
    p.add_argument(
        "--eval-fraction",
        type=float,
        default=0.05,
        help="Fraction of papers for eval set (default: 0.05)",
    )
    p.add_argument(
        "--dev-fraction",
        type=float,
        default=0.1,
        help="Fraction of papers for dev set (default: 0.1)",
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output filenames (default: derived from data-dir name, "
        "e.g. 'unarxiv' -> unarxiv_passage_train.jsonl)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Show passage selection stats without calling API"
    )


def _register_eval_passages(subparsers):
    p = subparsers.add_parser("eval-passages", help="Evaluate passage-level retrieval")
    p.add_argument(
        "--test-set",
        type=str,
        default="data/processed/passage_test_set.jsonl",
        help="Path to passage test set JSONL",
    )
    p.add_argument(
        "--chunks",
        type=str,
        default="data/processed/chunks.jsonl",
        help="Path to chunks JSONL file",
    )
    p.add_argument(
        "--embedder",
        type=str,
        choices=EMBEDDER_CHOICES,
        default="minilm",
        help="Embedder for passage retrieval (default: minilm)",
    )


def _register_status(subparsers):
    subparsers.add_parser("status", help="Show training data and model status")


def _register_validate(subparsers):
    p = subparsers.add_parser("validate", help="Validate training data quality")
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/finetuning",
        help="Directory with training data JSONL files (default: data/finetuning)",
    )


# --- Command handlers ---


def cmd_finetune(args):
    """Dispatch finetune subcommands."""
    if args.ft_command == "prepare":
        cmd_finetune_prepare(args)
    elif args.ft_command == "train":
        cmd_finetune_train(args)
    elif args.ft_command == "train-reranker":
        cmd_finetune_train_reranker(args)
    elif args.ft_command == "generate-passages":
        cmd_finetune_generate_passages(args)
    elif args.ft_command == "generate-fulltext-passages":
        cmd_finetune_generate_fulltext_passages(args)
    elif args.ft_command == "eval-passages":
        cmd_finetune_eval_passages(args)
    elif args.ft_command == "status":
        cmd_finetune_status(args)
    elif args.ft_command == "validate":
        cmd_finetune_validate(args)
    else:
        print(
            "Usage: incite finetune {prepare|train|train-reranker|generate-passages|"
            "generate-fulltext-passages|eval-passages|status|validate}"
        )


def cmd_finetune_prepare(args):
    """Prepare training data for fine-tuning."""
    use_pipeline = any(
        [
            args.from_all,
            args.from_s2orc,
            args.from_synthetic,
            args.from_scicite,
        ]
    )

    if use_pipeline:
        from incite.finetuning.data_pipeline import DataPipeline, PipelineConfig
        from incite.finetuning.data_sources import (
            ExistingDataSource,
            S2ORCSource,
            SciCiteSource,
            SyntheticZoteroSource,
        )

        config = PipelineConfig(
            output_dir=Path(args.output_dir),
            dev_fraction=args.dev_fraction,
        )
        pipeline = DataPipeline(config)

        sources = []
        per_limit = args.limit

        if args.from_all or args.from_s2orc:
            s2orc_limit = per_limit if per_limit else 100_000
            sources.append((S2ORCSource(), s2orc_limit))

        if args.from_all or args.from_synthetic:
            sources.append((SyntheticZoteroSource(), per_limit))

        if args.from_all or args.from_scicite:
            s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
            sources.append((SciCiteSource(s2_api_key=s2_key), per_limit))

        if args.from_all:
            sources.append(
                (
                    ExistingDataSource(
                        test_set_path=Path(args.test_set),
                        corpus_path=Path(args.corpus),
                    ),
                    per_limit,
                )
            )

        if not sources:
            print("Error: No data sources selected.")
            sys.exit(1)

        pipeline.build(sources)

    elif args.from_unarxiv:
        from incite.finetuning.data_preparation import mine_training_data

        email = args.email or os.getenv("OPENALEX_EMAIL")
        mine_training_data(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            test_set_path=Path(args.test_set),
            openalex_email=email,
            target_source_papers=args.target,
            max_hard_negatives=args.max_negatives,
            dev_fraction=args.dev_fraction,
        )
    else:
        from incite.finetuning.data_preparation import prepare_from_existing

        prepare_from_existing(
            test_set_path=Path(args.test_set),
            corpus_path=Path(args.corpus),
            output_dir=Path(args.output_dir),
            max_hard_negatives=args.max_negatives,
            dev_fraction=args.dev_fraction,
        )

    print("\nNext step: incite finetune train")


def cmd_finetune_train(args):
    """Train fine-tuned model."""
    from incite.finetuning.train import train

    data_dir = Path(args.data_dir)

    # Resolve train/dev paths: explicit --train/--dev, or defaults in data_dir
    if args.train_file:
        train_path = data_dir / args.train_file
    else:
        train_path = data_dir / "train.jsonl"

    if args.dev_file:
        dev_path = data_dir / args.dev_file
    else:
        dev_path = data_dir / "dev.jsonl"

    if not train_path.exists():
        print(f"Error: {train_path} not found. Run 'incite finetune prepare' first.")
        sys.exit(1)
    if not dev_path.exists():
        print(f"Error: {dev_path} not found. Run 'incite finetune prepare' first.")
        sys.exit(1)

    general_data_path = Path(args.general_data) if args.general_data else None

    use_cached = None if not args.no_cached_mnrl else False

    # Parse matryoshka dims from comma-separated string
    matryoshka_dims = None
    if args.matryoshka_dims:
        matryoshka_dims = [int(d.strip()) for d in args.matryoshka_dims.split(",")]

    stats = train(
        train_path=train_path,
        dev_path=dev_path,
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        pretrained_model=args.pretrained_model,
        general_data_path=general_data_path,
        general_data_ratio=args.general_data_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_steps=args.eval_steps,
        mini_batch_size=args.mini_batch_size,
        use_cached_mnrl=use_cached,
        max_hard_negatives=args.max_hard_negatives,
        matryoshka_dims=matryoshka_dims,
        early_stopping_patience=args.early_stopping_patience,
    )

    if "error" not in stats:
        print("\nNext steps:")
        print("  1. incite evaluate --embedder minilm-ft --method hybrid --name 'MiniLM-FT v1'")
        print("  2. incite experiments compare <baseline_id> <finetuned_id>")


def cmd_finetune_train_reranker(args):
    """Train cross-encoder reranker on citation data."""
    from incite.finetuning.train_reranker import train_reranker

    data_dir = Path(args.data_dir)

    if args.train_file:
        train_path = data_dir / args.train_file
    else:
        train_path = data_dir / "master_train.jsonl"

    if args.dev_file:
        dev_path = data_dir / args.dev_file
    else:
        dev_path = data_dir / "master_dev.jsonl"

    if not train_path.exists():
        print(f"Error: {train_path} not found. Run 'incite finetune prepare' first.")
        sys.exit(1)
    if not dev_path.exists():
        print(f"Error: {dev_path} not found. Run 'incite finetune prepare' first.")
        sys.exit(1)

    corpus_path = Path(args.corpus) if args.corpus else None

    stats = train_reranker(
        train_path=train_path,
        dev_path=dev_path,
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        max_negatives=args.max_negatives,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        loss_type=args.loss,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.scheduler,
        early_stopping_patience=args.early_stopping_patience,
        corpus_path=corpus_path,
        use_full_text=args.use_full_text,
    )

    if "error" not in stats:
        print("\nNext steps:")
        print(
            "  1. incite evaluate --embedder minilm-ft --method hybrid "
            "--reranker citation-ft --name 'reranker v3'"
        )


def cmd_finetune_generate_passages(args):
    """Generate passage-level training data using LLM."""
    import json
    from collections import defaultdict

    from incite.corpus.loader import load_chunks, load_corpus
    from incite.evaluation.passage_metrics import save_passage_test_set
    from incite.finetuning.passage_generation import (
        contexts_to_training_examples,
        generate_passage_contexts_batch,
        select_passages,
        split_passage_data,
    )

    corpus = load_corpus(args.corpus)
    papers_by_id = {p.id: p for p in corpus}
    print(f"Loaded {len(corpus)} papers from corpus")

    chunks = load_chunks(args.chunks)
    chunks_by_paper: dict[str, list] = defaultdict(list)
    for c in chunks:
        chunks_by_paper[c.paper_id].append(c)
    print(f"Loaded {len(chunks)} chunks from {len(chunks_by_paper)} papers")

    papers = corpus
    if args.limit > 0:
        paper_ids_with_chunks = [pid for pid in chunks_by_paper if pid in papers_by_id]
        paper_ids_with_chunks = paper_ids_with_chunks[: args.limit]
        papers = [papers_by_id[pid] for pid in paper_ids_with_chunks]
        print(f"Limited to {len(papers)} papers")

    all_chunks_for_selection = []
    for p in papers:
        all_chunks_for_selection.extend(chunks_by_paper.get(p.id, []))

    selected = select_passages(all_chunks_for_selection, max_per_paper=args.max_per_paper)
    papers_with_passages = len({c.paper_id for c in selected})
    print("\nPassage selection:")
    print(f"  Papers with chunks: {len([p for p in papers if p.id in chunks_by_paper])}")
    print(f"  Passages selected:  {len(selected)}")
    print(f"  Papers represented: {papers_with_passages}")
    print(f"  Avg per paper:      {len(selected) / max(1, papers_with_passages):.1f}")

    if args.dry_run:
        print("\nDry run complete. Use without --dry-run to generate contexts.")
        return

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    contexts, stats = generate_passage_contexts_batch(
        papers=papers,
        chunks_by_paper=dict(chunks_by_paper),
        api_key=api_key,
        model=args.model,
        max_per_paper=args.max_per_paper,
    )

    print("\nGeneration stats:")
    print(f"  Passages processed: {stats['generated']}")
    print(f"  Contexts created:   {stats['contexts_created']}")
    print(f"  Failed:             {stats['failed']}")

    if not contexts:
        print("No contexts generated. Exiting.")
        return

    examples = contexts_to_training_examples(contexts, papers_by_id, dict(chunks_by_paper))
    print(f"  Training examples:  {len(examples)}")

    train, dev, eval_set = split_passage_data(
        examples,
        dev_fraction=args.dev_fraction,
        eval_fraction=args.eval_fraction,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "passage_train.jsonl"
    dev_path = output_dir / "passage_dev.jsonl"
    eval_path = output_dir / "passage_test_set.jsonl"

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex.to_dict()) + "\n")

    with open(dev_path, "w") as f:
        for ex in dev:
            f.write(json.dumps(ex.to_dict()) + "\n")

    save_passage_test_set(eval_set, eval_path)

    print("\nOutput files:")
    print(f"  Train: {train_path} ({len(train)} examples)")
    print(f"  Dev:   {dev_path} ({len(dev)} examples)")
    print(f"  Eval:  {eval_path} ({len(eval_set)} examples)")


def cmd_finetune_generate_fulltext_passages(args):
    """Generate passage-level training data from full-text corpora."""
    from incite.finetuning.fulltext_passages import generate_passage_data

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")

    stats = generate_passage_data(
        data_dir=Path(args.data_dir),
        target_papers=args.target_papers,
        max_per_paper=args.max_per_paper,
        output_dir=Path(args.output_dir),
        api_key=api_key,
        model=args.model,
        dev_fraction=args.dev_fraction,
        eval_fraction=args.eval_fraction,
        dry_run=args.dry_run,
        output_prefix=args.output_prefix,
    )

    if "error" in stats:
        print(f"\nError: {stats['error']}")
        sys.exit(1)


def cmd_finetune_eval_passages(args):
    """Evaluate passage-level retrieval."""
    from collections import defaultdict

    from incite.corpus.loader import load_chunks
    from incite.evaluation.passage_metrics import (
        evaluate_passage_retrieval,
        load_passage_test_set,
    )

    test_set_path = Path(args.test_set)
    if not test_set_path.exists():
        print(f"Error: {test_set_path} not found.")
        sys.exit(1)

    test_set = load_passage_test_set(test_set_path)
    print(f"Loaded {len(test_set)} passage test examples")

    chunks = load_chunks(args.chunks)
    chunks_by_paper: dict[str, list] = defaultdict(list)
    for c in chunks:
        chunks_by_paper[c.paper_id].append(c)
    print(f"Loaded {len(chunks)} chunks from {len(chunks_by_paper)} papers")

    embedder = None
    if args.embedder:
        from incite.retrieval.factory import EMBEDDERS

        embedder_cls = EMBEDDERS.get(args.embedder)
        if embedder_cls:
            print(f"Loading embedder: {args.embedder}...")
            embedder = embedder_cls()

    result = evaluate_passage_retrieval(embedder, test_set, chunks_by_paper)
    print(f"\n{result}")


def cmd_finetune_status(args):
    """Show training data and model status."""
    import json

    experiments_path = Path("data/experiments/experiments.jsonl")

    print("Training Data:")

    # Check known data files
    data_files = [
        ("data/finetuning/master_train.jsonl", "MASTER train (merged)"),
        ("data/finetuning/master_dev.jsonl", "MASTER dev (merged)"),
        ("data/finetuning/master_eval.jsonl", "MASTER eval (merged)"),
        ("data/finetuning/train.jsonl", "paper-level"),
        ("data/finetuning/dev.jsonl", "paper-level"),
        ("data/finetuning/passage_train.jsonl", "zotero passage"),
        ("data/finetuning/passage_dev.jsonl", "zotero passage"),
        ("data/finetuning/unarxiv_passage_train.jsonl", "unarxiv passage"),
        ("data/finetuning/unarxiv_passage_dev.jsonl", "unarxiv passage"),
        ("data/finetuning/s2orc_passage_train.jsonl", "s2orc passage"),
        ("data/finetuning/s2orc_passage_dev.jsonl", "s2orc passage"),
    ]

    for filepath, label in data_files:
        p = Path(filepath)
        if p.exists():
            count = sum(1 for line in open(p) if line.strip())
            print(f"  {filepath:<50} {count:>6,} examples  ({label})")
        else:
            print(f"  {filepath:<50}     -- not yet generated")

    # Check models
    print("\nModels:")
    models_dir = Path("models")
    if models_dir.exists():
        for model_dir in sorted(models_dir.iterdir()):
            if model_dir.is_dir():
                final_dir = model_dir / "final"
                if final_dir.exists():
                    print(f"  {model_dir}/final/    exists")
                else:
                    print(f"  {model_dir}/    (no final/ checkpoint)")
    else:
        print("  No models directory found.")

    # Show recent experiments
    print("\nRecent Experiments:")
    if experiments_path.exists():
        lines = []
        with open(experiments_path) as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line))
        # Show last 5
        for run in lines[-5:]:
            run_id = run.get("id", "?")
            config = run.get("config", {})
            results = run.get("results", {})
            model = config.get("model_name", "?")
            method = config.get("method", "?")
            r10 = results.get("recall@10", 0)
            ts = run.get("timestamp", "")[:10]
            print(f"  #{run_id:<4}  {model:<16} {method:<8} R@10={r10:.1%}  {ts}")
    else:
        print("  No experiments logged yet.")

    # Suggest next steps
    print("\nNext Steps:")
    step = 1
    has_fulltext = (
        Path("data/finetuning/unarxiv_passage_train.jsonl").exists()
        or Path("data/finetuning/s2orc_passage_train.jsonl").exists()
    )
    if not has_fulltext:
        print(
            f"  {step}. Generate fulltext passages: "
            "incite finetune generate-fulltext-passages --data-dir data/raw/unarxiv"
        )
        step += 1
    if not Path("data/finetuning/train.jsonl").exists():
        print(f"  {step}. Prepare paper-level data: incite finetune prepare")
        step += 1
    else:
        print(f"  {step}. Train: incite finetune train")
        step += 1


def cmd_finetune_validate(args):
    """Validate training data quality."""
    from incite.finetuning.validation import validate_dataset

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {data_dir}")
        return

    all_ok = True
    for filepath in jsonl_files:
        issues = validate_dataset(filepath)
        count = sum(1 for line in open(filepath) if line.strip())
        if issues:
            all_ok = False
            print(f"Validating {filepath}... ISSUES ({count:,} examples, {len(issues)} issues)")
            for issue in issues[:10]:
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print(f"Validating {filepath}... OK ({count:,} examples, 0 issues)")

    if all_ok:
        print("\nAll datasets passed validation.")
    else:
        print("\nSome datasets have issues. Review and fix before training.")
        sys.exit(1)
