"""Unsupervised domain pretraining for citation retrieval.

Uses TSDAE (Transformer-based Sequential Denoising Auto-Encoder) to adapt
the base model to scientific text before supervised contrastive training.
This follows the OpenScholar finding that domain pretraining before supervised
training improves downstream retrieval (Asai et al., 2026).

Usage:
    python -m incite.finetuning.pretrain
    python -m incite.finetuning.pretrain --method simcse --epochs 2
"""

import json
import random
from pathlib import Path
from typing import Optional  # noqa: UP035


def _load_corpus_texts(
    corpus_path: Path = Path("data/processed/corpus.jsonl"),
    chunk_cache_dir: Optional[Path] = None,
    max_texts: int = 100_000,
    min_length: int = 50,
) -> list[str]:
    """Load text passages from corpus abstracts and cached chunk paragraphs.

    Args:
        corpus_path: Path to corpus.jsonl
        chunk_cache_dir: Path to ~/.incite/ for loading cached chunks
        max_texts: Maximum number of passages to use
        min_length: Minimum character length for a passage

    Returns:
        Deduplicated list of text passages
    """
    texts = set()

    # Load abstracts from corpus
    if corpus_path.exists():
        print(f"Loading abstracts from {corpus_path}...")
        with open(corpus_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    abstract = data.get("abstract", "")
                    if len(abstract) >= min_length:
                        texts.add(abstract)
        print(f"  Loaded {len(texts)} abstracts")

    # Load chunk paragraphs from cache
    if chunk_cache_dir is None:
        chunk_cache_dir = Path.home() / ".incite"

    for pattern in ["zotero_chunks_paragraph.jsonl", "zotero_chunks_grobid.jsonl"]:
        chunks_path = chunk_cache_dir / pattern
        if chunks_path.exists():
            print(f"Loading chunks from {chunks_path}...")
            count = 0
            with open(chunks_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        text = data.get("text", "")
                        if len(text) >= min_length:
                            texts.add(text)
                            count += 1
            print(f"  Loaded {count} chunk paragraphs")

    # Deduplicate and sample
    texts_list = list(texts)
    if len(texts_list) > max_texts:
        random.shuffle(texts_list)
        texts_list = texts_list[:max_texts]

    print(f"Total unique passages: {len(texts_list)}")
    return texts_list


def pretrain_domain(
    corpus_texts: list[str] | None = None,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: Path = Path("models/minilm-pretrained"),
    epochs: int = 1,
    batch_size: int = 64,
    method: str = "tsdae",
    seed: int = 42,
) -> str:
    """Unsupervised domain pretraining using TSDAE or SimCSE.

    TSDAE (Wang et al., 2021) corrupts input sentences by deleting tokens
    and trains the model to reconstruct them. This teaches domain-specific
    vocabulary and writing style without labeled data.

    SimCSE (Gao et al., 2021) uses dropout as noise for contrastive learning
    on the same sentence. Simpler but less effective for domain adaptation.

    Args:
        corpus_texts: List of text passages. If None, loads from default paths.
        base_model: HuggingFace model name for base
        output_dir: Directory to save pretrained model
        epochs: Number of pretraining epochs
        batch_size: Training batch size
        method: "tsdae" or "simcse"
        seed: Random seed

    Returns:
        Path to saved pretrained model directory
    """
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import DenoisingAutoEncoderLoss
    from sentence_transformers.trainer import SentenceTransformerTrainer

    from incite.utils import get_best_device

    # Load corpus texts if not provided
    if corpus_texts is None:
        corpus_texts = _load_corpus_texts()

    if not corpus_texts:
        raise ValueError("No corpus texts found for pretraining")

    print(f"\nDomain pretraining with {method.upper()}")
    print(f"  Passages: {len(corpus_texts)}")
    print(f"  Base model: {base_model}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")

    # Load model
    device = get_best_device()
    print(f"  Device: {device}")
    model = SentenceTransformer(base_model, device=device)

    # Create dataset
    if method == "tsdae":
        # TSDAE: sentence-transformers v5+ expects both "anchor" (corrupted) and
        # "positive" (original) columns. The loss reconstructs originals from corrupted.
        dataset = Dataset.from_dict(
            {
                "anchor": corpus_texts,
                "positive": corpus_texts,  # Same text — the loss handles corruption of anchor
            }
        )
        loss = DenoisingAutoEncoderLoss(
            model=model,
            decoder_name_or_path=base_model,
            # True breaks with transformers>=5 (missing _tie_encoder_decoder_weights)
            tie_encoder_decoder=False,
        )
    elif method == "simcse":
        # SimCSE: each example needs "anchor" and "positive" (same text, different dropout)
        dataset = Dataset.from_dict(
            {
                "anchor": corpus_texts,
                "positive": corpus_texts,  # Same text — dropout creates the contrast
            }
        )
        from sentence_transformers.losses import MultipleNegativesRankingLoss

        loss = MultipleNegativesRankingLoss(model=model)
    else:
        raise ValueError(f"Unknown pretraining method: {method}. Use 'tsdae' or 'simcse'.")

    # Output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine MPS compatibility
    is_mps = device == "mps"

    # Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=(device == "cuda"),
        bf16=False,
        seed=seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=not is_mps,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )

    # Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
    )

    print("\nStarting pretraining...")
    trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    model.save(str(final_dir))
    print(f"\nPretrained model saved to {final_dir}")

    # Save pretraining metadata
    meta = {
        "method": method,
        "base_model": base_model,
        "corpus_size": len(corpus_texts),
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
    }
    meta_path = output_dir / "pretrain_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return str(final_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Domain pretraining for citation retrieval")
    parser.add_argument(
        "--base-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model to pretrain",
    )
    parser.add_argument("--output", default="models/minilm-pretrained", help="Output directory")
    parser.add_argument(
        "--method", choices=["tsdae", "simcse"], default="tsdae", help="Pretraining method"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    args = parser.parse_args()
    pretrain_domain(
        base_model=args.base_model,
        output_dir=Path(args.output),
        method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
