#!/usr/bin/env python3
"""Export fine-tuned MiniLM model to ONNX format for lightweight CPU inference.

Usage:
    python scripts/export_minilm_onnx.py [--input PATH] [--output PATH]

Requires: pip install optimum[onnxruntime] sentence-transformers
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Export MiniLM model to ONNX")
    parser.add_argument(
        "--input",
        default="models/minilm-citation-v4/final",
        help="Path to PyTorch model directory",
    )
    parser.add_argument(
        "--output",
        default="src/incite/models/minilm-onnx",
        help="Path to write ONNX model (inside package for bundling)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"PyTorch model not found at {input_path}")

    # Export to ONNX via optimum
    print(f"Exporting {input_path} -> {output_path}")

    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(input_path))
    ort_model = ORTModelForFeatureExtraction.from_pretrained(str(input_path), export=True)

    output_path.mkdir(parents=True, exist_ok=True)
    ort_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"ONNX model saved to {output_path}")

    # Sanity check: compare PyTorch vs ONNX embeddings
    print("\nRunning sanity check...")
    from sentence_transformers import SentenceTransformer

    pt_model = SentenceTransformer(str(input_path))
    test_texts = [
        "social network effects on political polarization",
        "This paper studies how online platforms contribute to ideological sorting.",
    ]

    # PyTorch embeddings (mean pooling + normalize, default for MiniLM)
    pt_embeddings = pt_model.encode(test_texts, normalize_embeddings=True)

    # ONNX embeddings (mean pooling — matches OnnxMiniLMEmbedder._mean_pooling)
    onnx_tokenizer = AutoTokenizer.from_pretrained(str(output_path))
    onnx_model = ORTModelForFeatureExtraction.from_pretrained(str(output_path))

    encoded = onnx_tokenizer(
        test_texts, padding=True, truncation=True, max_length=512, return_tensors="np"
    )
    outputs = onnx_model(**encoded)
    token_embs = outputs[0]
    if hasattr(token_embs, "numpy"):
        token_embs = token_embs.numpy()

    # Mean pooling with attention mask
    attention_mask = encoded["attention_mask"]
    if hasattr(attention_mask, "numpy"):
        attention_mask = attention_mask.numpy()
    mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    mask_expanded = np.broadcast_to(mask_expanded, token_embs.shape)
    sum_embeddings = np.sum(token_embs * mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
    onnx_embeddings = sum_embeddings / sum_mask

    # L2 normalize
    norms = np.linalg.norm(onnx_embeddings, axis=1, keepdims=True)
    onnx_embeddings = onnx_embeddings / np.where(norms == 0, 1, norms)

    # Compare
    for i, text in enumerate(test_texts):
        cos_sim = np.dot(pt_embeddings[i], onnx_embeddings[i])
        label = text[:60]
        status = "OK" if cos_sim > 0.99 else "WARN"
        print(f"  [{status}] '{label}...' cosine={cos_sim:.6f}")

    avg_sim = np.mean(
        [np.dot(pt_embeddings[i], onnx_embeddings[i]) for i in range(len(test_texts))]
    )
    print(f"\n  Average cosine similarity: {avg_sim:.6f}")
    if avg_sim > 0.99:
        print("  PASS: ONNX model matches PyTorch output")
    else:
        print("  WARNING: Cosine similarity below 0.99 — investigate before deploying")


if __name__ == "__main__":
    main()
