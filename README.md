# inCite

**Write text. Get relevant papers from your library.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/incite.svg)](https://pypi.org/project/incite/)

## Overview

inCite is a local-first citation recommendation system. It indexes your Zotero library or a folder of PDFs and suggests relevant papers as you write. Everything runs on your machine -- no cloud account, no API keys, no data leaving your laptop.

- **Local-first**: Your papers and writing stay on your machine
- **Works with what you have**: Zotero library, a folder of PDFs, or a JSONL corpus
- **Editor plugins**: Obsidian, VS Code, Google Docs, and Microsoft Word
- **Fine-tuned models**: Citation-specific sentence transformers trained on 64K academic citation contexts

## Quick Start

```bash
pip install incite
incite setup
```

The setup wizard auto-detects your Zotero library (or accepts a folder of PDFs), builds a search index, and verifies everything works.

## Usage

### Command Line

```bash
# Get recommendations for a passage
incite recommend "The relationship between CO2 emissions and global temperature..." -k 10

# Start the API server (for editor plugins)
incite serve --embedder minilm-ft

# Start the menu bar app (macOS, manages the server for you)
pip install incite[tray]
incite tray
```

### Python API

```python
from incite.agent import InCiteAgent

# From Zotero library
agent = InCiteAgent.from_zotero(embedder_type="minilm-ft")

# From a folder of PDFs
agent = InCiteAgent.from_folder("~/Papers")

# Get recommendations
response = agent.recommend("climate change and agricultural productivity", k=10)
for rec in response.recommendations:
    print(f"  {rec.rank}. [{rec.score:.2f}] {rec.title} ({rec.year})")
```

### REST API

```bash
incite serve --embedder minilm-ft
# API docs at http://localhost:8230/docs

curl -X POST http://localhost:8230/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "climate change impacts on crop yields", "k": 5}'
```

## Editor Plugins

inCite integrates with your writing environment via editor plugins that connect to the local API server.

| Editor | Status | Install |
|--------|--------|---------|
| **Obsidian** | Stable | Build from `editor-plugins/obsidian-incite/` |
| **VS Code** | Stable | Build from `editor-plugins/vscode-incite/` |
| **Google Docs** | Stable | Apps Script add-on via `clasp push` |
| **Microsoft Word** | Beta | Office.js add-in, sideload `manifest.xml` |

All plugins share the `@incite/shared` TypeScript package for API communication and context extraction.

## Paper Sources

- **Zotero** (recommended): Auto-detects your local Zotero library and reads directly from the SQLite database
- **PDF folder**: Point at any directory of PDFs -- metadata is extracted automatically
- **JSONL corpus**: Load a pre-built corpus file with title, abstract, authors, and other metadata

## How It Works

1. **Embed papers**: Each paper is embedded as `title. authors. year. journal. abstract` using a sentence transformer
2. **Embed your writing**: Your text is embedded with the same model
3. **Search**: FAISS finds the nearest papers by cosine similarity
4. **Fuse** (optional): BM25 keyword matching is combined with neural results via Reciprocal Rank Fusion for improved recall
5. **Evidence**: The best matching paragraph from each paper's full text is attached as supporting evidence

## Embedder Models

| Model | Key | Dims | Notes |
|-------|-----|------|-------|
| MiniLM fine-tuned v4 | `minilm-ft` | 384 | Default. Citation-specific, auto-downloads from HuggingFace |
| MiniLM | `minilm` | 384 | Fast, good baseline |
| SPECTER2 | `specter` | 768 | Scientific domain |
| Nomic v1.5 | `nomic` | 768 | Long context (8K tokens) |
| Granite | `granite` | 384 | IBM Granite, 8K context |

> For even better results (MRR 0.550 vs 0.428), try the cloud service at [inciteref.com](https://inciteref.com) which uses our best fine-tuned model.

## Fine-Tuning

You can fine-tune your own citation embedder on your training data:

```bash
pip install incite[finetune]
incite finetune train --train data.jsonl --dev dev.jsonl
```

The training pipeline uses Matryoshka representation learning with cached multiple negatives ranking loss, supporting hard negatives for best results.

## Development

```bash
git clone https://github.com/galenphall/incite.git
pip install -e ".[dev]"
pytest
ruff check src/incite && ruff format src/incite
```

## Optional Dependencies

inCite's core is Apache 2.0 licensed. Some optional features depend on copyleft-licensed libraries and are packaged as extras to keep the default installation permissive.

```bash
pip install incite[pdf]       # PyMuPDF for PDF text extraction (AGPL)
pip install incite[zotero]    # pyzotero for Zotero integration (GPL)
pip install incite[api]       # FastAPI server
pip install incite[webapp]    # Streamlit UI
pip install incite[finetune]  # Training pipeline
pip install incite[tray]      # macOS menu bar app
pip install incite[all]       # Everything
```

> **Note**: The `pdf` and `zotero` extras pull in AGPL and GPL dependencies respectively. If license compatibility matters for your use case, install only the extras you need.

## Cloud Service

[inciteref.com](https://inciteref.com) offers a hosted version of inCite with additional features:

- **Better model**: Granite-FT fine-tuned embedder (MRR 0.550 vs 0.428 for the default local model)
- **Cloud PDF processing**: Full-text extraction without running GROBID locally
- **Reference manager**: Collections, tags, notes, and citation export (BibTeX/RIS)
- **Multi-device sync**: Access your library from anywhere

The local CLI and cloud service are complementary -- use whichever fits your workflow.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[Apache License 2.0](LICENSE)

## Citation

```bibtex
@software{incite2025,
  author       = {Hall, Galen},
  title        = {inCite: Local-First Citation Recommendation},
  year         = {2025},
  url          = {https://github.com/galenphall/incite},
  license      = {Apache-2.0}
}
```
