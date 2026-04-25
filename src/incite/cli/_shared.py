"""Shared constants and helpers used across CLI subcommand modules.

EMBEDDER_CHOICES is the canonical list of available embedder keys. It is used
by serve.py, cloud.py, paperpile.py, and other subcommands to validate the
``--embedder`` argument and must stay in sync with the EMBEDDERS dict in
retrieval/factory.py.
"""

# Available embedder types (matches factory.py EMBEDDERS keys)
EMBEDDER_CHOICES = [
    "minilm",
    "e5",
    "specter",
    "nomic",
    "voyage",
    "minilm-ft",
    "minilm-ft-onnx",
    "modernbert",
    "scincl",
    "granite",
    "granite-ft",
]
