"""Command-line interface for inCite."""

import argparse
import sys

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


def main():
    """Main CLI entry point."""
    from incite.cli import (
        acquire,
        agent,
        cloud,
        core,
        data,
        doctor,
        experiments,
        finetune,
        llm,
        paperpile,
        serve,
        setup,
    )

    modules = [
        setup,
        core,
        data,
        llm,
        finetune,
        experiments,
        agent,
        serve,
        acquire,
        cloud,
        paperpile,
        doctor,
    ]

    parser = argparse.ArgumentParser(
        prog="incite",
        description="Local-first citation recommendation system",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    for mod in modules:
        mod.register(subparsers)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)
