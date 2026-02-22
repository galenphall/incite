"""Command-line interface for inCite."""

import argparse
import sys

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


def main():
    """Main CLI entry point."""
    from incite.cli import (
        agent,
        cloud,
        core,
        data,
        doctor,
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
        agent,
        serve,
        cloud,
        paperpile,
        doctor,
    ]

    from incite import __version__

    parser = argparse.ArgumentParser(
        prog="incite",
        description="Local-first citation recommendation system",
    )
    parser.add_argument("--version", action="version", version=f"incite {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    for mod in modules:
        mod.register(subparsers)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)
