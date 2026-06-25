"""Command-line entry point for the framework.

A thin dispatcher over the experiment runner and the analysis pipeline. The
subcommands are placeholders until the corresponding layers are implemented.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="aos-ga", description=__doc__)
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("run", help="execute configured experiment runs")
    subparsers.add_parser("analyze", help="aggregate results and run the analysis pipeline")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from aos_ga import __version__

        print(__version__)
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    parser.error(f"command {args.command!r} is not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
