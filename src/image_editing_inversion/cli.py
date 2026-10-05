"""Command-line entry point for inversion workflows."""

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence


def build_diffuse_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diffuse", description="Invert, reconstruct, and edit the entire project dataset."
    )
    parser.add_argument(
        "--method", action="append", required=True, metavar="METHOD|ALL",
        help="Registered method ID, or ALL for every discovered method; repeatable, duplicates run once."
    )
    parser.add_argument(
        "--h-params",
        required=True,
        metavar="FILENAME|ALL",
        help="YAML filename inside pipeline_h_params/, or ALL to run every file sequentially.",
    )
    return parser


def _execute(action: Callable[[], Path]) -> int:
    try:
        run_dir = action()
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(run_dir)
    return 0


def diffuse(argv: Sequence[str] | None = None) -> int:
    args = build_diffuse_parser().parse_args(argv)

    def action() -> Path:
        from .workflows import run_methods

        parameter_file = None if args.h_params == "ALL" else args.h_params
        return run_methods(args.method, pipeline_h_params_file=parameter_file)

    return _execute(action)
