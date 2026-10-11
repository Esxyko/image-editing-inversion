"""Command-line entry points for inversion workflows."""

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence


def build_diffuse_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diffuse", description="Invert, reconstruct, and edit project dataset records."
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
    parser.add_argument(
        "--dataset",
        choices=("pie-bench", "magic-brush"),
        help="Only process records from this source dataset; omitted processes every record.",
    )
    return parser


def _single_parameter_file(value: str) -> str:
    if value == "ALL":
        raise argparse.ArgumentTypeError("experiment requires one YAML filename; ALL is not allowed")
    return value


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("nsamples must be a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("nsamples must be a positive integer")
    return result


def build_experiment_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment",
        description="Compare every inversion method on the first selected project dataset records.",
    )
    parser.add_argument(
        "--h-params", required=True, type=_single_parameter_file, metavar="FILENAME",
        help="One YAML filename inside pipeline_h_params/; ALL is not allowed.",
    )
    parser.add_argument(
        "--nsamples", required=True, type=_positive_integer, metavar="N",
        help="Process the first N records in dataset order after source filtering; N must be positive.",
    )
    parser.add_argument(
        "--dataset",
        choices=("pie-bench", "magic-brush"),
        help="Only select records from this source dataset; omitted selects every source.",
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
        return run_methods(
            args.method, pipeline_h_params_file=parameter_file, dataset=args.dataset,
        )

    return _execute(action)


def experiment(argv: Sequence[str] | None = None) -> int:
    args = build_experiment_parser().parse_args(argv)

    def action() -> Path:
        from .workflows import run_methods

        return run_methods(
            ["ALL"], pipeline_h_params_file=args.h_params,
            dataset=args.dataset, nsamples=args.nsamples,
        )

    return _execute(action)
