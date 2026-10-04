"""Command-line entry points for inversion experiments."""

import argparse
import sys
from pathlib import Path
from typing import Sequence


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Local Hugging Face Dataset directory; otherwise use the project Hub loader.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--pipeline-h-params",
        type=Path,
        help="YAML filename inside pipeline_h_params/; otherwise run every file sequentially.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image-editing-inversion")
    commands = parser.add_subparsers(dest="command", required=True)

    edit = commands.add_parser(
        "edit-artifact", help="Edit source images from saved inversion artifacts."
    )
    _add_common_arguments(edit)
    edit.add_argument(
        "--artifact",
        type=Path,
        action="append",
        required=True,
        help="Artifact directory; repeat to process multiple samples.",
    )

    run = commands.add_parser(
        "run", help="Invert and edit selected samples with registered methods."
    )
    _add_common_arguments(run)
    run.add_argument(
        "--method", action="append", required=True, help="Registered method ID; repeatable."
    )
    run.add_argument(
        "--uid", action="append", required=True, help="Dataset UID; repeatable."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from .experiments import edit_artifacts, run_methods

        if args.command == "edit-artifact":
            run_dir = edit_artifacts(
                args.dataset_path, args.artifact, args.output,
                pipeline_h_params_file=args.pipeline_h_params,
            )
        else:
            run_dir = run_methods(
                args.dataset_path, args.method, args.uid, args.output,
                pipeline_h_params_file=args.pipeline_h_params,
            )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(run_dir)
    return 0
