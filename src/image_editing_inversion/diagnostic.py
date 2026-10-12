"""Temporary command for a fresh Null-text experiment with detailed diagnostics."""

import argparse
from pathlib import Path
from typing import Sequence

from .cli import _execute, _positive_integer, _single_parameter_file


def experiment(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="experiment-diagnostic",
        description="Run a fresh Null-text experiment and record optimization and reconstruction diagnostics.",
    )
    parser.add_argument(
        "--method", action="append", choices=("null-text",),
        help="Only null-text is supported; omitted selects null-text. Repeated selections run once.",
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
        "--dataset", choices=("pie-bench", "magic-brush"),
        help="Filter the eval split by source dataset; omitted selects every source in eval.",
    )
    args = parser.parse_args(argv)

    def action() -> Path:
        from .workflows.null_text_diagnostic import DiagnosticWorkflowSession

        return DiagnosticWorkflowSession(
            args.h_params, dataset=args.dataset, nsamples=args.nsamples,
        ).run_methods(("null-text",))

    return _execute(action)


if __name__ == "__main__":
    raise SystemExit(experiment())
