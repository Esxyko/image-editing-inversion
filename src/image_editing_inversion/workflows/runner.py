"""Reusable public workflow facade; execution state belongs to private sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..artifacts import ArtifactReference, ArtifactRepository
from ..artifacts.layout import method_directory_name
from ..inversion import discover_methods
from .session import WorkflowSession


def _resolve_method_ids(method_ids: Sequence[str]) -> tuple[str, ...]:
    """Expand selectors once without constructing adapter instances."""
    if isinstance(method_ids, (str, bytes)) or not method_ids:
        raise ValueError("At least one inversion method is required as a sequence of IDs.")
    discovered = discover_methods()
    available = set(discovered)
    resolved: dict[str, None] = {}
    for selector in method_ids:
        selected = discovered if selector == "ALL" else (selector,)
        for method_id in selected:
            method_directory_name(method_id)
            if method_id not in available:
                raise KeyError(f"Inversion method {method_id!r} is not registered; install its adapter first")
            resolved[method_id] = None
    if not resolved:
        raise ValueError("No inversion methods are registered.")
    return tuple(resolved)


class WorkflowRunner:
    """Create an independent invocation for every inversion or replay call."""

    def __init__(self, *, pipeline_h_params_file: str | Path | None = None) -> None:
        self._pipeline_h_params_file = pipeline_h_params_file

    def run_methods(self, method_ids: Sequence[str]) -> Path:
        resolved = _resolve_method_ids(method_ids)
        return WorkflowSession(self._pipeline_h_params_file).run_methods(resolved)

    def edit_artifacts(self, artifact_references: Sequence[ArtifactReference]) -> Path:
        if not artifact_references:
            raise ValueError("At least one saved inversion artifact is required.")
        discover_methods()
        return WorkflowSession(self._pipeline_h_params_file).edit_artifacts(artifact_references)


def edit_artifacts(artifact_id: str | None = None) -> Path:
    """Replay one raw UID or all published artifacts across every parameter file."""
    repository = ArtifactRepository()
    references = repository.discover() if artifact_id is None else repository.resolve_all(artifact_id)
    return WorkflowRunner().edit_artifacts(references)


def run_methods(
    method_ids: Sequence[str], *, pipeline_h_params_file: str | Path | None = None,
) -> Path:
    """Invert and edit once per selected method; ALL expands discovered IDs."""
    return WorkflowRunner(pipeline_h_params_file=pipeline_h_params_file).run_methods(method_ids)
