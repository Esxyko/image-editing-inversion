"""Shared model runtime and dataset identity for inversion adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import ExperimentConfig

if TYPE_CHECKING:
    from ..editing import Editor


@dataclass(frozen=True, slots=True)
class InversionContext:
    """Shared model runtime and dataset identity passed to method adapters."""

    editor: Editor
    config: ExperimentConfig
    dataset_ref: str
    dataset_fingerprint: str
