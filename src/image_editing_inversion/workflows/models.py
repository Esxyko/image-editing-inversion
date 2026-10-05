"""Internal records exchanged by workflow collaborators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..artifacts import ArtifactReference, InversionArtifact
from ..generation import PromptToPrompt
from ..inversion import DenoisingHook, InversionMethod


@dataclass(frozen=True, slots=True)
class InversionRecord:
    uid: str
    inversion_seconds: float
    inversion_batch_seconds: float
    inversion_batch_size: int
    artifact_reused: bool
    batch_id: str


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    artifact: InversionArtifact
    reference: ArtifactReference
    sample: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LoadResult:
    reference: ArtifactReference
    loaded: LoadedArtifact | None = None
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class PreparedEdit:
    loaded: LoadedArtifact
    method: InversionMethod
    policy_class: type[PromptToPrompt]
    settings: Mapping[str, Any]
    hook: DenoisingHook | None


@dataclass(frozen=True, slots=True)
class WorkItem:
    prepared: PreparedEdit
    inversion: InversionRecord | None = None

    def result_values(self) -> dict[str, Any]:
        return reference_values(self.prepared.loaded.reference, self.inversion)


def reference_values(
    reference: ArtifactReference, inversion: InversionRecord | None = None,
) -> dict[str, Any]:
    return {
        "uid": reference.sample_uid,
        "method_id": reference.group.method_id,
        "artifact": str(reference.path),
        "artifact_index": reference.index,
        "artifact_reused": True if inversion is None else inversion.artifact_reused,
        "inversion_batch_id": None if inversion is None else inversion.batch_id,
        "inversion_seconds": None if inversion is None else inversion.inversion_seconds,
        "inversion_batch_seconds": None if inversion is None else inversion.inversion_batch_seconds,
        "inversion_batch_size": None if inversion is None else inversion.inversion_batch_size,
    }
