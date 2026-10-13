"""Reuse captured source embeddings without changing denoising trajectories."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from ...artifacts import ArtifactCompatibilityError, InversionArtifact
from ..context import InversionContext
from .common import require_finite


SOURCE_CONDITIONAL_KEY = "source_prompt_embeddings"
SOURCE_UNCONDITIONAL_KEY = "source_negative_prompt_embeddings"
SOURCE_EMBEDDING_VERSION = 1


def reject_trajectory_overwrite(artifact: InversionArtifact, label: str) -> None:
    if {"source_latents", "direct_source_latents"} & artifact.per_step_state.keys():
        raise ArtifactCompatibilityError(
            f"{label} artifact uses withdrawn source-trajectory overwrite state; "
            "regenerate with experiment or diffuse"
        )


def source_embedding_state(embeddings: torch.Tensor) -> list[Mapping[str, torch.Tensor]]:
    unconditional, conditional = embeddings.detach().to(device="cpu").chunk(2)
    return [{SOURCE_CONDITIONAL_KEY: conditional[index:index + 1],
             SOURCE_UNCONDITIONAL_KEY: unconditional[index:index + 1]}
            for index in range(len(conditional))]


def validate_source_embeddings(artifact: InversionArtifact, label: str,
                               context: InversionContext | None = None) -> None:
    """Legacy artifacts may lack encodings; partial or incompatible state fails."""
    if not artifact.static_state:
        return
    if set(artifact.static_state) != {SOURCE_CONDITIONAL_KEY, SOURCE_UNCONDITIONAL_KEY}:
        raise ValueError(f"{label} artifacts require both original source embedding tensors")
    conditional = artifact.static_state[SOURCE_CONDITIONAL_KEY]
    unconditional = artifact.static_state[SOURCE_UNCONDITIONAL_KEY]
    if (not isinstance(conditional, torch.Tensor) or not isinstance(unconditional, torch.Tensor)
            or conditional.ndim != 3 or conditional.shape[0] != 1 or min(conditional.shape) < 1
            or not conditional.is_floating_point() or conditional.shape != unconditional.shape
            or conditional.dtype != artifact.terminal_latent.dtype
            or unconditional.dtype != conditional.dtype):
        raise ValueError(f"{label} source embeddings must share floating [1, tokens, hidden] shape and latent dtype")
    require_finite(conditional, "conditional embedding", label)
    require_finite(unconditional, "unconditional embedding", label)
    if context is not None:
        shape = (1, context.editor.pipeline.tokenizer.model_max_length,
                 context.editor.pipeline.text_encoder.config.hidden_size)
        if tuple(conditional.shape) != shape:
            raise ValueError(f"{label} source embedding dimensions do not match the active text encoder")
        if conditional.dtype != context.editor.dtype:
            raise ArtifactCompatibilityError(f"{label} source embedding dtype does not match this run")


def replay_source_embeddings(artifacts: Sequence[InversionArtifact], label: str,
                             context: InversionContext | None) -> tuple[torch.Tensor, torch.Tensor] | None:
    for artifact in artifacts:
        reject_trajectory_overwrite(artifact, label)
        validate_source_embeddings(artifact, label, context)
    present = [bool(artifact.static_state) for artifact in artifacts]
    if not any(present):
        return None
    if not all(present):
        raise ArtifactCompatibilityError(
            f"{label} replay cannot mix artifacts with and without original embeddings; regenerate this batch"
        )
    return (torch.cat([artifact.static_state[SOURCE_CONDITIONAL_KEY] for artifact in artifacts]),
            torch.cat([artifact.static_state[SOURCE_UNCONDITIONAL_KEY] for artifact in artifacts]))
