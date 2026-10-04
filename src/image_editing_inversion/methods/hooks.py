"""Method-specific contracts around shared denoising steps."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DenoisingStepState:
    """One source/target pair in source-first order during a denoising step.

    ``variance_noise`` is present in a before-step call when DDIM eta is
    positive. A before-step hook may replace it to control that stochastic
    DDIM update; after-step calls receive ``None``.
    """

    latents: torch.Tensor
    prompt_embeddings: torch.Tensor
    negative_prompt_embeddings: torch.Tensor
    variance_noise: torch.Tensor | None = None


class DenoisingHook:
    """Optional method-specific changes around each shared denoising step."""

    def before_step(
        self, step_index: int, timestep: int, state: DenoisingStepState
    ) -> DenoisingStepState:
        return state

    def after_step(
        self, step_index: int, timestep: int, state: DenoisingStepState
    ) -> DenoisingStepState:
        return state
