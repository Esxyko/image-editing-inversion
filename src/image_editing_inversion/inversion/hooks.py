"""Method-specific contracts around shared denoising steps."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DenoisingStepState:
    """Source/target tensors in source-first order during a denoising step.

    Pair hooks receive two branches. Batched hooks receive all N source
    branches followed by all N target branches.

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


class BatchedDenoisingHook(DenoisingHook):
    """Apply one method's hook to a complete source-first editing batch.

    ``sample_uids`` identifies the exact artifact order handled by this hook.
    The editor validates that order before any model calls. Implementations
    can return unchanged fields directly instead of copying each image pair.
    """

    def __init__(self, sample_uids: tuple[str, ...]) -> None:
        if (not sample_uids or any(not isinstance(uid, str) or not uid.strip() for uid in sample_uids)
                or len(set(sample_uids)) != len(sample_uids)):
            raise ValueError("A batched denoising hook requires unique sample UIDs")
        self.sample_uids = tuple(sample_uids)

    @property
    def batch_size(self) -> int:
        return len(self.sample_uids)
