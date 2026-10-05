"""Conditional DDIM inversion with source-only latent residual correction.

Based on Ju et al.'s Direct Inversion, later published as PnP Inversion:
https://github.com/cure-lab/PnPInversion
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ...artifacts import InversionArtifact
from ...config import ExperimentConfig
from ..base import InversionMethod
from ..context import InversionContext
from ..validation import inverse_scheduler as _inverse_scheduler, validate_sampling
from ..hooks import DenoisingHook, DenoisingStepState
from .common import (
    GUIDANCE_KEY as _GUIDANCE_KEY, build_artifact, conditional_pivots, guidance_state,
    prepare_sources, require_finite, validate_ddim_scheduler,
    validate_guidance_compatibility, validate_guidance_state,
)

if TYPE_CHECKING:
    from diffusers import DDIMInverseScheduler


_OFFSETS_KEY = "direct_inversion_offsets"
_require_finite = partial(require_finite, label="Direct")
_validate_scheduler = partial(validate_ddim_scheduler, label="Direct")


def _artifact_state(artifact: InversionArtifact) -> tuple[torch.Tensor, float]:
    """Validate residuals without importing Diffusers or loading a model."""
    if artifact.method_id != "direct-inversion":
        raise ValueError("Direct inversion replay requires a direct-inversion artifact")
    if artifact.scheduler_id != "ddim" or artifact.eta != 0:
        raise ValueError("Direct inversion replay requires DDIM with artifact eta == 0")
    _validate_scheduler(artifact.scheduler_config)
    if set(artifact.per_step_state) != {_OFFSETS_KEY, _GUIDANCE_KEY}:
        raise ValueError(
            "Direct inversion artifacts require direct_inversion_offsets "
            "and guidance_scale state"
        )
    offsets = artifact.per_step_state[_OFFSETS_KEY]
    expected_shape = (len(artifact.timesteps), *artifact.terminal_latent.shape)
    if (
        not isinstance(offsets, torch.Tensor)
        or not offsets.is_floating_point()
        or offsets.ndim != 5
        or tuple(offsets.shape) != expected_shape
        or min(offsets.shape) < 1
    ):
        raise ValueError(
            "Direct inversion offsets must be floating tensors of shape "
            "[timesteps, 1, 4, latent_height, latent_width] matching the terminal latent"
        )
    _require_finite(artifact.terminal_latent, "terminal latent")
    _require_finite(offsets, "saved offsets")
    scale = validate_guidance_state(artifact, "Direct inversion")
    return offsets, scale


class _DirectInversionHook(DenoisingHook):
    """Correct only the source latent after each shared DDIM update."""

    def __init__(self, artifact: InversionArtifact) -> None:
        self._offsets, _ = _artifact_state(artifact)
        self._timesteps = artifact.timesteps

    def after_step(
        self, step_index: int, timestep: int, state: DenoisingStepState
    ) -> DenoisingStepState:
        if type(step_index) is not int or not 0 <= step_index < len(self._timesteps):
            raise ValueError("Direct inversion hook received an invalid step index")
        if type(timestep) is not int or timestep != self._timesteps[step_index]:
            raise ValueError("Direct inversion hook timestep does not match its artifact")
        offset = self._offsets[step_index]
        latents = state.latents
        expected_shape = (2, *offset.shape[1:])
        if tuple(latents.shape) != expected_shape:
            raise ValueError(
                f"Direct inversion hook expected latents of shape {expected_shape}, "
                f"got {tuple(latents.shape)}"
            )
        offset = offset.to(device=latents.device, dtype=latents.dtype)
        _require_finite(offset, "replay offset")
        source = latents[:1] + offset
        _require_finite(source, "corrected source latent")
        return replace(state, latents=torch.cat((source, latents[1:]), dim=0))


class DirectInversion(InversionMethod):
    """Invert source batches and record each image's reconstruction residuals."""

    @property
    def method_id(self) -> str:
        return "direct-inversion"

    def inversion_cache_parameters(self, context: InversionContext) -> Mapping[str, Any]:
        return {}

    def validate_inversion_config(self, config: ExperimentConfig) -> None:
        validate_sampling(config, "Direct")

    def validate_inversion_context(self, context: InversionContext) -> None:
        self._prepare_inverse_scheduler(context)

    def _prepare_inverse_scheduler(self, context: InversionContext) -> DDIMInverseScheduler:
        self._validate_context(context)
        return _inverse_scheduler(context, "Direct")

    @staticmethod
    def _validate_context(context: InversionContext) -> dict[str, Any]:
        config = context.config
        if config != context.editor.config:
            raise ValueError("Direct inversion context must use the editor's configuration")
        validate_sampling(config, "Direct")
        scheduler_config = context.editor.scheduler_config
        _validate_scheduler(scheduler_config)
        return scheduler_config

    def validate_replay(
        self, artifact: InversionArtifact, context: InversionContext
    ) -> None:
        _, guidance_scale = _artifact_state(artifact)
        artifact.validate_compatibility(
            context.config,
            dataset_ref=context.dataset_ref,
            dataset_fingerprint=context.dataset_fingerprint,
            timesteps=context.editor.expected_timesteps,
            scheduler_config=context.editor.scheduler_config,
        )
        validate_guidance_compatibility(guidance_scale, context, "Direct inversion")
        self._validate_context(context)

    def create_denoising_hook(self, artifact: InversionArtifact) -> DenoisingHook:
        return _DirectInversionHook(artifact)

    @staticmethod
    def _calculate_offsets(
        pivots: list[torch.Tensor],
        embeddings: torch.Tensor,
        context: InversionContext,
    ) -> torch.Tensor:
        editor = context.editor
        scale = context.config.sampling.guidance_scale
        latent = editor.to_device(pivots[-1], dtype=editor.dtype)
        offsets: list[torch.Tensor] = []
        for step_index, timestep in enumerate(editor.scheduler.timesteps):
            previous_pivot = editor.to_device(
                pivots[-step_index - 2], dtype=editor.dtype
            )
            model_input = torch.cat((latent, latent), dim=0)
            model_input = editor.scheduler.scale_model_input(model_input, timestep)
            prediction = editor.pipeline.unet(
                model_input, timestep, encoder_hidden_states=embeddings
            ).sample
            unconditional, conditional = prediction.chunk(2)
            guided = unconditional + scale * (conditional - unconditional)
            predicted_previous = editor.scheduler.step(
                guided, timestep, latent, eta=0
            ).prev_sample
            offset = previous_pivot - predicted_previous
            _require_finite(offset, "reconstruction offset")
            offsets.append(offset.detach().to(device="cpu").contiguous())
            # Advance from the corrected reconstruction, as replay will do.
            latent = predicted_previous + offset
            _require_finite(latent, "corrected trajectory latent")
        return torch.stack(offsets).contiguous()

    def invert(
        self, sample: Mapping[str, Any], context: InversionContext
    ) -> InversionArtifact:
        return self.invert_batch([sample], context)[0]

    @torch.inference_mode()
    def invert_batch(
        self, samples: Sequence[Mapping[str, Any]], context: InversionContext
    ) -> list[InversionArtifact]:
        """Build conditional pivots and record residuals at the run's guidance."""
        uids, prompts, images = self._validate_batch(samples, context)
        self.validate_inversion_config(context.config)
        inverse_scheduler = self._prepare_inverse_scheduler(context)
        editor = context.editor

        pipeline = editor.pipeline
        try:
            pipeline.maybe_free_model_hooks()
            sources = prepare_sources(uids, prompts, images, context, "Direct")
            embeddings = sources.embeddings
            pivots = conditional_pivots(sources, uids, inverse_scheduler, context, "Direct")
            offsets = self._calculate_offsets(pivots, embeddings, context)
            artifacts: list[InversionArtifact] = []
            for index, uid in enumerate(uids):
                artifact = build_artifact(self.method_id, context, uid, pivots[-1][index:index + 1], {
                    _OFFSETS_KEY: offsets[:, index:index + 1],
                    _GUIDANCE_KEY: guidance_state(context),
                })
                self.validate_replay(artifact, context)
                artifacts.append(artifact)
            return artifacts
        finally:
            pipeline.maybe_free_model_hooks()
