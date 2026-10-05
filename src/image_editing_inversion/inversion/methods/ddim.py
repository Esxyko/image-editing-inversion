"""Deterministic DDIM inversion using the shared editing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ...artifacts import InversionArtifact
from ...config import ExperimentConfig
from ..base import InversionMethod
from ..context import InversionContext
from ..validation import inverse_scheduler as _inverse_scheduler, validate_sampling
from .common import build_artifact, prepare_sources


if TYPE_CHECKING:
    from diffusers import DDIMInverseScheduler


class DDIMInversion(InversionMethod):
    """Invert source batches with the run's DDIM schedule and guidance."""

    @property
    def method_id(self) -> str:
        return "ddim"

    def inversion_cache_parameters(self, context: InversionContext) -> Mapping[str, Any]:
        return {}

    def validate_inversion_config(self, config: ExperimentConfig) -> None:
        validate_sampling(config, "DDIM")

    def validate_inversion_context(self, context: InversionContext) -> None:
        self._prepare_inverse_scheduler(context)

    def _prepare_inverse_scheduler(self, context: InversionContext) -> DDIMInverseScheduler:
        validate_sampling(context.config, "DDIM")
        if context.editor.scheduler_config.get("thresholding", False):
            raise ValueError("DDIM inversion does not support dynamic thresholding")
        return _inverse_scheduler(context, "DDIM")

    def invert(
        self, sample: Mapping[str, Any], context: InversionContext
    ) -> InversionArtifact:
        return self.invert_batch([sample], context)[0]

    @torch.inference_mode()
    def invert_batch(
        self, samples: Sequence[Mapping[str, Any]], context: InversionContext
    ) -> list[InversionArtifact]:
        """Encode sources together, then follow DDIM's ascending noise schedule."""
        uids, prompts, images = self._validate_batch(samples, context)

        self.validate_inversion_config(context.config)
        inverse_scheduler = self._prepare_inverse_scheduler(context)
        editor = context.editor
        config = context.config

        pipeline = editor.pipeline
        try:
            # Manual model calls need the same offload cleanup as pipeline calls.
            pipeline.maybe_free_model_hooks()
            sources = prepare_sources(uids, prompts, images, context, "DDIM")
            latents, embeddings = sources.latent, sources.embeddings
            for timestep in inverse_scheduler.timesteps:
                model_input = torch.cat((latents, latents), dim=0)
                model_input = inverse_scheduler.scale_model_input(model_input, timestep)
                prediction = pipeline.unet(
                    model_input, timestep, encoder_hidden_states=embeddings
                ).sample
                unconditional, conditional = prediction.chunk(2)
                guided = unconditional + config.sampling.guidance_scale * (
                    conditional - unconditional
                )
                latents = inverse_scheduler.step(
                    guided, timestep, latents
                ).prev_sample

            if not torch.isfinite(latents).all().item():
                raise ValueError("DDIM inversion produced a non-finite terminal latent")
            terminal = latents.detach().to(device="cpu")
            artifacts: list[InversionArtifact] = []
            for index, uid in enumerate(uids):
                artifact = build_artifact(self.method_id, context, uid, terminal[index:index + 1])
                editor.validate_artifact(artifact)
                artifacts.append(artifact)
            return artifacts
        finally:
            pipeline.maybe_free_model_hooks()
