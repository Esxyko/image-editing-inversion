"""Deterministic DDIM inversion using the shared editing pipeline."""

from __future__ import annotations

from typing import Any, Mapping

from PIL import Image
import torch

from ...artifacts import InversionArtifact
from ..base import InversionMethod
from ..context import InversionContext


class DDIMInversion(InversionMethod):
    """Invert one source image with the run's DDIM schedule and guidance."""

    @property
    def method_id(self) -> str:
        return "ddim"

    def inversion_cache_parameters(self, context: InversionContext) -> Mapping[str, Any]:
        return {}

    @torch.inference_mode()
    def invert(
        self, sample: Mapping[str, Any], context: InversionContext
    ) -> InversionArtifact:
        """Encode the source image, then follow DDIM's ascending noise schedule."""
        if not isinstance(sample, Mapping):
            raise TypeError("DDIM inversion sample must be a mapping")
        uid = sample.get("uid")
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("Dataset uid must be a nonempty string")
        prompt = sample.get("source_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Dataset source_prompt must be a nonempty string")
        image = sample.get("source_img")
        if not isinstance(image, Image.Image):
            raise TypeError("Dataset source_img must be a Pillow image")

        editor = context.editor
        config = context.config
        if config != editor.config:
            raise ValueError("DDIM inversion context must use the editor's configuration")
        if config.sampling.eta != 0:
            raise ValueError("DDIM inversion requires sampling.eta == 0")
        scheduler_config = editor.scheduler_config
        if scheduler_config.get("thresholding", False):
            raise ValueError("DDIM inversion does not support dynamic thresholding")

        # Discovery can load this adapter without importing Diffusers or a runtime.
        from diffusers import DDIMInverseScheduler

        try:
            inverse_scheduler = DDIMInverseScheduler.from_config(scheduler_config)
            inverse_scheduler.set_timesteps(
                config.sampling.num_inference_steps, device=editor.device
            )
        except (ValueError, NotImplementedError) as exc:
            raise ValueError(
                f"DDIM inversion cannot use the editor's scheduler settings: {exc}"
            ) from exc
        timesteps = editor.expected_timesteps
        inverse_timesteps = tuple(
            int(step) for step in inverse_scheduler.timesteps.tolist()
        )
        if inverse_timesteps != tuple(reversed(timesteps)):
            raise ValueError(
                "DDIM inversion timesteps must match the reverse of the editor's schedule"
            )

        pipeline = editor.pipeline
        try:
            # Manual model calls need the same offload cleanup as pipeline calls.
            pipeline.maybe_free_model_hooks()
            pixels = pipeline.image_processor.preprocess(
                image.convert("RGB"), height=config.model.height, width=config.model.width
            ).to(device=editor.device, dtype=editor.dtype)
            latents = pipeline.vae.encode(pixels).latent_dist.mode()
            latents = latents * pipeline.vae.config.scaling_factor
            del pixels
            expected_shape = (
                1, 4, config.model.height // 8, config.model.width // 8
            )
            if tuple(latents.shape) != expected_shape:
                raise ValueError(
                    f"DDIM inversion expected latent shape {expected_shape}, "
                    f"got {tuple(latents.shape)}"
                )
            pipeline.maybe_free_model_hooks()

            embeddings = editor.encode_prompts(["", prompt])
            pipeline.maybe_free_model_hooks()
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
            artifact = InversionArtifact(
                method_id=self.method_id,
                model_id=config.model.model_id,
                model_revision=config.model.revision,
                dataset_ref=context.dataset_ref,
                dataset_fingerprint=context.dataset_fingerprint,
                sample_uid=uid,
                scheduler_id="ddim",
                scheduler_config=scheduler_config,
                eta=config.sampling.eta,
                timesteps=timesteps,
                terminal_latent=latents.detach().to(device="cpu").contiguous(),
            )
            editor.validate_artifact(artifact)
            return artifact
        finally:
            pipeline.maybe_free_model_hooks()
