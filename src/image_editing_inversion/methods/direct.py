"""Conditional DDIM inversion with source-only latent residual correction.

Based on Ju et al.'s Direct Inversion, later published as PnP Inversion:
https://github.com/cure-lab/PnPInversion
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import TYPE_CHECKING, Any, Mapping

from PIL import Image
import torch

from ..artifacts import ArtifactCompatibilityError, InversionArtifact
from .base import InversionMethod
from .context import InversionContext
from .hooks import DenoisingHook, DenoisingStepState

if TYPE_CHECKING:
    from diffusers import DDIMInverseScheduler


_OFFSETS_KEY = "direct_inversion_offsets"
_GUIDANCE_KEY = "guidance_scale"


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"Direct inversion produced non-finite {name}")


def _validate_scheduler(scheduler_config: Mapping[str, Any]) -> None:
    for name, required in (
        ("prediction_type", "epsilon"),
        ("timestep_spacing", "leading"),
        ("clip_sample", False),
        ("thresholding", False),
    ):
        if scheduler_config.get(name) != required:
            raise ValueError(
                f"Direct inversion requires DDIM {name} == {required!r}"
            )


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
    guidance = artifact.per_step_state[_GUIDANCE_KEY]
    if (
        not isinstance(guidance, torch.Tensor)
        or not guidance.is_floating_point()
        or tuple(guidance.shape) != (len(artifact.timesteps),)
    ):
        raise ValueError(
            "Direct inversion guidance_scale must be a floating [timesteps] tensor"
        )
    _require_finite(artifact.terminal_latent, "terminal latent")
    _require_finite(offsets, "saved offsets")
    _require_finite(guidance, "saved guidance scale")
    scale = float(guidance[0].item())
    if scale < 0 or not torch.all(guidance == guidance[0]).item():
        raise ValueError("Direct inversion guidance_scale must be constant and nonnegative")
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
    """Invert one sample and record its per-step source reconstruction residuals."""

    @property
    def method_id(self) -> str:
        return "direct-inversion"

    def inversion_cache_parameters(self, context: InversionContext) -> Mapping[str, Any]:
        return {}

    @staticmethod
    def _validate_context(context: InversionContext) -> dict[str, Any]:
        config = context.config
        if config != context.editor.config:
            raise ValueError("Direct inversion context must use the editor's configuration")
        if config.sampling.eta != 0:
            raise ValueError("Direct inversion requires sampling.eta == 0")
        scale = config.sampling.guidance_scale
        if not math.isfinite(scale) or scale < 0:
            raise ValueError(
                "Direct inversion requires finite, nonnegative sampling.guidance_scale"
            )
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
        if not math.isclose(
            guidance_scale, context.config.sampling.guidance_scale,
            rel_tol=0, abs_tol=1e-9,
        ):
            raise ArtifactCompatibilityError(
                "Direct inversion artifact guidance scale does not match this run"
            )
        self._validate_context(context)

    def create_denoising_hook(self, artifact: InversionArtifact) -> DenoisingHook:
        return _DirectInversionHook(artifact)

    @staticmethod
    def _encode_source(image: Image.Image, context: InversionContext) -> torch.Tensor:
        editor = context.editor
        model = context.config.model
        pixels = editor.pipeline.image_processor.preprocess(
            image.convert("RGB"), height=model.height, width=model.width
        ).to(device=editor.device, dtype=editor.dtype)
        latent = editor.pipeline.vae.encode(pixels).latent_dist.mode()
        latent = latent * editor.pipeline.vae.config.scaling_factor
        expected_shape = (1, 4, model.height // 8, model.width // 8)
        if tuple(latent.shape) != expected_shape:
            raise ValueError(
                f"Direct inversion expected latent shape {expected_shape}, "
                f"got {tuple(latent.shape)}"
            )
        _require_finite(latent, "encoded latent")
        return latent.detach()

    @staticmethod
    def _pivot_trajectory(
        latent: torch.Tensor,
        conditional: torch.Tensor,
        scheduler: DDIMInverseScheduler,
        context: InversionContext,
    ) -> list[torch.Tensor]:
        # Keep the pivot history on CPU rather than occupying accelerator memory.
        pivots = [latent.detach().to(device="cpu").contiguous()]
        for timestep in scheduler.timesteps:
            model_input = scheduler.scale_model_input(latent, timestep)
            prediction = context.editor.pipeline.unet(
                model_input, timestep, encoder_hidden_states=conditional
            ).sample
            latent = scheduler.step(prediction, timestep, latent).prev_sample
            _require_finite(latent, "pivot latent")
            pivots.append(latent.detach().to(device="cpu").contiguous())
        return pivots

    @staticmethod
    def _calculate_offsets(
        pivots: list[torch.Tensor],
        embeddings: torch.Tensor,
        context: InversionContext,
    ) -> torch.Tensor:
        editor = context.editor
        scale = context.config.sampling.guidance_scale
        latent = pivots[-1].to(device=editor.device, dtype=editor.dtype)
        offsets: list[torch.Tensor] = []
        for step_index, timestep in enumerate(editor.scheduler.timesteps):
            previous_pivot = pivots[-step_index - 2].to(
                device=editor.device, dtype=editor.dtype
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

    @torch.inference_mode()
    def invert(
        self, sample: Mapping[str, Any], context: InversionContext
    ) -> InversionArtifact:
        """Build conditional pivots and record residuals at the run's guidance."""
        if not isinstance(sample, Mapping):
            raise TypeError("Direct inversion sample must be a mapping")
        uid = sample.get("uid")
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("Dataset uid must be a nonempty string")
        prompt = sample.get("source_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Dataset source_prompt must be a nonempty string")
        image = sample.get("source_img")
        if not isinstance(image, Image.Image):
            raise TypeError("Dataset source_img must be a Pillow image")
        scheduler_config = self._validate_context(context)

        # Import only during inversion so entry-point discovery stays lightweight.
        from diffusers import DDIMInverseScheduler

        editor = context.editor
        config = context.config
        try:
            inverse_scheduler = DDIMInverseScheduler.from_config(scheduler_config)
            inverse_scheduler.set_timesteps(
                config.sampling.num_inference_steps, device=editor.device
            )
        except (ValueError, NotImplementedError) as exc:
            raise ValueError(
                f"Direct inversion cannot use the editor's scheduler settings: {exc}"
            ) from exc
        timesteps = editor.expected_timesteps
        if tuple(int(step) for step in inverse_scheduler.timesteps.tolist()) != tuple(
            reversed(timesteps)
        ):
            raise ValueError("Direct inverse timesteps must reverse the editor's schedule")

        pipeline = editor.pipeline
        try:
            pipeline.maybe_free_model_hooks()
            latent = self._encode_source(image, context)
            pipeline.maybe_free_model_hooks()
            embeddings = editor.encode_prompts(["", prompt]).detach()
            _require_finite(embeddings, "caption embeddings")
            pipeline.maybe_free_model_hooks()
            pivots = self._pivot_trajectory(
                latent, embeddings[1:], inverse_scheduler, context
            )
            offsets = self._calculate_offsets(pivots, embeddings, context)
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
                terminal_latent=pivots[-1],
                per_step_state={
                    _OFFSETS_KEY: offsets,
                    _GUIDANCE_KEY: torch.full(
                        (len(timesteps),), config.sampling.guidance_scale,
                        dtype=torch.float64, device="cpu",
                    ),
                },
            )
            self.validate_replay(artifact, context)
            return artifact
        finally:
            pipeline.maybe_free_model_hooks()
