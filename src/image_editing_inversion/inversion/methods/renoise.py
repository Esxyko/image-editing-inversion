"""Core ReNoise inversion with iterative deterministic DDIM noising.

Based on https://github.com/garibida/ReNoise-Inversion. Noise regularization
and stochastic noise correction are outside this adapter's scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from ...artifacts import InversionArtifact
from ...config import ConfigError, ExperimentConfig
from ..base import InversionMethod
from ..context import InversionContext
from ..validation import validate_sampling
from ..hooks import DenoisingHook, DenoisingStepState
from .common import (
    GUIDANCE_KEY as _GUIDANCE_KEY, build_artifact, guidance_state, prepare_sources,
    require_finite, validate_ddim_scheduler,
    validate_guidance_compatibility, validate_guidance_state,
)


_SETTINGS_PATH = Path("method_h_params/ReNoise.yaml")
_LOW_TIMESTEP_BOUNDARY = 250


_require_finite = partial(require_finite, label="ReNoise")


@dataclass(frozen=True, slots=True)
class _ReNoiseSettings:
    num_renoise_steps: int = 9
    max_num_renoise_steps_first_step: int = 5
    average_latent_estimations: bool = True
    average_first_step_range: tuple[int, int] = (0, 5)
    average_step_range: tuple[int, int] = (8, 10)

    @classmethod
    def load(cls, path: Path) -> _ReNoiseSettings:
        try:
            with path.open("r", encoding="utf-8") as stream:
                values = yaml.safe_load(stream)
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ConfigError(
                f"Cannot load ReNoise hyperparameters at {path.resolve()}: {exc}"
            ) from exc
        if not isinstance(values, dict) or any(
            not isinstance(key, str) for key in values
        ):
            raise ConfigError("ReNoise hyperparameters must be a mapping with string keys")
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ConfigError(
                "Unknown ReNoise hyperparameters: " + ", ".join(sorted(unknown))
            )
        defaults = cls()
        counts: dict[str, int] = {}
        for name in ("num_renoise_steps", "max_num_renoise_steps_first_step"):
            value = values.get(name, getattr(defaults, name))
            if type(value) is not int or value < 0:
                raise ConfigError(f"ReNoise {name} must be a nonnegative integer")
            counts[name] = value
        averaging = values.get(
            "average_latent_estimations", defaults.average_latent_estimations
        )
        if type(averaging) is not bool:
            raise ConfigError("ReNoise average_latent_estimations must be a boolean")
        budgets = {
            "average_first_step_range": min(
                counts["num_renoise_steps"], counts["max_num_renoise_steps_first_step"]
            ) + 1,
            "average_step_range": counts["num_renoise_steps"] + 1,
        }
        ranges: dict[str, tuple[int, int]] = {}
        for name, budget in budgets.items():
            value = values.get(name, getattr(defaults, name))
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(type(index) is not int for index in value)
                or not 0 <= value[0] < value[1] <= budget
            ):
                raise ConfigError(
                    f"ReNoise {name} must be [start, end] with "
                    f"0 <= start < end <= {budget} (exclusive end)"
                )
            ranges[name] = (value[0], value[1])
        return cls(**counts, average_latent_estimations=averaging, **ranges)

    def for_timestep(self, timestep: int) -> tuple[int, tuple[int, int]]:
        if timestep < _LOW_TIMESTEP_BOUNDARY:
            return (
                min(self.num_renoise_steps, self.max_num_renoise_steps_first_step),
                self.average_first_step_range,
            )
        return self.num_renoise_steps, self.average_step_range


_validate_scheduler = partial(validate_ddim_scheduler, label="ReNoise")


def _artifact_state(artifact: InversionArtifact) -> float:
    """Validate replay state without loading settings, Diffusers, or a runtime."""
    if artifact.method_id != "renoise":
        raise ValueError("ReNoise replay requires a renoise artifact")
    if artifact.scheduler_id != "ddim" or artifact.eta != 0:
        raise ValueError("ReNoise replay requires DDIM with artifact eta == 0")
    _validate_scheduler(artifact.scheduler_config)
    if set(artifact.per_step_state) != {_GUIDANCE_KEY}:
        raise ValueError("ReNoise artifacts require guidance_scale state only")
    _require_finite(artifact.terminal_latent, "terminal latent")
    return validate_guidance_state(artifact, "ReNoise")


class _ReNoiseHook(DenoisingHook):
    """Validate replay steps while leaving the shared denoising state unchanged."""

    def __init__(self, artifact: InversionArtifact) -> None:
        _artifact_state(artifact)
        self._timesteps = artifact.timesteps
        self._latent_shape = (2, *artifact.terminal_latent.shape[1:])

    def before_step(
        self, step_index: int, timestep: int, state: DenoisingStepState
    ) -> DenoisingStepState:
        if type(step_index) is not int or not 0 <= step_index < len(self._timesteps):
            raise ValueError("ReNoise hook received an invalid step index")
        if type(timestep) is not int or timestep != self._timesteps[step_index]:
            raise ValueError("ReNoise hook timestep does not match its artifact")
        if tuple(state.latents.shape) != self._latent_shape:
            raise ValueError(
                f"ReNoise hook expected latents of shape {self._latent_shape}, "
                f"got {tuple(state.latents.shape)}"
            )
        return state


class ReNoiseInversion(InversionMethod):
    """Invert source batches by refining and averaging DDIM noise predictions."""

    def __init__(self) -> None:
        self._settings: _ReNoiseSettings | None = None

    @property
    def method_id(self) -> str:
        return "renoise"

    def _inversion_settings(self) -> _ReNoiseSettings:
        if self._settings is None:
            self._settings = _ReNoiseSettings.load(_SETTINGS_PATH)
        return self._settings

    def inversion_cache_parameters(self, context: InversionContext) -> Mapping[str, Any]:
        settings = self._inversion_settings()
        # Cache effective predictions, not unused windows or caps. Predictions
        # after the averaging window cannot change the selected final average.
        schedule = []
        for timestep in context.editor.expected_timesteps:
            refinements, (start, end) = settings.for_timestep(timestep)
            selected = [start, end] if settings.average_latent_estimations else [refinements, refinements + 1]
            schedule.append({"prediction_range": selected})
        return {"prediction_schedule": schedule}

    def validate_inversion_config(self, config: ExperimentConfig) -> None:
        validate_sampling(config, "ReNoise")
        self._inversion_settings()

    def validate_inversion_context(self, context: InversionContext) -> None:
        self._prepare_coefficients(context)

    def _prepare_coefficients(
        self, context: InversionContext,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        self._validate_context(context)
        from diffusers import DDIMScheduler

        if not isinstance(context.editor.scheduler, DDIMScheduler):
            raise ValueError("ReNoise inversion requires the editor's DDIM scheduler")
        return self._coefficients(context)

    @staticmethod
    def _validate_context(context: InversionContext) -> dict[str, Any]:
        config = context.config
        if config != context.editor.config:
            raise ValueError("ReNoise context must use the editor's configuration")
        validate_sampling(config, "ReNoise")
        scheduler_config = context.editor.scheduler_config
        _validate_scheduler(scheduler_config)
        return scheduler_config

    def validate_replay(
        self, artifact: InversionArtifact, context: InversionContext
    ) -> None:
        guidance_scale = _artifact_state(artifact)
        artifact.validate_compatibility(
            context.config,
            dataset_ref=context.dataset_ref,
            dataset_fingerprint=context.dataset_fingerprint,
            timesteps=context.editor.expected_timesteps,
            scheduler_config=context.editor.scheduler_config,
        )
        validate_guidance_compatibility(guidance_scale, context, "ReNoise")
        self._validate_context(context)

    def create_denoising_hook(self, artifact: InversionArtifact) -> DenoisingHook:
        return _ReNoiseHook(artifact)

    @staticmethod
    def _coefficients(
        context: InversionContext,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        """Express each editor DDIM step as previous = a * current + b * noise."""
        scheduler = context.editor.scheduler
        steps = context.config.sampling.num_inference_steps
        timesteps = context.editor.expected_timesteps
        train_steps = scheduler.config.num_train_timesteps
        if scheduler.num_inference_steps != steps or len(timesteps) != steps:
            raise ValueError("ReNoise scheduler step count must match this run")
        stride = train_steps // steps
        if (
            stride < 1
            or not timesteps
            or any(not 0 <= step < train_steps for step in timesteps)
            or timesteps[-1] - stride >= 0
            or any(
                left - right != stride for left, right in zip(timesteps, timesteps[1:])
            )
        ):
            raise ValueError("ReNoise requires a complete, leading DDIM timestep schedule")
        coefficients = []
        for timestep in reversed(timesteps):
            previous_timestep = timestep - stride
            alpha_t = scheduler.alphas_cumprod[timestep]
            alpha_previous = (
                scheduler.alphas_cumprod[previous_timestep]
                if previous_timestep >= 0 else scheduler.final_alpha_cumprod
            )
            alpha_t = context.editor.to_device(alpha_t, dtype=torch.float32)
            alpha_previous = context.editor.to_device(
                alpha_previous, dtype=torch.float32
            )
            for alpha in (alpha_t, alpha_previous):
                if not torch.isfinite(alpha).item() or not 0 < alpha.item() <= 1:
                    raise ValueError("ReNoise requires finite DDIM alpha products in (0, 1]")
            a = (alpha_previous / alpha_t).sqrt()
            b = (1 - alpha_previous).sqrt() - a * (1 - alpha_t).sqrt()
            _require_finite(a, "DDIM coefficient a")
            _require_finite(b, "DDIM coefficient b")
            if a.item() <= 0:
                raise ValueError("ReNoise requires a positive DDIM coefficient a")
            coefficients.append((timestep, a, b))
        return coefficients

    @staticmethod
    def _renoise_step(
        previous: torch.Tensor,
        timestep: int,
        a: torch.Tensor,
        b: torch.Tensor,
        embeddings: torch.Tensor,
        settings: _ReNoiseSettings,
        context: InversionContext,
    ) -> torch.Tensor:
        editor = context.editor
        refinements, (start, end) = settings.for_timestep(timestep)
        # Keep this anchor fixed throughout all refinements at the current t.
        previous = previous.float()
        estimate = previous
        average: torch.Tensor | None = None
        count = 0
        for index in range(refinements + 1):
            model_latent = estimate.to(dtype=editor.dtype)
            _require_finite(model_latent, "UNet input")
            model_input = torch.cat((model_latent, model_latent), dim=0)
            model_input = editor.scheduler.scale_model_input(model_input, timestep)
            prediction = editor.pipeline.unet(
                model_input, timestep, encoder_hidden_states=embeddings
            ).sample
            unconditional, conditional = prediction.chunk(2)
            guided = unconditional + context.config.sampling.guidance_scale * (
                conditional - unconditional
            )
            _require_finite(guided, "guided noise prediction")
            noise = guided.float()
            if settings.average_latent_estimations and start <= index < end:
                count += 1
                average = (
                    noise.clone() if average is None
                    else average + (noise - average) / count
                )
            estimate = (previous - b * noise) / a
            _require_finite(estimate, "renoised latent")
        if average is not None:
            estimate = (previous - b * average) / a
            _require_finite(estimate, "averaged latent")
        estimate = estimate.to(dtype=editor.dtype)
        _require_finite(estimate, "runtime-precision latent")
        return estimate

    def invert(
        self, sample: Mapping[str, Any], context: InversionContext
    ) -> InversionArtifact:
        return self.invert_batch([sample], context)[0]

    @torch.inference_mode()
    def invert_batch(
        self, samples: Sequence[Mapping[str, Any]], context: InversionContext
    ) -> list[InversionArtifact]:
        """Refine a batch of sources with independent DDIM noise estimates."""
        uids, prompts, images = self._validate_batch(samples, context)
        self.validate_inversion_config(context.config)
        coefficients = self._prepare_coefficients(context)
        settings = self._inversion_settings()
        editor = context.editor
        pipeline = editor.pipeline
        try:
            pipeline.maybe_free_model_hooks()
            sources = prepare_sources(uids, prompts, images, context, "ReNoise")
            latent, embeddings = sources.latent, sources.embeddings
            for timestep, a, b in coefficients:
                latent = self._renoise_step(
                    latent, timestep, a, b, embeddings, settings, context
                )
            terminal = latent.detach().to(device="cpu")
            artifacts: list[InversionArtifact] = []
            for index, uid in enumerate(uids):
                artifact = build_artifact(self.method_id, context, uid, terminal[index:index + 1], {
                    _GUIDANCE_KEY: guidance_state(context),
                })
                self.validate_replay(artifact, context)
                artifacts.append(artifact)
            return artifacts
        finally:
            pipeline.maybe_free_model_hooks()
