"""Pivotal DDIM inversion with per-timestep null-text optimization.

Based on https://null-text-inversion.github.io/. Only unconditional text
embeddings are optimized; the shared model and conditional caption stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
import torch.nn.functional as functional
import yaml

from ...artifacts import InversionArtifact
from ...config import ConfigError, ExperimentConfig
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


_SETTINGS_PATH = Path("method_h_params/Null_text.yaml")
_EMBEDDINGS_KEY = "null_text_embeddings"
_require_finite = partial(require_finite, label="Null-text")


@dataclass(frozen=True, slots=True)
class _NullTextSettings:
    num_inner_steps: int = 10
    learning_rate: float = 0.01
    early_stop_epsilon: float = 0.00001
    epsilon_increment: float = 0.00002

    @classmethod
    def load(cls, path: Path) -> _NullTextSettings:
        try:
            with path.open("r", encoding="utf-8") as stream:
                values = yaml.safe_load(stream)
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ConfigError(
                f"Cannot load null-text hyperparameters at {path.resolve()}: {exc}"
            ) from exc
        if not isinstance(values, dict) or any(
            not isinstance(key, str) for key in values
        ):
            raise ConfigError(
                "Null-text hyperparameters must be a mapping with string keys"
            )
        defaults = cls()
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ConfigError(
                "Unknown null-text hyperparameters: " + ", ".join(sorted(unknown))
            )
        steps = values.get("num_inner_steps", defaults.num_inner_steps)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ConfigError("null-text num_inner_steps must be a positive integer")
        numbers: dict[str, float] = {}
        for name in ("learning_rate", "early_stop_epsilon", "epsilon_increment"):
            value = values.get(name, getattr(defaults, name))
            try:
                valid = (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    and (value > 0 if name == "learning_rate" else value >= 0)
                )
            except OverflowError:
                valid = False
            if not valid:
                bound = "positive" if name == "learning_rate" else "nonnegative"
                raise ConfigError(f"null-text {name} must be a finite {bound} number")
            numbers[name] = float(value)
        return cls(num_inner_steps=steps, **numbers)


def _artifact_state(artifact: InversionArtifact) -> tuple[torch.Tensor, float]:
    """Validate state without loading hyperparameters or a model runtime."""
    if artifact.method_id != "null-text":
        raise ValueError("Null-text replay requires a null-text artifact")
    if artifact.eta != 0:
        raise ValueError("Null-text replay requires artifact eta == 0")
    if set(artifact.per_step_state) != {_EMBEDDINGS_KEY, _GUIDANCE_KEY}:
        raise ValueError(
            "Null-text artifacts require null_text_embeddings and guidance_scale state"
        )
    embeddings = artifact.per_step_state[_EMBEDDINGS_KEY]
    if (
        not isinstance(embeddings, torch.Tensor)
        or not embeddings.is_floating_point()
        or embeddings.ndim != 4
        or embeddings.shape[:2] != (len(artifact.timesteps), 1)
        or min(embeddings.shape[2:]) < 1
    ):
        raise ValueError(
            "Null-text embeddings must be floating tensors of shape "
            "[timesteps, 1, token_count, embedding_dim]"
        )
    _require_finite(artifact.terminal_latent, "terminal latent")
    _require_finite(embeddings, "saved embeddings")
    scale = validate_guidance_state(artifact, "Null-text", null_text=True)
    return embeddings, scale


class _NullTextHook(DenoisingHook):
    """Use each optimized embedding in both unconditional caption branches."""

    def __init__(self, artifact: InversionArtifact) -> None:
        self._embeddings, _ = _artifact_state(artifact)
        self._timesteps = artifact.timesteps

    def before_step(
        self, step_index: int, timestep: int, state: DenoisingStepState
    ) -> DenoisingStepState:
        if type(step_index) is not int or not 0 <= step_index < len(self._timesteps):
            raise ValueError("Null-text hook received an invalid step index")
        if timestep != self._timesteps[step_index]:
            raise ValueError("Null-text hook timestep does not match its artifact")
        embedding = self._embeddings[step_index]
        expected_shape = (2, *embedding.shape[1:])
        negative = state.negative_prompt_embeddings
        if tuple(negative.shape) != expected_shape:
            raise ValueError(
                "Null-text hook expected negative embeddings of shape "
                f"{expected_shape}, "
                f"got {tuple(negative.shape)}"
            )
        embedding = embedding.to(device=negative.device, dtype=negative.dtype)
        _require_finite(embedding, "replay embeddings")
        return replace(state, negative_prompt_embeddings=embedding.expand_as(negative))


class NullTextInversion(InversionMethod):
    """Invert source batches with independent unconditional embedding optimization."""

    def __init__(self) -> None:
        self._settings: _NullTextSettings | None = None

    @property
    def method_id(self) -> str:
        return "null-text"

    def _inversion_settings(self) -> _NullTextSettings:
        if self._settings is None:
            self._settings = _NullTextSettings.load(_SETTINGS_PATH)
        return self._settings

    def inversion_cache_parameters(self, context: InversionContext) -> Mapping[str, Any]:
        settings = self._inversion_settings()
        return {
            "num_inner_steps": settings.num_inner_steps,
            "learning_rate": settings.learning_rate,
            "early_stop_thresholds": [settings.early_stop_epsilon + index * settings.epsilon_increment
                                      for index in range(len(context.editor.expected_timesteps))],
        }

    def validate_inversion_config(self, config: ExperimentConfig) -> None:
        validate_sampling(config, "Null-text", null_text=True)
        if config.runtime.cpu_offload == "sequential":
            raise ValueError(
                "Null-text inversion does not support sequential CPU offload; "
                "use runtime.cpu_offload='none' or 'model'"
            )
        self._inversion_settings()

    def validate_inversion_context(self, context: InversionContext) -> None:
        self._prepare_inverse_scheduler(context)

    def _prepare_inverse_scheduler(self, context: InversionContext) -> DDIMInverseScheduler:
        self._validate_context(context)
        return _inverse_scheduler(context, "Null-text")

    @staticmethod
    def _validate_context(context: InversionContext) -> dict[str, Any]:
        config = context.config
        if config != context.editor.config:
            raise ValueError("Null-text context must use the editor's configuration")
        validate_sampling(config, "Null-text", null_text=True)
        scheduler_config = context.editor.scheduler_config
        validate_ddim_scheduler(scheduler_config, "Null-text")
        return scheduler_config

    def validate_replay(
        self, artifact: InversionArtifact, context: InversionContext
    ) -> None:
        embeddings, guidance_scale = _artifact_state(artifact)
        artifact.validate_compatibility(
            context.config,
            dataset_ref=context.dataset_ref,
            dataset_fingerprint=context.dataset_fingerprint,
            timesteps=context.editor.expected_timesteps,
            scheduler_config=context.editor.scheduler_config,
        )
        validate_guidance_compatibility(guidance_scale, context, "Null-text")
        self._validate_context(context)
        pipeline = context.editor.pipeline
        expected_shape = (
            len(artifact.timesteps), 1,
            pipeline.tokenizer.model_max_length,
            pipeline.text_encoder.config.hidden_size,
        )
        if tuple(embeddings.shape) != expected_shape:
            raise ValueError(
                f"Null-text replay expected embedding shape {expected_shape}, "
                f"got {tuple(embeddings.shape)}"
            )

    def create_denoising_hook(self, artifact: InversionArtifact) -> DenoisingHook:
        return _NullTextHook(artifact)

    @staticmethod
    def _optimize(
        pivots: list[torch.Tensor],
        unconditional: torch.Tensor,
        conditional: torch.Tensor,
        settings: _NullTextSettings,
        context: InversionContext,
    ) -> torch.Tensor:
        editor = context.editor
        scale = context.config.sampling.guidance_scale
        step_count = len(editor.expected_timesteps)
        latent = editor.to_device(pivots[-1], dtype=editor.dtype)
        batch_size = latent.shape[0]
        saved_embeddings: list[torch.Tensor] = []
        with torch.enable_grad():
            for step_index, timestep in enumerate(editor.scheduler.timesteps):
                target = editor.to_device(
                    pivots[-step_index - 2], dtype=editor.dtype
                )
                # Adam's parameter and moments stay float32 even for a half UNet.
                parameters = [
                    embedding.detach().to(dtype=torch.float32).clone().requires_grad_(True)
                    for embedding in unconditional.split(1)
                ]
                optimizers = [
                    torch.optim.Adam(
                        [parameter],
                        lr=settings.learning_rate * (1 - step_index / (2 * step_count)),
                    )
                    for parameter in parameters
                ]
                active = list(range(batch_size))
                tolerance = (
                    settings.early_stop_epsilon + step_index * settings.epsilon_increment
                )
                model_input = editor.scheduler.scale_model_input(latent, timestep)
                with torch.no_grad():
                    conditional_prediction = editor.pipeline.unet(
                        model_input, timestep, encoder_hidden_states=conditional
                    ).sample
                for _ in range(settings.num_inner_steps):
                    indices = torch.tensor(active, device=editor.device, dtype=torch.long)
                    for index in active:
                        optimizers[index].zero_grad(set_to_none=True)
                    active_embeddings = torch.cat([parameters[index] for index in active])
                    prediction = editor.pipeline.unet(
                        model_input.index_select(0, indices), timestep,
                        encoder_hidden_states=active_embeddings.to(dtype=editor.dtype),
                    ).sample
                    guided = prediction + scale * (
                        conditional_prediction.index_select(0, indices) - prediction
                    )
                    reconstructed = editor.scheduler.step(
                        guided, timestep, latent.index_select(0, indices), eta=0
                    ).prev_sample
                    active_target = target.index_select(0, indices)
                    # Sum individual means so each gradient matches singleton inversion.
                    losses = torch.stack([
                        functional.mse_loss(
                            reconstructed[position:position + 1].float(),
                            active_target[position:position + 1].float(),
                        )
                        for position in range(len(active))
                    ])
                    _require_finite(losses, "optimization loss")
                    losses.sum().backward()
                    for index in active:
                        parameter = parameters[index]
                        if parameter.grad is None:
                            raise RuntimeError(
                                "Null-text optimization did not receive embedding gradients"
                            )
                        _require_finite(parameter.grad, "embedding gradients")
                        optimizers[index].step()
                        _require_finite(parameter, "optimized embeddings")
                    loss_values = losses.detach().cpu().tolist()
                    active = [
                        index for index, loss in zip(active, loss_values, strict=True)
                        if loss >= tolerance
                    ]
                    # Release backward graphs before the next call or offload cleanup.
                    del prediction, guided, reconstructed, losses, active_embeddings
                    if not active:
                        break
                with torch.no_grad():
                    unconditional = torch.cat(parameters).detach()
                    del parameters, optimizers, parameter
                    embedding = unconditional.to(dtype=editor.dtype)
                    _require_finite(embedding, "model embeddings")
                    saved_embeddings.append(
                        embedding.detach().to(device="cpu").contiguous()
                    )
                    prediction = editor.pipeline.unet(
                        model_input, timestep, encoder_hidden_states=embedding
                    ).sample
                    guided = prediction + scale * (conditional_prediction - prediction)
                    latent = editor.scheduler.step(
                        guided, timestep, latent, eta=0
                    ).prev_sample.detach()
                    _require_finite(latent, "optimized trajectory latent")
        return torch.stack(saved_embeddings).contiguous()

    def invert(
        self, sample: Mapping[str, Any], context: InversionContext
    ) -> InversionArtifact:
        return self.invert_batch([sample], context)[0]

    @torch.inference_mode(False)
    def invert_batch(
        self, samples: Sequence[Mapping[str, Any]], context: InversionContext
    ) -> list[InversionArtifact]:
        """Build pivots at guidance 1, then optimize using the run's guidance."""
        uids, prompts, images = self._validate_batch(samples, context)
        self.validate_inversion_config(context.config)
        inverse_scheduler = self._prepare_inverse_scheduler(context)
        settings = self._inversion_settings()

        editor = context.editor

        pipeline = editor.pipeline
        parameter_flags = [
            (parameter, parameter.requires_grad)
            for parameter in pipeline.unet.parameters()
        ]
        try:
            pipeline.maybe_free_model_hooks()
            for parameter, _ in parameter_flags:
                parameter.requires_grad_(False)
            sources = prepare_sources(uids, prompts, images, context, "Null-text")
            unconditional, conditional = sources.embeddings.chunk(2)
            pivots = conditional_pivots(sources, uids, inverse_scheduler, context, "Null-text")
            # Whole-model offload keeps the UNet resident until explicit cleanup.
            # Do not call another pipeline component or offload it during backward.
            optimized = self._optimize(
                pivots, unconditional, conditional, settings, context
            )
            artifacts: list[InversionArtifact] = []
            for index, uid in enumerate(uids):
                artifact = build_artifact(self.method_id, context, uid, pivots[-1][index:index + 1], {
                    _EMBEDDINGS_KEY: optimized[:, index:index + 1],
                    _GUIDANCE_KEY: guidance_state(context),
                })
                self.validate_replay(artifact, context)
                artifacts.append(artifact)
            return artifacts
        finally:
            for parameter, requires_grad in parameter_flags:
                parameter.requires_grad_(requires_grad)
            pipeline.maybe_free_model_hooks()
