"""Pivotal DDIM inversion with per-timestep null-text optimization.

Based on https://null-text-inversion.github.io/. Only unconditional text
embeddings are optimized; the shared model and conditional caption stay fixed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import partial
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

import torch
import yaml

from ...artifacts import ArtifactCompatibilityError, InversionArtifact
from ...config import ConfigError, ExperimentConfig
from ...config.method_parameters import MethodGuidanceSettings
from ..base import InversionMethod
from ..context import InversionContext
from ..components import prepare_sources, conditional_pivots
from ..validation import inverse_scheduler as _inverse_scheduler, require_finite_many, validate_sampling
from ..hooks import BatchedDenoisingHook, DenoisingHook, DenoisingStepState
from .common import (
    _ArtifactBatchHook,
    model_execution,
    GUIDANCE_KEY as _GUIDANCE_KEY, build_artifact, guidance_state,
    require_finite, validate_ddim_scheduler,
    validate_guidance_compatibility, validate_guidance_state,
)
from .source_embeddings import (
    SOURCE_EMBEDDING_VERSION, reject_trajectory_overwrite, replay_source_embeddings,
    source_embedding_state, validate_source_embeddings,
)

if TYPE_CHECKING:
    from diffusers import DDIMInverseScheduler


_SETTINGS_FILENAME = "Null_text.yaml"
_EMBEDDINGS_KEY = "null_text_embeddings"
_INNER_STEPS_KEY = "actual_num_inner_steps"
_RECONSTRUCTION_MSE_KEY = "reconstruction_mse"
_OPTIMIZATION_GUIDANCE_KEY = "optimization_guidance_scale"
_PIVOT_GUIDANCE_KEY = "pivot_guidance_scale"
_require_finite = partial(require_finite, label="Null-text")


class _OptimizationObserver(Protocol):
    """Observe existing optimization evaluations without changing the trajectory."""

    def start_step(self, step_index: int, timestep: int, learning_rate: float, tolerance: float) -> None: ...

    def before_update(
        self, active: Sequence[int], losses: torch.Tensor,
        gradients: Sequence[torch.Tensor], parameters: Sequence[torch.Tensor], dtype: torch.dtype,
    ) -> None: ...

    def after_update(self, active: Sequence[int], parameters: Sequence[torch.Tensor], dtype: torch.dtype) -> None: ...

    def finish_step(self, errors: torch.Tensor, counts: Sequence[int], latent: torch.Tensor) -> None: ...


@dataclass(frozen=True, slots=True)
class _NullTextSettings(MethodGuidanceSettings):
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
        guidance = MethodGuidanceSettings.from_mapping(values)
        if guidance.inversion_guidance_scale < 1:
            raise ConfigError("null-text inversion_guidance_scale must be >= 1")
        return cls(num_inner_steps=steps, **numbers, **asdict(guidance))


def _artifact_state(artifact: InversionArtifact) -> tuple[torch.Tensor, float, float]:
    """Validate state without loading hyperparameters or a model runtime."""
    reject_trajectory_overwrite(artifact, "Null-text")
    validate_source_embeddings(artifact, "Null-text")
    if artifact.method_id != "null-text":
        raise ValueError("Null-text replay requires a null-text artifact")
    if artifact.eta != 0:
        raise ValueError("Null-text replay requires artifact eta == 0")
    if _OPTIMIZATION_GUIDANCE_KEY not in artifact.per_step_state:
        raise ArtifactCompatibilityError("Null-text artifact lacks editing-CFG optimization state; regenerate it")
    if _PIVOT_GUIDANCE_KEY not in artifact.per_step_state:
        raise ArtifactCompatibilityError("Null-text artifact lacks inversion-CFG pivot state; regenerate it")
    required = {_EMBEDDINGS_KEY, _GUIDANCE_KEY, _OPTIMIZATION_GUIDANCE_KEY, _PIVOT_GUIDANCE_KEY}
    keys = set(artifact.per_step_state)
    if not required <= keys or keys - required - {_INNER_STEPS_KEY, _RECONSTRUCTION_MSE_KEY}:
        raise ValueError(
            "Null-text artifacts require null_text_embeddings, guidance_scale, "
            "pivot_guidance_scale and optimization_guidance_scale state"
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
    pivot_scale = validate_guidance_state(
        artifact, "Null-text", null_text=True, state_key=_PIVOT_GUIDANCE_KEY,
    )
    if pivot_scale != scale:
        raise ArtifactCompatibilityError("Null-text pivot guidance scale does not match inversion CFG")
    optimization_scale = validate_guidance_state(
        artifact, "Null-text", null_text=True, state_key=_OPTIMIZATION_GUIDANCE_KEY,
    )
    inner_steps = artifact.per_step_state.get(_INNER_STEPS_KEY)
    if inner_steps is not None and (
        not isinstance(inner_steps, torch.Tensor)
        or inner_steps.dtype != torch.int64
        or tuple(inner_steps.shape) != (len(artifact.timesteps),)
        or not torch.all(inner_steps >= 0 if optimization_scale == 1 else inner_steps > 0).item()
    ):
        raise ValueError(
            "Null-text actual_num_inner_steps must be an int64 [timesteps] tensor "
            "with positive counts, or nonnegative counts at optimization guidance 1"
        )
    reconstruction_mse = artifact.per_step_state.get(_RECONSTRUCTION_MSE_KEY)
    if reconstruction_mse is not None:
        if (
            not isinstance(reconstruction_mse, torch.Tensor)
            or not reconstruction_mse.is_floating_point()
            or tuple(reconstruction_mse.shape) != (len(artifact.timesteps),)
        ):
            raise ValueError("Null-text reconstruction_mse must be a floating [timesteps] tensor")
        _require_finite(reconstruction_mse, "saved reconstruction MSE")
        if not torch.all(reconstruction_mse >= 0).item():
            raise ValueError("Null-text reconstruction_mse must be nonnegative")
    return embeddings, scale, optimization_scale


class _NullTextHook(DenoisingHook):
    """Use each optimized embedding in both unconditional caption branches."""

    def __init__(self, artifact: InversionArtifact) -> None:
        self._embeddings, _, self._optimization_scale = _artifact_state(artifact)
        self._timesteps = artifact.timesteps

    @property
    def reconstruction_guidance_scale(self) -> float:
        return self._optimization_scale

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


class _NullTextBatchHook(_ArtifactBatchHook):
    """Transfer one batch of optimized embeddings per denoising step."""

    def __init__(self, artifacts: Sequence[InversionArtifact]) -> None:
        super().__init__(artifacts, "Null-text")
        self._embeddings = torch.cat([_artifact_state(artifact)[0] for artifact in artifacts], dim=1)
        scales = [_artifact_state(artifact)[2] for artifact in artifacts]
        if any(scale != scales[0] for scale in scales):
            raise ValueError("Null-text batch artifacts must share optimization guidance")
        self._optimization_scale = scales[0]

    @property
    def reconstruction_guidance_scale(self) -> float:
        return self._optimization_scale

    def before_step(
        self, step_index: int, timestep: int, state: DenoisingStepState,
    ) -> DenoisingStepState:
        self._validate_step(step_index, timestep, state)
        embedding = self._embeddings[step_index]
        negative = state.negative_prompt_embeddings
        if tuple(negative.shape) != (2 * self.batch_size, *embedding.shape[1:]):
            raise ValueError("Null-text batch hook received invalid negative embedding dimensions")
        embedding = embedding.to(device=negative.device, dtype=negative.dtype)
        _require_finite(embedding, "replay embeddings")
        return replace(state, negative_prompt_embeddings=torch.cat((embedding, embedding)))


class NullTextInversion(InversionMethod):
    """Invert source batches with independent unconditional embedding optimization."""

    def __init__(self) -> None:
        self._settings: _NullTextSettings | None = None

    @property
    def method_id(self) -> str:
        return "null-text"

    def _inversion_settings(self, config: ExperimentConfig) -> _NullTextSettings:
        if self._settings is None:
            self._settings = _NullTextSettings.load(config.project.method_h_params / _SETTINGS_FILENAME)
        return self._settings

    def method_h_params(self, config: ExperimentConfig) -> Mapping[str, Any]:
        return asdict(self._inversion_settings(config))

    def inversion_cache_parameters(self, context: InversionContext) -> Mapping[str, Any]:
        settings = self._inversion_settings(context.config)
        return {
            "source_embedding_version": SOURCE_EMBEDDING_VERSION,
            "optimization_guidance_version": 1,
            "pivot_guidance_version": 1,
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
        self._inversion_settings(config)

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
        embeddings, guidance_scale, optimization_scale = _artifact_state(artifact)
        validate_source_embeddings(artifact, "Null-text", context)
        artifact.validate_compatibility(
            context.config,
            dataset_ref=context.dataset_ref,
            dataset_fingerprint=context.dataset_fingerprint,
            timesteps=context.editor.expected_timesteps,
            scheduler_config=context.editor.scheduler_config,
        )
        validate_guidance_compatibility(guidance_scale, context, "Null-text")
        if not math.isclose(optimization_scale, context.config.editing_guidance_scale, rel_tol=0, abs_tol=1e-9):
            raise ArtifactCompatibilityError("Null-text optimization guidance scale does not match editing CFG")
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

    def replay_source_embeddings(self, artifacts: Sequence[InversionArtifact],
                                 context: InversionContext) -> tuple[torch.Tensor, torch.Tensor] | None:
        return replay_source_embeddings(artifacts, "Null-text", context)

    def create_denoising_hook(self, artifact: InversionArtifact) -> DenoisingHook:
        return _NullTextHook(artifact)

    def create_batch_denoising_hook(
        self, artifacts: Sequence[InversionArtifact],
    ) -> BatchedDenoisingHook | None:
        if type(self) is not NullTextInversion:
            return None
        return _NullTextBatchHook(artifacts)

    def result_metadata(self, artifact: InversionArtifact) -> Mapping[str, Any]:
        inner_steps = artifact.per_step_state.get(_INNER_STEPS_KEY)
        counts = None if inner_steps is None else inner_steps.detach().cpu().tolist()
        reconstruction_mse = artifact.per_step_state.get(_RECONSTRUCTION_MSE_KEY)
        errors = None if reconstruction_mse is None else reconstruction_mse.detach().cpu().tolist()
        return {
            _OPTIMIZATION_GUIDANCE_KEY: _artifact_state(artifact)[2],
            _INNER_STEPS_KEY: counts,
            "total_inner_steps": None if counts is None else sum(counts),
            _RECONSTRUCTION_MSE_KEY: errors,
            "final_reconstruction_mse": None if errors is None else errors[-1],
        }

    @staticmethod
    def _optimize(
        pivots: list[torch.Tensor],
        unconditional: torch.Tensor,
        conditional: torch.Tensor,
        settings: _NullTextSettings,
        context: InversionContext,
        *, observer: _OptimizationObserver | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        editor = context.editor
        scale = context.config.editing_guidance_scale
        step_count = len(editor.expected_timesteps)
        latent = editor.to_device(pivots[-1], dtype=editor.dtype)
        batch_size = latent.shape[0]
        saved_embeddings: list[torch.Tensor] = []
        inner_steps: list[list[int]] = []
        reconstruction_errors: list[torch.Tensor] = []
        with torch.enable_grad():
            for step_index, timestep in enumerate(editor.scheduler.timesteps):
                target = editor.to_device(
                    pivots[-step_index - 2], dtype=editor.dtype
                )
                if scale == 1:
                    # Conditional-only prediction is independent of null-text embeddings.
                    # Preserve the empty embedding for target replay and record no updates.
                    with torch.no_grad():
                        if observer is not None:
                            observer.start_step(
                                step_index, int(timestep),
                                settings.learning_rate * (1 - step_index / (2 * step_count)),
                                settings.early_stop_epsilon + step_index * settings.epsilon_increment,
                            )
                        embedding = unconditional.to(dtype=editor.dtype)
                        saved_embeddings.append(embedding.detach().to(device="cpu").contiguous())
                        prediction = editor.pipeline.unet(
                            editor.scheduler.scale_model_input(latent, timestep), timestep,
                            encoder_hidden_states=conditional,
                        ).sample
                        latent = editor.scheduler.step(
                            prediction, timestep, latent, eta=0,
                        ).prev_sample.detach()
                        _require_finite(latent, "conditional trajectory latent")
                        errors = (latent.float() - target.float()).square().flatten(1).mean(1)
                        _require_finite(errors, "reconstruction MSE")
                        errors_cpu = errors.to(device="cpu")
                        counts = [0] * batch_size
                        inner_steps.append(counts)
                        reconstruction_errors.append(errors_cpu)
                        if observer is not None:
                            observer.finish_step(errors_cpu, counts, latent)
                    continue
                # Adam's parameter and moments stay float32 even for a half UNet.
                parameters = [
                    embedding.detach().to(dtype=torch.float32).clone().requires_grad_(True)
                    for embedding in unconditional.split(1)
                ]
                optimizer = torch.optim.Adam(
                    parameters,
                    lr=settings.learning_rate * (1 - step_index / (2 * step_count)),
                    foreach=editor.device.type == "cuda",
                )
                active = list(range(batch_size))
                counts = [0] * batch_size
                tolerance = (
                    settings.early_stop_epsilon + step_index * settings.epsilon_increment
                )
                if observer is not None:
                    observer.start_step(
                        step_index, int(timestep), optimizer.param_groups[0]["lr"], tolerance,
                    )
                model_input = editor.scheduler.scale_model_input(latent, timestep)
                with torch.no_grad():
                    conditional_prediction = editor.pipeline.unet(
                        model_input, timestep, encoder_hidden_states=conditional
                    ).sample
                for _ in range(settings.num_inner_steps):
                    indices = torch.tensor(active, device=editor.device, dtype=torch.long)
                    # Inactive parameters retain grad=None, so Adam skips their moments and counters.
                    optimizer.zero_grad(set_to_none=True)
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
                    losses = (reconstructed.float() - active_target.float()).square().flatten(1).mean(1)
                    _require_finite(losses, "optimization loss")
                    losses.sum().backward()
                    gradients = [parameters[index].grad for index in active]
                    if any(gradient is None for gradient in gradients):
                        raise RuntimeError("Null-text optimization did not receive embedding gradients")
                    require_finite_many(gradients, "embedding gradients", "Null-text")
                    if observer is not None:
                        observer.before_update(active, losses, gradients, parameters, editor.dtype)
                    optimizer.step()
                    if observer is not None:
                        observer.after_update(active, parameters, editor.dtype)
                    for index in active:
                        counts[index] += 1
                    require_finite_many(
                        [parameters[index] for index in active], "optimized embeddings", "Null-text",
                    )
                    loss_values = losses.detach().cpu().tolist()
                    active = [
                        index for index, loss in zip(active, loss_values, strict=True)
                        if loss >= tolerance
                    ]
                    # Release backward graphs before the next call or offload cleanup.
                    del prediction, guided, reconstructed, losses, active_embeddings
                    if not active:
                        break
                inner_steps.append(counts)
                with torch.no_grad():
                    unconditional = torch.cat(parameters).detach()
                    del parameters, optimizer, gradients
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
                    # Measure after the last Adam update using the saved embedding.
                    errors = (latent.float() - target.float()).square().flatten(1).mean(1)
                    _require_finite(errors, "reconstruction MSE")
                    errors_cpu = errors.to(device="cpu")
                    reconstruction_errors.append(errors_cpu)
                    if observer is not None:
                        observer.finish_step(errors_cpu, counts, latent)
        return (
            torch.stack(saved_embeddings).contiguous(),
            torch.tensor(inner_steps, dtype=torch.int64, device="cpu"),
            torch.stack(reconstruction_errors).contiguous(),
        )

    def invert(
        self, sample: Mapping[str, Any], context: InversionContext
    ) -> InversionArtifact:
        return self.invert_batch([sample], context)[0]

    @torch.inference_mode(False)
    def invert_batch(
        self, samples: Sequence[Mapping[str, Any]], context: InversionContext
    ) -> list[InversionArtifact]:
        """Build pivots at inversion CFG, then optimize using editing CFG."""
        uids, prompts, images = self._validate_batch(samples, context)
        self.validate_inversion_config(context.config)
        inverse_scheduler = self._prepare_inverse_scheduler(context)
        settings = self._inversion_settings(context.config)

        editor = context.editor

        pipeline = editor.pipeline
        parameter_flags = [
            (parameter, parameter.requires_grad)
            for parameter in pipeline.unet.parameters()
        ]
        with model_execution(pipeline, restorations=[
            partial(parameter.requires_grad_, requires_grad)
            for parameter, requires_grad in parameter_flags
        ]):
            pipeline.maybe_free_model_hooks()
            for parameter, _ in parameter_flags:
                parameter.requires_grad_(False)
            sources = prepare_sources(uids, prompts, images, context, "Null-text")
            unconditional, conditional = sources.embeddings.chunk(2)
            pivots = conditional_pivots(
                sources, uids, inverse_scheduler, context, "Null-text",
                guidance_scale=context.config.inversion_guidance_scale,
            )
            # Whole-model offload keeps the UNet resident until explicit cleanup.
            # Do not call another pipeline component or offload it during backward.
            optimized, inner_steps, reconstruction_mse = self._optimize(
                pivots, unconditional, conditional, settings, context
            )
            encodings = source_embedding_state(sources.embeddings)
            artifacts: list[InversionArtifact] = []
            for index, uid in enumerate(uids):
                artifact = build_artifact(self.method_id, context, uid, pivots[-1][index:index + 1], {
                    _EMBEDDINGS_KEY: optimized[:, index:index + 1],
                    _GUIDANCE_KEY: guidance_state(context),
                    _PIVOT_GUIDANCE_KEY: guidance_state(context),
                    _OPTIMIZATION_GUIDANCE_KEY: torch.full(
                        (len(editor.expected_timesteps),), context.config.editing_guidance_scale,
                        dtype=torch.float64, device="cpu",
                    ),
                    _INNER_STEPS_KEY: inner_steps[:, index],
                    _RECONSTRUCTION_MSE_KEY: reconstruction_mse[:, index],
                }, static_state=encodings[index])
                self.validate_replay(artifact, context)
                artifacts.append(artifact)
            return artifacts
