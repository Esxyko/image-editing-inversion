"""Pivotal DDIM inversion with per-timestep null-text optimization.

Based on https://null-text-inversion.github.io/. Only unconditional text
embeddings are optimized; the shared model and conditional caption stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from PIL import Image
import torch
import torch.nn.functional as functional
import yaml

from ..artifacts import InversionArtifact
from ..config import ConfigError
from .base import InversionMethod
from .context import InversionContext
from .hooks import DenoisingHook, DenoisingStepState

if TYPE_CHECKING:
    from diffusers import DDIMInverseScheduler


_SETTINGS_PATH = Path("method_h_params/Null_text.yaml")
_EMBEDDINGS_KEY = "null_text_embeddings"
_GUIDANCE_KEY = "guidance_scale"


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"Null-text inversion produced non-finite {name}")


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
    guidance = artifact.per_step_state[_GUIDANCE_KEY]
    if (
        not isinstance(guidance, torch.Tensor)
        or not guidance.is_floating_point()
        or tuple(guidance.shape) != (len(artifact.timesteps),)
    ):
        raise ValueError(
            "Null-text guidance_scale must be a floating [timesteps] tensor"
        )
    _require_finite(artifact.terminal_latent, "terminal latent")
    _require_finite(embeddings, "saved embeddings")
    _require_finite(guidance, "saved guidance scale")
    scale = float(guidance[0].item())
    if scale <= 1 or not torch.all(guidance == guidance[0]).item():
        raise ValueError("Null-text guidance_scale must be constant and greater than 1")
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
    """Invert one sample by optimizing only its unconditional embeddings."""

    def __init__(self) -> None:
        self._settings: _NullTextSettings | None = None

    @property
    def method_id(self) -> str:
        return "null-text"

    @staticmethod
    def _validate_context(context: InversionContext) -> dict[str, Any]:
        config = context.config
        if config != context.editor.config:
            raise ValueError("Null-text context must use the editor's configuration")
        if config.sampling.eta != 0:
            raise ValueError("Null-text inversion requires sampling.eta == 0")
        scale = config.sampling.guidance_scale
        if not math.isfinite(scale) or scale <= 1:
            raise ValueError("Null-text inversion requires sampling.guidance_scale > 1")
        scheduler_config = context.editor.scheduler_config
        for name, required in (
            ("prediction_type", "epsilon"),
            ("timestep_spacing", "leading"),
            ("clip_sample", False),
            ("thresholding", False),
        ):
            if scheduler_config.get(name) != required:
                raise ValueError(
                    f"Null-text inversion requires DDIM {name} == {required!r}"
                )
        return scheduler_config

    def validate_replay(
        self, artifact: InversionArtifact, context: InversionContext
    ) -> None:
        scheduler_config = self._validate_context(context)
        artifact.validate_compatibility(
            context.config,
            dataset_ref=context.dataset_ref,
            dataset_fingerprint=context.dataset_fingerprint,
            timesteps=context.editor.expected_timesteps,
            scheduler_config=scheduler_config,
        )
        embeddings, guidance_scale = _artifact_state(artifact)
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
        if not math.isclose(
            guidance_scale, context.config.sampling.guidance_scale,
            rel_tol=0, abs_tol=1e-9,
        ):
            raise ValueError(
                "Null-text artifact guidance scale does not match this run"
            )

    def create_denoising_hook(self, artifact: InversionArtifact) -> DenoisingHook:
        return _NullTextHook(artifact)

    @staticmethod
    @torch.no_grad()
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
                f"Null-text inversion expected latent shape {expected_shape}, "
                f"got {tuple(latent.shape)}"
            )
        _require_finite(latent, "encoded latent")
        return latent.detach()

    @staticmethod
    @torch.no_grad()
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
        latent = pivots[-1].to(device=editor.device, dtype=editor.dtype)
        saved_embeddings: list[torch.Tensor] = []
        with torch.enable_grad():
            for step_index, timestep in enumerate(editor.scheduler.timesteps):
                target = pivots[-step_index - 2].to(
                    device=editor.device, dtype=editor.dtype
                )
                # Adam's parameter and moments stay float32 even for a half UNet.
                unconditional = unconditional.detach().to(dtype=torch.float32).clone()
                unconditional.requires_grad_(True)
                optimizer = torch.optim.Adam(
                    [unconditional],
                    lr=settings.learning_rate * (1 - step_index / (2 * step_count)),
                )
                tolerance = (
                    settings.early_stop_epsilon + step_index * settings.epsilon_increment
                )
                model_input = editor.scheduler.scale_model_input(latent, timestep)
                with torch.no_grad():
                    conditional_prediction = editor.pipeline.unet(
                        model_input, timestep, encoder_hidden_states=conditional
                    ).sample
                for _ in range(settings.num_inner_steps):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = editor.pipeline.unet(
                        model_input, timestep,
                        encoder_hidden_states=unconditional.to(dtype=editor.dtype),
                    ).sample
                    guided = prediction + scale * (conditional_prediction - prediction)
                    reconstructed = editor.scheduler.step(
                        guided, timestep, latent, eta=0
                    ).prev_sample
                    loss = functional.mse_loss(reconstructed.float(), target.float())
                    _require_finite(loss, "optimization loss")
                    loss.backward()
                    if unconditional.grad is None:
                        raise RuntimeError(
                            "Null-text optimization did not receive embedding gradients"
                        )
                    _require_finite(unconditional.grad, "embedding gradients")
                    optimizer.step()
                    _require_finite(unconditional, "optimized embeddings")
                    if loss.detach().item() < tolerance:
                        break
                # Release the last backward graph before advancing or offloading.
                del prediction, guided, reconstructed, loss, optimizer
                with torch.no_grad():
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

    @torch.inference_mode(False)
    def invert(
        self, sample: Mapping[str, Any], context: InversionContext
    ) -> InversionArtifact:
        """Build pivots at guidance 1, then optimize using the run's guidance."""
        if not isinstance(sample, Mapping):
            raise TypeError("Null-text inversion sample must be a mapping")
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
        if context.config.runtime.cpu_offload == "sequential":
            raise ValueError(
                "Null-text inversion does not support sequential CPU offload; "
                "use runtime.cpu_offload='none' or 'model'"
            )
        if self._settings is None:
            self._settings = _NullTextSettings.load(_SETTINGS_PATH)

        # Import only during inversion so entry-point discovery stays lightweight.
        from diffusers import DDIMInverseScheduler

        editor = context.editor
        config = context.config
        inverse_scheduler = DDIMInverseScheduler.from_config(scheduler_config)
        inverse_scheduler.set_timesteps(
            config.sampling.num_inference_steps, device=editor.device
        )
        timesteps = editor.expected_timesteps
        if tuple(int(step) for step in inverse_scheduler.timesteps.tolist()) != tuple(
            reversed(timesteps)
        ):
            raise ValueError(
                "Null-text inverse timesteps must reverse the editor's schedule"
            )
        pipeline = editor.pipeline
        parameter_flags = [
            (parameter, parameter.requires_grad)
            for parameter in pipeline.unet.parameters()
        ]
        try:
            pipeline.maybe_free_model_hooks()
            for parameter, _ in parameter_flags:
                parameter.requires_grad_(False)
            latent = self._encode_source(image, context)
            pipeline.maybe_free_model_hooks()
            with torch.no_grad():
                embeddings = editor.encode_prompts(["", prompt]).detach()
                _require_finite(embeddings, "caption embeddings")
                unconditional, conditional = embeddings.chunk(2)
            pipeline.maybe_free_model_hooks()
            pivots = self._pivot_trajectory(
                latent, conditional, inverse_scheduler, context
            )
            # Whole-model offload keeps the UNet resident until explicit cleanup.
            # Do not call another pipeline component or offload it during backward.
            optimized = self._optimize(
                pivots, unconditional, conditional, self._settings, context
            )
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
                    _EMBEDDINGS_KEY: optimized,
                    _GUIDANCE_KEY: torch.full(
                        (len(timesteps),), config.sampling.guidance_scale,
                        dtype=torch.float64, device="cpu",
                    ),
                },
            )
            self.validate_replay(artifact, context)
            return artifact
        finally:
            for parameter, requires_grad in parameter_flags:
                parameter.requires_grad_(requires_grad)
            pipeline.maybe_free_model_hooks()
