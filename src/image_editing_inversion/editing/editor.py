"""Reconstruct and edit compatible artifacts with shared DDIM denoising."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
import torch

from ..artifacts import InversionArtifact, normalize_scheduler_config
from ..config import ExperimentConfig
from ..methods.hooks import DenoisingHook, DenoisingStepState
from .processor import _PromptToPromptProcessor
from .prompt_to_prompt import PromptToPrompt
from .result import EditResult
from .runtime import ModelRuntime

if TYPE_CHECKING:
    from diffusers import DDIMScheduler, StableDiffusionPipeline


class Editor:
    """Load one SD 1.5 pipeline and edit batches of compatible artifacts."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._runtime = ModelRuntime(config)
        self.config = config
        self.device = self._runtime.device
        self.dtype = self._runtime.dtype
        self._pipeline = self._runtime.pipeline

    def reconfigure(self, config: ExperimentConfig) -> None:
        """Apply the next parameter file without loading model weights again."""
        self._runtime.reconfigure(config)
        self.config = config

    def inversion_cache_settings(self) -> dict[str, Any]:
        return self._runtime.inversion_cache_settings()

    @property
    def pipeline(self) -> StableDiffusionPipeline:
        return self._pipeline

    @property
    def scheduler(self) -> DDIMScheduler:
        return self._pipeline.scheduler

    @property
    def expected_timesteps(self) -> tuple[int, ...]:
        return tuple(int(step) for step in self.scheduler.timesteps.tolist())

    @property
    def scheduler_config(self) -> dict[str, Any]:
        return normalize_scheduler_config(self.scheduler.config)

    def validate_artifact(self, artifact: InversionArtifact) -> None:
        artifact.validate_compatibility(
            self.config,
            dataset_ref=artifact.dataset_ref,
            dataset_fingerprint=artifact.dataset_fingerprint,
            timesteps=self.expected_timesteps,
            scheduler_config=self.scheduler_config,
        )

    def encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        """Encode captions with the shared CLIP token-limit validation."""
        tokenizer = self.pipeline.tokenizer
        tokens = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=False,
            return_tensors="pt",
        )
        if tokens.input_ids.shape[-1] != tokenizer.model_max_length:
            raise ValueError("Caption exceeds the model's CLIP token limit")
        return self.pipeline.text_encoder(
            tokens.input_ids.to(self.device)
        )[0]

    @staticmethod
    def _check_hook_state(
        updated: DenoisingStepState,
        original: DenoisingStepState,
    ) -> None:
        if not isinstance(updated, DenoisingStepState):
            raise TypeError("Denoising hook must return DenoisingStepState")
        for field_name in (
            "latents", "prompt_embeddings", "negative_prompt_embeddings"
        ):
            actual = getattr(updated, field_name)
            expected = getattr(original, field_name)
            if (
                not isinstance(actual, torch.Tensor)
                or actual.shape != expected.shape
                or actual.device != expected.device
                or actual.dtype != expected.dtype
            ):
                raise ValueError(
                    f"Denoising hook returned invalid {field_name}: expected "
                    f"shape {tuple(expected.shape)}, device {expected.device}, "
                    f"and dtype {expected.dtype}"
                )
        if original.variance_noise is None:
            if updated.variance_noise is not None:
                raise ValueError(
                    "Denoising hook may supply variance_noise only before a "
                    "DDIM step with eta > 0"
                )
        else:
            actual = updated.variance_noise
            expected = original.variance_noise
            if (
                not isinstance(actual, torch.Tensor)
                or actual.shape != expected.shape
                or actual.device != expected.device
                or actual.dtype != expected.dtype
            ):
                raise ValueError(
                    "Denoising hook returned invalid variance_noise: expected "
                    f"shape {tuple(expected.shape)}, device {expected.device}, "
                    f"and dtype {expected.dtype}"
                )

    def _apply_hooks(
        self,
        hooks: Sequence[DenoisingHook | None],
        stage: str,
        step_index: int,
        timestep: int,
        latents: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        negative_prompt_embeddings: torch.Tensor,
        variance_noise: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        if not any(hook is not None for hook in hooks):
            return (
                latents, prompt_embeddings, negative_prompt_embeddings, variance_noise
            )
        batch_size = len(hooks)
        changed: list[DenoisingStepState] = []
        for index, hook in enumerate(hooks):
            pair = DenoisingStepState(
                latents=torch.cat((latents[index:index + 1], latents[batch_size + index:batch_size + index + 1])),
                prompt_embeddings=torch.cat((prompt_embeddings[index:index + 1], prompt_embeddings[batch_size + index:batch_size + index + 1])),
                negative_prompt_embeddings=torch.cat((negative_prompt_embeddings[index:index + 1], negative_prompt_embeddings[batch_size + index:batch_size + index + 1])),
                variance_noise=(
                    None if variance_noise is None else torch.cat(
                        (
                            variance_noise[index:index + 1],
                            variance_noise[batch_size + index:batch_size + index + 1],
                        )
                    )
                ),
            )
            if hook is None:
                changed.append(pair)
                continue
            result = (
                hook.before_step(step_index, timestep, pair)
                if stage == "before"
                else hook.after_step(step_index, timestep, pair)
            )
            self._check_hook_state(result, pair)
            changed.append(result)
        def reassemble(field_name: str) -> torch.Tensor:
            return torch.cat(
                [getattr(item, field_name)[:1] for item in changed]
                + [getattr(item, field_name)[1:] for item in changed]
            )
        return (
            reassemble("latents"),
            reassemble("prompt_embeddings"),
            reassemble("negative_prompt_embeddings"),
            None if variance_noise is None else reassemble("variance_noise"),
        )

    @staticmethod
    def _noise_generator(seed: int, uid: str, device: torch.device) -> torch.Generator:
        digest = hashlib.sha256(uid.encode("utf-8")).digest()
        pair_seed = (seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1)
        return torch.Generator(device=device).manual_seed(pair_seed)

    def _prepare_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda" and latent.device.type == "cpu":
            if self.config.runtime.pin_memory and not latent.is_pinned():
                latent = latent.pin_memory()
            return latent.to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=self.config.runtime.pin_memory,
            )
        return latent.to(device=self.device, dtype=self.dtype)

    def _install_processors(
        self, controller: PromptToPrompt
    ) -> dict[str, Any]:
        unet = self.pipeline.unet
        original = dict(unet.attn_processors)
        replacement = {}
        for name, processor in original.items():
            if not isinstance(processor, (AttnProcessor, AttnProcessor2_0)):
                raise ValueError(
                    f"Prompt-to-Prompt cannot control unsupported UNet attention "
                    f"processor {name}: {type(processor).__name__}"
                )
            if name.endswith(".attn1.processor"):
                is_cross = False
            elif name.endswith(".attn2.processor"):
                is_cross = True
            else:
                raise ValueError(f"Unexpected SD 1.5 UNet attention layer: {name}")
            replacement[name] = _PromptToPromptProcessor(
                processor,
                controller,
                is_cross,
                self.config.runtime.attention_query_chunk_size,
                name,
            )
        if not replacement or not any(
            name.endswith(".attn2.processor") for name in replacement
        ):
            raise ValueError("The loaded UNet has no controllable cross-attention layers")
        try:
            unet.set_attn_processor(dict(replacement))
        except Exception:
            unet.set_attn_processor(dict(original))
            raise
        return original

    @torch.inference_mode()
    def edit_batch(
        self,
        samples: Sequence[Mapping[str, Any]],
        artifacts: Sequence[InversionArtifact],
        hooks: Sequence[DenoisingHook | None] | None = None,
        *,
        prompt_to_prompt_class: type[PromptToPrompt] = PromptToPrompt,
        method_settings: Mapping[str, Any] | None = None,
    ) -> list[EditResult]:
        """Denoise up to ``runtime.batch_size`` source/target pairs together."""
        batch_size = len(samples)
        if not batch_size or batch_size > self.config.runtime.batch_size:
            raise ValueError(
                f"edit_batch requires 1 to {self.config.runtime.batch_size} samples"
            )
        if len(artifacts) != batch_size:
            raise ValueError("Each dataset sample needs one inversion artifact")
        if hooks is None:
            hooks = [None] * batch_size
        if len(hooks) != batch_size:
            raise ValueError("Each dataset sample needs one hook entry")
        if not isinstance(prompt_to_prompt_class, type) or not issubclass(
            prompt_to_prompt_class, PromptToPrompt
        ):
            raise TypeError("prompt_to_prompt_class must subclass PromptToPrompt")
        validated_settings = prompt_to_prompt_class.validate_method_settings(
            {} if method_settings is None else method_settings
        )
        if not isinstance(validated_settings, Mapping):
            raise TypeError("Prompt-to-Prompt settings validator must return a mapping")
        for sample, artifact, hook in zip(samples, artifacts, hooks):
            if sample.get("uid") != artifact.sample_uid:
                raise ValueError(
                    f"Artifact UID {artifact.sample_uid!r} does not match dataset sample"
                )
            self.validate_artifact(artifact)
            if artifact.requires_denoising_hook and hook is None:
                raise ValueError(
                    f"Artifact {artifact.sample_uid!r} has method-specific per-step "
                    "state but no denoising hook was provided"
                )
            if hook is not None and not isinstance(hook, DenoisingHook):
                raise TypeError("hooks must contain DenoisingHook instances or None")

        tokenizer = self.pipeline.tokenizer
        max_length = tokenizer.model_max_length
        source_prompts: list[str] = []
        target_prompts: list[str] = []
        for sample in samples:
            source = sample.get("source_prompt")
            target = sample.get("target_prompt")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Dataset source_prompt must be a nonempty string")
            if not isinstance(target, str) or not target.strip():
                raise ValueError("Dataset target_prompt must be a nonempty string")
            source_prompts.append(source)
            target_prompts.append(target)
        controller = prompt_to_prompt_class(
            self.config.prompt_to_prompt,
            tokenizer,
            source_prompts,
            target_prompts,
            artifacts,
            num_steps=len(self.expected_timesteps),
            max_length=max_length,
            device=self.device,
            settings=validated_settings,
        )
        prompts = source_prompts + target_prompts
        prompt_embeddings = self.encode_prompts(prompts)
        negative_prompt_embeddings = self.encode_prompts([""] * (2 * batch_size))
        base_latents = torch.cat(
            [self._prepare_latent(artifact.terminal_latent) for artifact in artifacts]
        )
        latents = torch.cat((base_latents, base_latents.clone()), dim=0)
        generators = [
            self._noise_generator(self.config.sampling.seed, str(sample["uid"]), self.device)
            for sample in samples
        ]

        original_processors = self._install_processors(controller)
        try:
            for step_index, timestep in enumerate(self.scheduler.timesteps):
                step_number = int(timestep.item())
                controller.step_index = step_index
                variance_noise = None
                if self.config.sampling.eta > 0:
                    source_noise = torch.stack(
                        [
                            torch.randn(
                                latents[index].shape,
                                generator=generator,
                                device=self.device,
                                dtype=self.dtype,
                            )
                            for index, generator in enumerate(generators)
                        ]
                    )
                    variance_noise = torch.cat(
                        (source_noise, source_noise), dim=0
                    )
                latents, prompt_embeddings, negative_prompt_embeddings, variance_noise = self._apply_hooks(
                    hooks, "before", step_index, step_number,
                    latents, prompt_embeddings, negative_prompt_embeddings,
                    variance_noise,
                )
                model_input = torch.cat((latents, latents), dim=0)
                model_input = self.scheduler.scale_model_input(model_input, timestep)
                embeddings = torch.cat(
                    (negative_prompt_embeddings, prompt_embeddings), dim=0
                )
                prediction = self.pipeline.unet(
                    model_input, timestep, encoder_hidden_states=embeddings
                ).sample
                unconditional, conditional = prediction.chunk(2)
                guided = unconditional + self.config.sampling.guidance_scale * (
                    conditional - unconditional
                )
                step_kwargs: dict[str, Any] = {"eta": self.config.sampling.eta}
                if variance_noise is not None:
                    step_kwargs["variance_noise"] = variance_noise
                latents = self.scheduler.step(
                    guided, timestep, latents, **step_kwargs
                ).prev_sample
                latents, prompt_embeddings, negative_prompt_embeddings, _ = self._apply_hooks(
                    hooks, "after", step_index, step_number,
                    latents, prompt_embeddings, negative_prompt_embeddings,
                    None,
                )
        finally:
            self.pipeline.unet.set_attn_processor(dict(original_processors))

        decoded = self.pipeline.vae.decode(
            latents / self.pipeline.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        images = self.pipeline.image_processor.postprocess(
            decoded, output_type="pil"
        )
        return [
            EditResult(reconstructed=images[index], edited=images[batch_size + index])
            for index in range(batch_size)
        ]
