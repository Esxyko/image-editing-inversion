"""Reconstruct and edit compatible artifacts with shared DDIM denoising."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ..artifacts import InversionArtifact
from ..config import ExperimentConfig
from ..inversion.hooks import BatchedDenoisingHook, DenoisingHook
from .denoising import _DenoisingExecutor
from .prompt_to_prompt import PromptToPrompt
from .result import EditResult
from .runtime import ModelRuntime

if TYPE_CHECKING:
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from PIL import Image


class Editor:
    """Load one SD 1.5 pipeline and edit batches of compatible artifacts."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._runtime = ModelRuntime(config)
        self.config = config
        self.device = self._runtime.device
        self.dtype = self._runtime.dtype
        self._denoising = _DenoisingExecutor(self._runtime)

    def reconfigure(self, config: ExperimentConfig) -> None:
        """Apply the next parameter file without loading model weights again."""
        self._runtime.reconfigure(config)
        self.config = config

    def inversion_cache_settings(self) -> dict[str, Any]:
        return self._runtime.inversion_cache_settings()

    @property
    def model_content_identity(self) -> Mapping[str, Any]:
        return self._runtime.model_content_identity

    @property
    def model_commit(self) -> str | None:
        return self._runtime.model_commit

    def component_content_identity(self, components: tuple[str, ...]) -> Mapping[str, Any]:
        return self._runtime.component_content_identity(components)

    @property
    def pipeline(self) -> StableDiffusionPipeline:
        return self._runtime.pipeline

    @property
    def scheduler(self) -> DDIMScheduler:
        return self.pipeline.scheduler

    @property
    def expected_timesteps(self) -> tuple[int, ...]:
        return self._runtime.expected_timesteps

    def timesteps_for_steps(self, num_inference_steps: int) -> tuple[int, ...]:
        """Derive an artifact's schedule without changing the active scheduler."""
        return self._runtime.timesteps_for_steps(num_inference_steps)

    @property
    def scheduler_config(self) -> dict[str, Any]:
        return self._runtime.scheduler_config

    def validate_artifact(self, artifact: InversionArtifact) -> None:
        artifact.validate_terminal_latent()
        artifact.validate_compatibility(
            self.config,
            dataset_ref=artifact.dataset_ref,
            dataset_fingerprint=artifact.dataset_fingerprint,
            timesteps=self.expected_timesteps,
            scheduler_config=self.scheduler_config,
        )

    def encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        """Encode captions with the shared CLIP token-limit validation."""
        return self._runtime.encode_prompts(prompts)

    def empty_prompt_embeddings(self, count: int) -> torch.Tensor:
        """Return ordinary detached empty-caption embeddings without re-encoding."""
        return self._runtime.empty_prompt_embeddings(count)

    def close(self) -> None:
        """Release the model owned by this editor."""
        self._runtime.close()

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Encode RGB sources in one VAE batch using the configured transfers."""
        if not images:
            raise ValueError("Image encoding requires at least one source image")
        model = self.config.model
        pixels = self.pipeline.image_processor.preprocess(
            [image.convert("RGB") for image in images],
            height=model.height,
            width=model.width,
        )
        pixels = self.to_device(pixels, dtype=self.dtype)
        latents = self.pipeline.vae.encode(pixels).latent_dist.mode()
        latents = latents * self.pipeline.vae.config.scaling_factor
        expected_shape = (len(images), 4, model.height // 8, model.width // 8)
        if tuple(latents.shape) != expected_shape:
            raise ValueError(
                f"Image encoding expected latent shape {expected_shape}, "
                f"got {tuple(latents.shape)}"
            )
        if not torch.isfinite(latents).all().item():
            raise ValueError("Image encoding produced non-finite latents")
        return latents.detach()

    def to_device(
        self, tensor: torch.Tensor, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Transfer model inputs; preserve dtype unless explicitly requested."""
        return self._runtime.to_device(tensor, dtype=dtype)

    @torch.inference_mode()
    def edit_batch(
        self,
        samples: Sequence[Mapping[str, Any]],
        artifacts: Sequence[InversionArtifact],
        hooks: Sequence[DenoisingHook | None] | None = None,
        *,
        prompt_to_prompt_class: type[PromptToPrompt] = PromptToPrompt,
        method_settings: Mapping[str, Any] | None = None,
        batch_hook: BatchedDenoisingHook | None = None,
    ) -> list[EditResult]:
        """Denoise up to ``runtime.batch_size`` source/target pairs together."""
        error: BaseException | None = None
        try:
            return self._edit_batch(
                samples, artifacts, hooks,
                prompt_to_prompt_class=prompt_to_prompt_class,
                method_settings=method_settings, batch_hook=batch_hook,
            )
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                self.pipeline.maybe_free_model_hooks()
            except BaseException as cleanup:
                if error is None:
                    raise
                error.add_note(f"Model offload cleanup failed: {cleanup}")

    def _edit_batch(
        self,
        samples: Sequence[Mapping[str, Any]],
        artifacts: Sequence[InversionArtifact],
        hooks: Sequence[DenoisingHook | None] | None,
        *,
        prompt_to_prompt_class: type[PromptToPrompt],
        method_settings: Mapping[str, Any] | None,
        batch_hook: BatchedDenoisingHook | None,
    ) -> list[EditResult]:
        """Execute editing inside the public method's model cleanup boundary."""
        batch_size = len(samples)
        if not batch_size or batch_size > self.config.runtime.batch_size:
            raise ValueError(
                f"edit_batch requires 1 to {self.config.runtime.batch_size} samples"
            )
        if len(artifacts) != batch_size:
            raise ValueError("Each dataset sample needs one inversion artifact")
        if batch_hook is not None:
            if not isinstance(batch_hook, BatchedDenoisingHook):
                raise TypeError("batch_hook must be BatchedDenoisingHook or None")
            if hooks is not None:
                raise ValueError("Pass either pair hooks or batch_hook, not both")
            if batch_hook.sample_uids != tuple(artifact.sample_uid for artifact in artifacts):
                raise ValueError("Batched hook UIDs must match the artifact order")
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
            if artifact.requires_denoising_hook and hook is None and batch_hook is None:
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
        negative_prompt_embeddings = self.empty_prompt_embeddings(2 * batch_size)
        base_latents = self.to_device(
            torch.cat([artifact.terminal_latent for artifact in artifacts]), dtype=self.dtype,
        )
        latents = self._denoising.denoise(
            [str(sample["uid"]) for sample in samples],
            torch.cat((base_latents, base_latents)), prompt_embeddings,
            negative_prompt_embeddings, hooks, batch_hook, controller,
        )

        decoded = self.pipeline.vae.decode(
            latents / self.pipeline.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        images = self.pipeline.image_processor.postprocess(
            decoded, output_type="pt"
        )
        if not torch.isfinite(images).all().item():
            raise ValueError("Image decoding produced non-finite RGB pixels")
        # Match Diffusers' PNG pixel quantization, transferring uint8 instead of floats.
        images = images.float().mul_(255).round_().clamp_(0, 255).to(dtype=torch.uint8)
        images = images.cpu().contiguous()
        return [
            EditResult(reconstructed=images[index], edited=images[batch_size + index])
            for index in range(batch_size)
        ]
