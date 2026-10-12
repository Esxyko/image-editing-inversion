"""Load and configure the shared Stable Diffusion model runtime."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch

from ..config import ExperimentConfig
from ..artifacts import normalize_scheduler_config
from .identity import ModelContentIdentity


class ModelRuntime:
    """Own pipeline loading, accelerator selection, and memory settings."""

    def __init__(self, config: ExperimentConfig) -> None:
        config.validate_hardware()
        self.config = config
        self.device = torch.device(config.runtime.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[config.runtime.dtype]
        model_path = config.project.model_path(config.model.model_id)
        commit = None
        if not model_path.is_dir():
            model_path = Path(StableDiffusionPipeline.download(
                config.model.model_id, revision=config.model.revision,
                safety_checker=None,
            ))
            commit = model_path.name
        model_path = model_path.resolve()
        pipeline = StableDiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.set_timesteps(
            config.sampling.num_inference_steps, device=self.device
        )
        if config.runtime.vae_slicing:
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
            else:
                pipeline.vae.enable_slicing()
        if config.runtime.vae_tiling:
            if hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling()
            else:
                pipeline.vae.enable_tiling()
        if config.runtime.cpu_offload == "none":
            pipeline.to(self.device)
        else:
            gpu_id = self.device.index or 0
            if config.runtime.cpu_offload == "model":
                pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
            else:
                pipeline.enable_sequential_cpu_offload(gpu_id=gpu_id)
        pipeline.unet.eval()
        pipeline.text_encoder.eval()
        pipeline.vae.eval()
        self._pipeline: StableDiffusionPipeline | None = pipeline
        self._empty_embeddings: torch.Tensor | None = None
        self._schedules: dict[int, tuple[int, ...]] = {}
        self._cache_schedule()
        self.model_commit = commit
        self.model_content_identity = ModelContentIdentity(model_path).describe(pipeline)

    @property
    def pipeline(self) -> StableDiffusionPipeline:
        if self._pipeline is None:
            raise RuntimeError("The model runtime is closed")
        return self._pipeline

    def _cache_schedule(self) -> None:
        scheduler = self.pipeline.scheduler
        self._expected_timesteps = tuple(int(step) for step in scheduler.timesteps.tolist())
        self._scheduler_config = normalize_scheduler_config(scheduler.config)
        self._schedules.clear()
        self._schedules[self.config.sampling.num_inference_steps] = self._expected_timesteps

    @property
    def expected_timesteps(self) -> tuple[int, ...]:
        return self._expected_timesteps

    @property
    def scheduler_config(self) -> dict[str, Any]:
        return deepcopy(self._scheduler_config)

    def timesteps_for_steps(self, steps: int) -> tuple[int, ...]:
        if steps not in self._schedules:
            scheduler = type(self.pipeline.scheduler).from_config(self.pipeline.scheduler.config)
            scheduler.set_timesteps(steps, device="cpu")
            self._schedules[steps] = tuple(int(step) for step in scheduler.timesteps.tolist())
        return self._schedules[steps]

    def to_device(
        self, tensor: torch.Tensor, *, dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Apply this runtime's pinning and device-transfer settings."""
        if self.device.type == "cuda" and tensor.device.type == "cpu":
            if self.config.runtime.pin_memory and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            return tensor.to(
                device=self.device, dtype=dtype,
                non_blocking=self.config.runtime.pin_memory,
            )
        return tensor.to(device=self.device, dtype=dtype)

    def encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        tokenizer = self.pipeline.tokenizer
        tokens = tokenizer(
            prompts, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=False, return_tensors="pt",
        )
        if tokens.input_ids.shape[-1] != tokenizer.model_max_length:
            raise ValueError("Caption exceeds the model's CLIP token limit")
        return self.pipeline.text_encoder(self.to_device(tokens.input_ids))[0]

    @torch.inference_mode(False)
    @torch.no_grad()
    def empty_prompt_embeddings(self, count: int) -> torch.Tensor:
        """Cache an ordinary detached tensor suitable for later Null-text autograd."""
        if count < 1:
            raise ValueError("Empty prompt encoding requires a positive batch size")
        if self._empty_embeddings is None:
            self._empty_embeddings = self.encode_prompts([""]).detach().cpu().contiguous()
        return self.to_device(self._empty_embeddings).expand(count, -1, -1).contiguous()

    def close(self) -> None:
        """Release invocation-owned references even if offload cleanup fails."""
        error: BaseException | None = None
        try:
            if self._pipeline is not None:
                for cleanup in (self._pipeline.maybe_free_model_hooks, self._pipeline.remove_all_hooks):
                    try:
                        cleanup()
                    except BaseException as exc:
                        if error is None:
                            error = exc
                        else:
                            error.add_note(f"Model hook removal failed: {exc}")
            if error is not None:
                raise error
        finally:
            self._empty_embeddings = None
            self._schedules.clear()
            self._pipeline = None

    def component_content_identity(self, components: tuple[str, ...]) -> dict[str, Any]:
        return ModelContentIdentity.for_components(self.model_content_identity, components)

    def reconfigure(self, config: ExperimentConfig) -> None:
        """Change sampling/edit settings while retaining the loaded model."""
        if config.model != self.config.model or config.runtime != self.config.runtime:
            raise ValueError("A shared model runtime cannot change model or runtime settings")
        self.pipeline.maybe_free_model_hooks()
        self.pipeline.scheduler.set_timesteps(
            config.sampling.num_inference_steps, device=self.device
        )
        self.config = config
        self._cache_schedule()

    def inversion_cache_settings(self) -> dict[str, Any]:
        """Keep precision and VAE tiling; tolerate minor execution differences."""
        settings: dict[str, Any] = {
            "dtype": str(self.dtype),
            "vae_tiling": self.config.runtime.vae_tiling,
        }
        if self.config.runtime.vae_tiling:
            settings["tiling"] = {
                name: getattr(self.pipeline.vae, name) for name in (
                    "tile_sample_min_size", "tile_latent_min_size", "tile_overlap_factor",
                )
            }
        return json.loads(json.dumps(settings, allow_nan=False))
