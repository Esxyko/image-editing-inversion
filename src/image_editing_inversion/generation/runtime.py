"""Load and configure the shared Stable Diffusion model runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch

from ..config import ExperimentConfig
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
        self.pipeline = pipeline
        self.model_commit = commit
        self.model_content_identity = ModelContentIdentity(model_path).describe(pipeline)

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
