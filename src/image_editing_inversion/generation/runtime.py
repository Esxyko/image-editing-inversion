"""Load and configure the shared Stable Diffusion model runtime."""

from __future__ import annotations

import hashlib
import platform
from typing import Any

from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch

from ..config import ExperimentConfig


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
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.set_timesteps(
            config.sampling.num_inference_steps, device=self.device
        )
        if config.runtime.vae_slicing:
            pipeline.enable_vae_slicing()
        if config.runtime.vae_tiling:
            pipeline.enable_vae_tiling()
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
        """Describe execution settings that can change inversion numerics."""
        settings: dict[str, Any] = {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "vae_slicing": self.config.runtime.vae_slicing,
            "vae_tiling": self.config.runtime.vae_tiling,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch_build_sha256": hashlib.sha256(
                torch.__config__.show().encode("utf-8")
            ).hexdigest(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
        if self.device.type == "cuda":
            settings["cuda"] = {
                "device_name": torch.cuda.get_device_name(self.device),
                "capability": list(torch.cuda.get_device_capability(self.device)),
                "version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
                "memory_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math_sdp": torch.backends.cuda.math_sdp_enabled(),
            }
        elif self.device.type == "cpu":
            settings["num_threads"] = torch.get_num_threads()
        return settings
