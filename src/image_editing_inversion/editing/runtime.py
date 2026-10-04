"""Load and configure the shared Stable Diffusion model runtime."""

from __future__ import annotations

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
