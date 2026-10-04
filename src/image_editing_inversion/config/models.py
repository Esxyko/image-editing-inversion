"""Shared experiment settings and hardware validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class ConfigError(ValueError):
    """A configuration value is missing, malformed, or unsupported."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    revision: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    num_inference_steps: int
    eta: float
    guidance_scale: float
    seed: int


@dataclass(frozen=True, slots=True)
class PromptToPromptConfig:
    mode: str
    cross_replace_fraction: float
    self_replace_fraction: float


@dataclass(frozen=True, slots=True)
class MethodConfig:
    prompt_to_prompt: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str
    dtype: str
    batch_size: int
    attention_query_chunk_size: int
    cpu_offload: str
    vae_slicing: bool
    vae_tiling: bool
    num_workers: int
    pin_memory: bool

    def validate_hardware(self) -> None:
        """Check that the selected accelerator exists before loading model weights."""
        try:
            import torch
        except ImportError as exc:
            raise ConfigError("PyTorch is required to validate the configured device") from exc

        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ConfigError(f"runtime.device={self.device!r} requires available CUDA")
            index = int(self.device.partition(":")[2] or "0")
            if index >= torch.cuda.device_count():
                raise ConfigError(
                    f"runtime.device={self.device!r} is unavailable; "
                    f"found {torch.cuda.device_count()} CUDA device(s)"
                )
            if self.dtype == "bfloat16" and torch.cuda.get_device_capability(index) < (8, 0):
                raise ConfigError(
                    f"runtime.dtype='bfloat16' is unsupported on {self.device}"
                )
        elif self.device == "mps" and not torch.backends.mps.is_available():
            raise ConfigError("runtime.device='mps' requires available Apple Metal")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: ModelConfig
    sampling: SamplingConfig
    prompt_to_prompt: PromptToPromptConfig
    runtime: RuntimeConfig
    methods: dict[str, MethodConfig] = field(default_factory=dict)

    def method_prompt_to_prompt(self, method_id: str) -> dict[str, Any]:
        """Return the extra P2P settings for one registered inversion method."""
        method = self.methods.get(method_id)
        return {} if method is None else dict(method.prompt_to_prompt)

    def to_dict(self) -> dict[str, Any]:
        """Return the fully resolved settings for the run record."""
        return asdict(self)

    def validate_hardware(self) -> None:
        self.runtime.validate_hardware()
