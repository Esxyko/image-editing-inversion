"""Shared experiment settings and hardware validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

from .._project import ProjectPaths, project_paths
from .validation import ConfigError, boolean, choice, integer, number, string



@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    revision: str
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("model_id", "revision"):
            object.__setattr__(self, name, string(getattr(self, name), f"model.{name}"))
        for name in ("width", "height"):
            integer(getattr(self, name), f"model.{name}", minimum=8)
        if self.width % 8 or self.height % 8:
            raise ConfigError("model.width and model.height must be divisible by 8")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    num_inference_steps: int
    eta: float
    guidance_scale: float
    seed: int

    def __post_init__(self) -> None:
        integer(self.num_inference_steps, "sampling.num_inference_steps", minimum=1)
        integer(self.seed, "sampling.seed")
        for name in ("eta", "guidance_scale"):
            object.__setattr__(self, name, number(getattr(self, name), f"sampling.{name}"))
        if self.eta > 1:
            raise ConfigError("sampling.eta must be <= 1 for the DDIM scheduler")


@dataclass(frozen=True, slots=True)
class PromptToPromptConfig:
    mode: str
    cross_replace_fraction: float
    self_replace_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", choice(self.mode, "prompt_to_prompt.mode", {"auto", "replace", "refine"}))
        for name in ("cross_replace_fraction", "self_replace_fraction"):
            value = number(getattr(self, name), f"prompt_to_prompt.{name}")
            if value > 1:
                raise ConfigError(f"prompt_to_prompt.{name} must be <= 1")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class MethodConfig:
    prompt_to_prompt: dict[str, Any]

    def __post_init__(self) -> None:
        settings = self.prompt_to_prompt
        if not isinstance(settings, dict) or any(not isinstance(key, str) for key in settings):
            raise ConfigError("Method Prompt-to-Prompt settings must be a mapping with string keys")
        try:
            object.__setattr__(self, "prompt_to_prompt", json.loads(json.dumps(settings, allow_nan=False)))
        except (TypeError, ValueError) as exc:
            raise ConfigError("Method Prompt-to-Prompt settings must contain JSON-compatible values") from exc


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
    inversion_batch_size: int = 1

    def __post_init__(self) -> None:
        device = string(self.device, "runtime.device")
        if not re.fullmatch(r"cuda(?::\d+)?|cpu|mps", device):
            raise ConfigError("runtime.device must be cpu, mps, cuda, or cuda:<index>")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "dtype", choice(self.dtype, "runtime.dtype", {"float16", "float32", "bfloat16"}))
        object.__setattr__(self, "cpu_offload", choice(self.cpu_offload, "runtime.cpu_offload", {"none", "model", "sequential"}))
        for name in ("batch_size", "inversion_batch_size", "attention_query_chunk_size"):
            integer(getattr(self, name), f"runtime.{name}", minimum=1)
        integer(self.num_workers, "runtime.num_workers")
        for name in ("vae_slicing", "vae_tiling", "pin_memory"):
            boolean(getattr(self, name), f"runtime.{name}")
        if device == "cpu" and self.dtype != "float32":
            raise ConfigError("runtime.device='cpu' requires runtime.dtype='float32'")
        if device == "mps" and self.dtype == "bfloat16":
            raise ConfigError("runtime.device='mps' does not support runtime.dtype='bfloat16'")
        if not device.startswith("cuda"):
            if self.cpu_offload != "none":
                raise ConfigError("runtime.cpu_offload requires a CUDA device")
            if self.pin_memory:
                raise ConfigError("runtime.pin_memory requires a CUDA device")

    def validate_hardware(self) -> None:
        """Check that the selected accelerator exists before loading model weights."""
        try:
            import torch
        except ImportError as exc:
            raise ConfigError("PyTorch is required to validate the configured device") from exc

        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ConfigError(f"runtime.device={self.device!r} requires available CUDA")
            configured_index = self.device.partition(":")[2]
            index = int(configured_index) if configured_index else torch.cuda.current_device()
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
    project: ProjectPaths = field(default_factory=project_paths, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, expected in (("model", ModelConfig), ("sampling", SamplingConfig),
                               ("prompt_to_prompt", PromptToPromptConfig), ("runtime", RuntimeConfig)):
            if not isinstance(getattr(self, name), expected):
                raise ConfigError(f"{name} must be {expected.__name__}")
        if not isinstance(self.methods, dict):
            raise ConfigError("methods must be a mapping")
        methods: dict[str, MethodConfig] = {}
        for key, value in self.methods.items():
            key = string(key, "methods method ID")
            if key in methods or not isinstance(value, MethodConfig):
                raise ConfigError("methods must contain unique IDs mapped to MethodConfig")
            methods[key] = value
        object.__setattr__(self, "methods", methods)

    def method_prompt_to_prompt(self, method_id: str) -> dict[str, Any]:
        """Return the extra P2P settings for one registered inversion method."""
        method = self.methods.get(method_id)
        return {} if method is None else deepcopy(method.prompt_to_prompt)

    def to_dict(self) -> dict[str, Any]:
        """Return the fully resolved settings for the run record."""
        return {
            name: asdict(getattr(self, name))
            for name in ("model", "sampling", "prompt_to_prompt", "runtime")
        } | {"methods": {key: asdict(value) for key, value in self.methods.items()}}

    def validate_hardware(self) -> None:
        self.runtime.validate_hardware()
