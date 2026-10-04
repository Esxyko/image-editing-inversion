"""Parse and validate experiment YAML configuration."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

import yaml

from .models import (
    ConfigError,
    ExperimentConfig,
    MethodConfig,
    ModelConfig,
    PromptToPromptConfig,
    RuntimeConfig,
    SamplingConfig,
)


def _section(data: dict[str, Any], key: str, required: set[str]) -> dict[str, Any]:
    value = data[key]
    if not isinstance(value, dict):
        raise ConfigError(f"Section {key!r} must be a YAML mapping")
    if any(not isinstance(name, str) for name in value):
        raise ConfigError(f"Section {key!r} setting names must be strings")
    missing = required - value.keys()
    extra = value.keys() - required
    if missing:
        raise ConfigError(f"Section {key!r} is missing: {', '.join(sorted(missing))}")
    if extra:
        raise ConfigError(f"Section {key!r} has unknown settings: {', '.join(sorted(extra))}")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a nonempty string")
    return value.strip()


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < minimum
    ):
        raise ConfigError(f"{name} must be a number >= {minimum:g}")
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false")
    return value


def _choice(value: Any, name: str, choices: set[str]) -> str:
    result = _string(value, name)
    if result not in choices:
        raise ConfigError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return result


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse and validate one YAML file; no pipeline defaults are hidden here."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except OSError as exc:
        raise ConfigError(f"Cannot read config at {config_path}: {exc}") from exc
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Invalid YAML at {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config at {config_path} must be a YAML mapping")
    if any(not isinstance(name, str) for name in data):
        raise ConfigError("Config section names must be strings")

    sections = {"model", "sampling", "prompt_to_prompt", "runtime"}
    missing = sections - data.keys()
    extra = data.keys() - sections - {"methods"}
    if missing:
        raise ConfigError(f"Config is missing sections: {', '.join(sorted(missing))}")
    if extra:
        raise ConfigError(f"Config has unknown sections: {', '.join(sorted(extra))}")

    model_data = _section(data, "model", {"model_id", "revision", "width", "height"})
    sampling_data = _section(
        data, "sampling", {"num_inference_steps", "eta", "guidance_scale", "seed"}
    )
    p2p_data = _section(
        data, "prompt_to_prompt", {"mode", "cross_replace_fraction", "self_replace_fraction"}
    )
    runtime_data = _section(
        data,
        "runtime",
        {
            "device",
            "dtype",
            "batch_size",
            "attention_query_chunk_size",
            "cpu_offload",
            "vae_slicing",
            "vae_tiling",
            "num_workers",
            "pin_memory",
        },
    )

    methods_data = data.get("methods", {})
    if not isinstance(methods_data, dict):
        raise ConfigError("Section 'methods' must be a YAML mapping")
    methods: dict[str, MethodConfig] = {}
    for method_id, method_data in methods_data.items():
        method_id = _string(method_id, "methods method ID")
        if method_id in methods:
            raise ConfigError(f"Duplicate method ID in methods section: {method_id!r}")
        if not isinstance(method_data, dict) or set(method_data) != {"prompt_to_prompt"}:
            raise ConfigError(
                f"methods.{method_id} must contain only a prompt_to_prompt mapping"
            )
        settings = method_data["prompt_to_prompt"]
        if not isinstance(settings, dict) or any(
            not isinstance(name, str) for name in settings
        ):
            raise ConfigError(
                f"methods.{method_id}.prompt_to_prompt must be a mapping with string keys"
            )
        try:
            normalized = json.loads(json.dumps(settings, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"methods.{method_id}.prompt_to_prompt must contain JSON-compatible values"
            ) from exc
        methods[method_id] = MethodConfig(prompt_to_prompt=normalized)

    model = ModelConfig(
        model_id=_string(model_data["model_id"], "model.model_id"),
        revision=_string(model_data["revision"], "model.revision"),
        width=_integer(model_data["width"], "model.width", minimum=8),
        height=_integer(model_data["height"], "model.height", minimum=8),
    )
    if model.width % 8 or model.height % 8:
        raise ConfigError("model.width and model.height must be divisible by 8")

    sampling = SamplingConfig(
        num_inference_steps=_integer(
            sampling_data["num_inference_steps"], "sampling.num_inference_steps", minimum=1
        ),
        eta=_number(sampling_data["eta"], "sampling.eta"),
        guidance_scale=_number(sampling_data["guidance_scale"], "sampling.guidance_scale"),
        seed=_integer(sampling_data["seed"], "sampling.seed"),
    )
    if sampling.eta > 1:
        raise ConfigError("sampling.eta must be <= 1 for the DDIM scheduler")

    prompt_to_prompt = PromptToPromptConfig(
        mode=_choice(p2p_data["mode"], "prompt_to_prompt.mode", {"auto", "replace", "refine"}),
        cross_replace_fraction=_number(
            p2p_data["cross_replace_fraction"], "prompt_to_prompt.cross_replace_fraction"
        ),
        self_replace_fraction=_number(
            p2p_data["self_replace_fraction"], "prompt_to_prompt.self_replace_fraction"
        ),
    )
    if prompt_to_prompt.cross_replace_fraction > 1:
        raise ConfigError("prompt_to_prompt.cross_replace_fraction must be <= 1")
    if prompt_to_prompt.self_replace_fraction > 1:
        raise ConfigError("prompt_to_prompt.self_replace_fraction must be <= 1")

    device = _string(runtime_data["device"], "runtime.device")
    if not re.fullmatch(r"cuda(?::\d+)?|cpu|mps", device):
        raise ConfigError("runtime.device must be cpu, mps, cuda, or cuda:<index>")
    runtime = RuntimeConfig(
        device=device,
        dtype=_choice(runtime_data["dtype"], "runtime.dtype", {"float16", "float32", "bfloat16"}),
        batch_size=_integer(runtime_data["batch_size"], "runtime.batch_size", minimum=1),
        attention_query_chunk_size=_integer(
            runtime_data["attention_query_chunk_size"],
            "runtime.attention_query_chunk_size",
            minimum=1,
        ),
        cpu_offload=_choice(
            runtime_data["cpu_offload"], "runtime.cpu_offload", {"none", "model", "sequential"}
        ),
        vae_slicing=_boolean(runtime_data["vae_slicing"], "runtime.vae_slicing"),
        vae_tiling=_boolean(runtime_data["vae_tiling"], "runtime.vae_tiling"),
        num_workers=_integer(runtime_data["num_workers"], "runtime.num_workers"),
        pin_memory=_boolean(runtime_data["pin_memory"], "runtime.pin_memory"),
    )
    if runtime.device == "cpu" and runtime.dtype != "float32":
        raise ConfigError("runtime.device='cpu' requires runtime.dtype='float32'")
    if runtime.device == "mps" and runtime.dtype == "bfloat16":
        raise ConfigError("runtime.device='mps' does not support runtime.dtype='bfloat16'")
    if not runtime.device.startswith("cuda"):
        if runtime.cpu_offload != "none":
            raise ConfigError("runtime.cpu_offload requires a CUDA device")
        if runtime.pin_memory:
            raise ConfigError("runtime.pin_memory requires a CUDA device")

    return ExperimentConfig(
        model=model,
        sampling=sampling,
        prompt_to_prompt=prompt_to_prompt,
        runtime=runtime,
        methods=methods,
    )
