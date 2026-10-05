"""Load one shared configuration snapshot and resolve pipeline parameter files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .._project import ProjectPaths, project_paths
from .models import (
    ConfigError, ExperimentConfig, MethodConfig, ModelConfig, PromptToPromptConfig,
    RuntimeConfig, SamplingConfig,
)
from .validation import string


def _section(
    data: dict[str, Any], key: str, required: set[str], *, optional: set[str] | None = None,
) -> dict[str, Any]:
    value = data[key]
    if not isinstance(value, dict):
        raise ConfigError(f"Section {key!r} must be a YAML mapping")
    if any(not isinstance(name, str) for name in value):
        raise ConfigError(f"Section {key!r} setting names must be strings")
    missing = required - value.keys()
    extra = value.keys() - required - (optional or set())
    if missing:
        raise ConfigError(f"Section {key!r} is missing: {', '.join(sorted(missing))}")
    if extra:
        raise ConfigError(f"Section {key!r} has unknown settings: {', '.join(sorted(extra))}")
    return value


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot load configuration at {path}: {exc}") from exc
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise ConfigError(f"Configuration at {path} must be a mapping with string keys")
    return data


def _pipeline_path(filename: str | Path, paths: ProjectPaths) -> Path:
    directory = paths.pipeline_h_params.resolve()
    requested = Path(filename)
    if requested.is_absolute():
        path = requested.resolve()
    elif len(requested.parts) == 1:
        path = (directory / requested).resolve()
    else:
        path = (paths.root / requested).resolve()
    if path.parent != directory or path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigError("Pipeline parameters must be a YAML file inside pipeline_h_params/")
    if not path.is_file():
        raise ConfigError(f"Pipeline parameter file does not exist: {path}")
    return path


def discover_pipeline_h_params(filename: str | Path | None = None) -> tuple[Path, ...]:
    """Select one file or all top-level parameter files in filename order."""
    paths = project_paths()
    if filename is not None:
        return (_pipeline_path(filename, paths),)
    directory = paths.pipeline_h_params
    if not directory.is_dir():
        raise ConfigError(f"Pipeline parameter directory does not exist: {directory}")
    files = sorted(
        (path for path in directory.iterdir()
         if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}),
        key=lambda path: path.name,
    )
    if not files:
        raise ConfigError(f"No YAML parameter files found in {directory}")
    return tuple(_pipeline_path(path, paths) for path in files)


class ConfigurationLoader:
    """Internal owner of the shared YAML snapshot for one invocation."""

    def __init__(self) -> None:
        self.paths = project_paths()
        data = _read_mapping(self.paths.config)
        required = {"model", "sampling", "prompt_to_prompt", "runtime"}
        missing = required - data.keys()
        extra = data.keys() - required - {"methods"}
        if missing:
            raise ConfigError(f"Config is missing sections: {', '.join(sorted(missing))}")
        if extra:
            raise ConfigError(f"Config has unknown sections: {', '.join(sorted(extra))}")
        self._model = ModelConfig(**_section(data, "model", {"model_id", "revision", "width", "height"}))
        self._sampling = _section(data, "sampling", {"eta", "seed"})
        self._p2p = _section(data, "prompt_to_prompt", {"mode"})
        self._runtime = RuntimeConfig(**_section(
            data, "runtime",
            {"device", "dtype", "batch_size", "attention_query_chunk_size", "cpu_offload",
             "vae_slicing", "vae_tiling", "num_workers", "pin_memory"},
            optional={"inversion_batch_size"},
        ))
        methods = data.get("methods", {})
        if not isinstance(methods, dict):
            raise ConfigError("Section 'methods' must be a YAML mapping")
        self._methods: dict[str, MethodConfig] = {}
        for key, values in methods.items():
            method_id = string(key, "methods method ID")
            if method_id in self._methods:
                raise ConfigError(f"Duplicate method ID in methods section: {method_id!r}")
            if not isinstance(values, dict) or set(values) != {"prompt_to_prompt"}:
                raise ConfigError(f"methods.{method_id} must contain only a prompt_to_prompt mapping")
            self._methods[method_id] = MethodConfig(**values)

    def load(self, filename: str | Path) -> ExperimentConfig:
        path = _pipeline_path(filename, self.paths)
        data = _read_mapping(path)
        if set(data) != {"sampling", "prompt_to_prompt"}:
            raise ConfigError(f"Pipeline parameters at {path} must contain only sampling and prompt_to_prompt sections")
        sampling = _section(data, "sampling", {"num_inference_steps", "guidance_scale"})
        p2p = _section(data, "prompt_to_prompt", {"cross_replace_fraction", "self_replace_fraction"})
        return ExperimentConfig(
            model=self._model,
            sampling=SamplingConfig(**self._sampling, **sampling),
            prompt_to_prompt=PromptToPromptConfig(**self._p2p, **p2p),
            runtime=self._runtime,
            methods={key: MethodConfig(value.prompt_to_prompt) for key, value in self._methods.items()},
        )


def load_config(pipeline_h_params_file: str | Path) -> ExperimentConfig:
    """Combine the source-relative config.yaml with one parameter file."""
    return ConfigurationLoader().load(pipeline_h_params_file)
