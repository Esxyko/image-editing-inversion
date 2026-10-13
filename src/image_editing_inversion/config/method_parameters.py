"""Shared CFG settings for method-owned YAML parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .loader import _read_mapping
from .validation import ConfigError, number


@dataclass(frozen=True, slots=True)
class MethodGuidanceSettings:
    inversion_guidance_scale: float = 7.5
    editing_guidance_scale: float = 7.5

    def __post_init__(self) -> None:
        for name in ("inversion_guidance_scale", "editing_guidance_scale"):
            object.__setattr__(self, name, number(getattr(self, name), f"method_h_params.{name}"))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> MethodGuidanceSettings:
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})

    @classmethod
    def load(cls, path: Path) -> MethodGuidanceSettings:
        values = _read_mapping(path)
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ConfigError(f"Unknown method hyperparameters at {path}: {', '.join(sorted(unknown))}")
        return cls.from_mapping(values)
