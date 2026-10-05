"""Semantic configuration checks shared by Python models and YAML loading."""

import math
from typing import Any


class ConfigError(ValueError):
    """A configuration value is missing, malformed, or unsupported."""


def string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a nonempty string")
    return value.strip()


def integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(float(value)) and value >= minimum)
    except OverflowError:
        valid = False
    if not valid:
        raise ConfigError(f"{name} must be a finite number >= {minimum:g}")
    return float(value)


def boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false")
    return value


def choice(value: Any, name: str, choices: set[str]) -> str:
    result = string(value, name)
    if result not in choices:
        raise ConfigError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return result
