"""Shared names for method and pipeline-parameter artifact directories."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def method_directory_name(method_id: str) -> str:
    """Keep the method ID itself as a portable directory name."""
    reserved = {"con", "prn", "aux", "nul"} | {
        f"{prefix}{index}" for prefix in ("com", "lpt") for index in range(1, 10)
    }
    if (
        not isinstance(method_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", method_id) is None
        or method_id.endswith(".")
        or method_id.split(".")[0] in reserved
    ):
        raise ValueError("Method IDs must be lowercase portable directory names")
    return method_id


def pipeline_group_name(parameters: Mapping[str, Any]) -> str:
    """Name a parameter combination without rounding its guidance value."""
    if set(parameters) != {"num_inference_steps", "guidance_scale"}:
        raise ValueError("Artifact grouping requires num_inference_steps and guidance_scale")
    steps = parameters["num_inference_steps"]
    guidance = parameters["guidance_scale"]
    if type(steps) is not int or steps < 1:
        raise ValueError("Artifact grouping num_inference_steps must be a positive integer")
    if type(guidance) not in (int, float) or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Artifact grouping guidance_scale must be finite and nonnegative")
    label = repr(float(guidance) if guidance else 0.0)
    if label.endswith(".0"):
        label = label[:-2]
    return f"steps-{steps}_guidance-{label}"


def pipeline_group_parameters(name: str) -> dict[str, Any]:
    """Read steps and guidance from a canonical group directory name."""
    match = re.fullmatch(r"steps-([1-9][0-9]*)_guidance-(.+)", name)
    if match is None:
        raise ValueError("Pipeline group must use steps-N_guidance-G format")
    parameters = {
        "num_inference_steps": int(match[1]), "guidance_scale": float(match[2]),
    }
    if pipeline_group_name(parameters) != name:
        raise ValueError("Pipeline group must use canonical steps and guidance values")
    return parameters


def is_pipeline_group(name: str) -> bool:
    """Recognize canonical readable names; legacy digest groups are excluded."""
    try:
        pipeline_group_parameters(name)
        return True
    except (ValueError, OverflowError):
        return False


@dataclass(frozen=True, slots=True)
class ArtifactGroup:
    """Validated method and sampling parameters of a published collection."""

    method_id: str
    num_inference_steps: int
    guidance_scale: float

    @classmethod
    def from_path(cls, path: Path) -> ArtifactGroup:
        if path.name != "artifacts.safetensors":
            raise ValueError("Published collections must use artifacts.safetensors")
        method_id = method_directory_name(path.parent.parent.name)
        return cls(method_id=method_id, **pipeline_group_parameters(path.parent.name))

    @property
    def parameters(self) -> dict[str, Any]:
        return {"num_inference_steps": self.num_inference_steps, "guidance_scale": self.guidance_scale}
