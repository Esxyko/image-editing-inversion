"""Per-image inversion state and compatibility validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

import torch


_STATE_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


class ArtifactCompatibilityError(ValueError):
    """A valid artifact cannot be replayed with the selected configuration."""


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, name)


def _json_object(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    try:
        # This also turns tuples and other JSON-compatible sequence values into
        # their stored representation before compatibility comparisons.
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only JSON-compatible values") from exc


def normalize_scheduler_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep scheduler settings without dependency-version identity."""
    config = _json_object(value, "scheduler_config")
    for name in ("_diffusers_version", "_use_default_values", "_class_name", "_name_or_path"):
        config.pop(name, None)
    if not config.get("clip_sample", True):
        config.pop("clip_sample_range", None)
    if not config.get("thresholding", False):
        for name in ("dynamic_thresholding_ratio", "sample_max_value"):
            config.pop(name, None)
    if config.get("trained_betas") is not None:
        for name in ("beta_start", "beta_end", "beta_schedule"):
            config.pop(name, None)
    return config


def _timesteps_tuple(values: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("timesteps must be a one-dimensional integer sequence")
        values = values.tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("timesteps must be a one-dimensional integer sequence")
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("timesteps must contain non-negative integers")
    if any(earlier <= later for earlier, later in zip(values, values[1:])):
        raise ValueError("timesteps must be strictly decreasing in denoising order")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Actual production inputs beyond the artifact's model/dataset/schedule fields."""

    model_content: Mapping[str, Any]
    dataset_content: str | None
    numerics: Mapping[str, Any]
    method_parameters: Mapping[str, Any] | None = None
    guidance_scale: float | None = None
    model_commit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_content", _json_object(self.model_content, "model_content"))
        object.__setattr__(self, "numerics", _json_object(self.numerics, "numerics"))
        _optional_string(self.model_commit, "model_commit")
        if self.dataset_content is not None and (
            not isinstance(self.dataset_content, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.dataset_content) is None
        ):
            raise ValueError("dataset_content must be a SHA-256 digest or None")
        if self.method_parameters is not None:
            object.__setattr__(self, "method_parameters", _json_object(self.method_parameters, "method_parameters"))
        if self.guidance_scale is not None:
            if (isinstance(self.guidance_scale, bool)
                    or not isinstance(self.guidance_scale, (int, float))
                    or not math.isfinite(self.guidance_scale) or self.guidance_scale < 0):
                raise ValueError("Saved guidance_scale must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ArtifactSettings:
    """Expected current inputs, never a substitute for saved production metadata."""

    model_id: str
    model_revision: str | None
    dataset_ref: str
    dataset_fingerprint: str | None
    scheduler_config: Mapping[str, Any]
    eta: float
    timesteps_for_steps: Callable[[int], Sequence[int]]
    scheduler_id: str = "ddim"
    model_content: Mapping[str, Any] | None = None
    dataset_content: str | None = None
    numerics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("model_content", "numerics"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _json_object(value, name))


@dataclass(slots=True)
class InversionArtifact:
    """A single sample's saved inversion result.

    ``per_step_state`` maps method-defined names to tensors whose first
    dimension matches ``timesteps``. The state is intentionally opaque to the
    common editor; a method-specific :class:`DenoisingHook` interprets it.
    """

    method_id: str
    model_id: str
    model_revision: str | None
    dataset_ref: str
    dataset_fingerprint: str | None
    sample_uid: str
    scheduler_id: str
    scheduler_config: Mapping[str, Any]
    eta: float
    timesteps: tuple[int, ...]
    terminal_latent: torch.Tensor
    per_step_state: Mapping[str, torch.Tensor] = field(default_factory=dict)
    provenance: ArtifactProvenance | None = None

    def __post_init__(self) -> None:
        self.method_id = _required_string(self.method_id, "method_id")
        self.model_id = _required_string(self.model_id, "model_id")
        self.model_revision = _optional_string(self.model_revision, "model_revision")
        self.dataset_ref = _required_string(self.dataset_ref, "dataset_ref")
        self.dataset_fingerprint = _optional_string(
            self.dataset_fingerprint, "dataset_fingerprint"
        )
        self.sample_uid = _required_string(self.sample_uid, "sample_uid")
        self.scheduler_id = _required_string(self.scheduler_id, "scheduler_id")
        self.scheduler_config = normalize_scheduler_config(self.scheduler_config)
        if isinstance(self.eta, bool) or not isinstance(self.eta, (int, float)):
            raise ValueError("eta must be a finite non-negative number")
        self.eta = float(self.eta)
        if not math.isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be a finite non-negative number")
        self.timesteps = _timesteps_tuple(self.timesteps)
        self.validate()

    def validate(self) -> None:
        """Validate current metadata and tensors without normalizing or mutating them."""
        if self.provenance is not None and not isinstance(self.provenance, ArtifactProvenance):
            raise TypeError("Artifact provenance must be ArtifactProvenance or None")
        self.validate_metadata(self.metadata())
        self.validate_terminal_latent()
        if not isinstance(self.per_step_state, Mapping):
            raise ValueError("per_step_state must be a mapping")
        for key, tensor in self.per_step_state.items():
            if not isinstance(key, str) or not _STATE_KEY_RE.fullmatch(key):
                raise ValueError(
                    "per_step_state keys must begin with a letter and contain "
                    "only letters, digits, underscores, dots, or hyphens"
                )
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"per_step_state[{key!r}] must be a PyTorch tensor")
            if tensor.ndim == 0 or tensor.shape[0] != len(self.timesteps):
                raise ValueError(
                    f"per_step_state[{key!r}] must have one entry per timestep"
                )

    def validate_terminal_latent(self) -> None:
        """Reject malformed or non-finite terminal state without loading other entries."""
        if not isinstance(self.terminal_latent, torch.Tensor):
            raise ValueError("terminal_latent must be a PyTorch tensor")
        if (
            self.terminal_latent.ndim != 4
            or self.terminal_latent.shape[:2] != (1, 4)
            or min(self.terminal_latent.shape[2:]) < 1
        ):
            raise ValueError("terminal_latent must have shape [1, 4, height, width] with positive dimensions")
        if not self.terminal_latent.is_floating_point():
            raise ValueError("terminal_latent must use a floating-point dtype")
        if not torch.isfinite(self.terminal_latent).all().item():
            raise ValueError("terminal_latent must contain only finite values")

    @property
    def requires_denoising_hook(self) -> bool:
        """Whether method-specific state must be interpreted during editing."""
        return bool(self.per_step_state)

    def metadata(self) -> dict[str, Any]:
        """Serialize only identity and production settings, without copying tensors."""
        return {
            "method_id": self.method_id, "sample_uid": self.sample_uid,
            "model_id": self.model_id, "model_revision": self.model_revision,
            "dataset_ref": self.dataset_ref, "dataset_fingerprint": self.dataset_fingerprint,
            "scheduler_id": self.scheduler_id, "scheduler_config": dict(self.scheduler_config),
            "eta": self.eta, "timesteps": list(self.timesteps),
            "provenance": None if self.provenance is None else asdict(self.provenance),
        }

    @staticmethod
    def validate_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a saved identity without loading its tensor payload."""
        value = _json_object(value, "artifact metadata")
        if set(value) != {
            "method_id", "sample_uid", "model_id", "model_revision", "dataset_ref",
            "dataset_fingerprint", "scheduler_id", "scheduler_config", "eta", "timesteps", "provenance",
        }:
            raise ValueError("Invalid artifact production metadata fields")
        for name in ("method_id", "sample_uid", "model_id", "dataset_ref", "scheduler_id"):
            _required_string(value[name], name)
        for name in ("model_revision", "dataset_fingerprint"):
            _optional_string(value[name], name)
        value["scheduler_config"] = normalize_scheduler_config(value["scheduler_config"])
        value["timesteps"] = list(_timesteps_tuple(value["timesteps"]))
        eta = value["eta"]
        if isinstance(eta, bool) or not isinstance(eta, (int, float)) or not math.isfinite(eta) or eta < 0:
            raise ValueError("Saved eta must be finite and nonnegative")
        if value["provenance"] is not None:
            provenance = value["provenance"]
            if not isinstance(provenance, dict) or set(provenance) != {
                "model_content", "dataset_content", "numerics", "method_parameters", "guidance_scale", "model_commit",
            }:
                raise ValueError("Invalid artifact provenance fields")
            value["provenance"] = asdict(ArtifactProvenance(**provenance))
        return value

    def _production_mismatches(self, expected: Mapping[str, Any]) -> list[str]:
        mismatches = []
        for name, value in expected.items():
            actual = getattr(self, name)
            if name == "eta":
                matches = math.isclose(actual, value, rel_tol=0, abs_tol=1e-9)
            else:
                matches = actual == value
            if not matches:
                mismatches.append(name)
        return mismatches

    def validate_settings(self, settings: ArtifactSettings, *, steps: int) -> None:
        """Compare saved production inputs with independently supplied expectations."""
        mismatches = self._production_mismatches({
            "model_id": settings.model_id, "model_revision": settings.model_revision,
            "dataset_ref": settings.dataset_ref, "dataset_fingerprint": settings.dataset_fingerprint,
            "scheduler_id": settings.scheduler_id,
            "scheduler_config": normalize_scheduler_config(settings.scheduler_config),
            "timesteps": _timesteps_tuple(settings.timesteps_for_steps(steps)), "eta": settings.eta,
        })
        for name in ("model_content", "dataset_content", "numerics"):
            expected = getattr(settings, name)
            if expected is not None and (self.provenance is None or getattr(self.provenance, name) != expected):
                mismatches.append(name)
        if mismatches:
            raise ArtifactCompatibilityError("Saved artifact inputs differ from this run: " + ", ".join(mismatches))

    def validate_compatibility(
        self, config: Any, *, dataset_ref: str, dataset_fingerprint: str | None,
        timesteps: Sequence[int] | torch.Tensor,
        scheduler_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Compare against the active denoising schedule and latent resolution."""
        expected = {
            "model_id": config.model.model_id, "model_revision": config.model.revision,
            "dataset_ref": dataset_ref, "dataset_fingerprint": dataset_fingerprint,
            "scheduler_id": "ddim", "eta": config.sampling.eta,
            "timesteps": _timesteps_tuple(timesteps),
        }
        if scheduler_config is not None:
            expected["scheduler_config"] = normalize_scheduler_config(scheduler_config)
        mismatches = self._production_mismatches(expected)
        if len(self.timesteps) != config.sampling.num_inference_steps:
            mismatches.append("step count")
        expected_shape = (1, 4, config.model.height // 8, config.model.width // 8)
        if tuple(self.terminal_latent.shape) != expected_shape:
            mismatches.append("latent resolution")
        if mismatches:
            raise ArtifactCompatibilityError("Inversion artifact is incompatible with this run: " + ", ".join(mismatches))
