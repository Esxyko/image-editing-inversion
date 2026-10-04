"""Portable inversion artifact validation and serialization.

Images, prompts, and masks remain in the dataset. Artifacts store the terminal
latent and provenance needed to reproduce its denoising schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


_SCHEMA_VERSION = 1
_METADATA_FILE = "artifact.json"
_TENSORS_FILE = "tensors.safetensors"
_TERMINAL_KEY = "terminal_latent"
_STATE_PREFIX = "state/"
_STATE_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


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


def _tensor_manifest(tensors: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        key: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for key, tensor in tensors.items()
    }


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
        self.scheduler_config = _json_object(self.scheduler_config, "scheduler_config")
        if isinstance(self.eta, bool) or not isinstance(self.eta, (int, float)):
            raise ValueError("eta must be a finite non-negative number")
        self.eta = float(self.eta)
        if not math.isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be a finite non-negative number")
        self.timesteps = _timesteps_tuple(self.timesteps)
        if not isinstance(self.terminal_latent, torch.Tensor):
            raise ValueError("terminal_latent must be a PyTorch tensor")
        if self.terminal_latent.ndim != 4 or self.terminal_latent.shape[:2] != (1, 4):
            raise ValueError("terminal_latent must have shape [1, 4, height, width]")
        if not self.terminal_latent.is_floating_point():
            raise ValueError("terminal_latent must use a floating-point dtype")
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

    @property
    def requires_denoising_hook(self) -> bool:
        """Whether method-specific state must be interpreted during editing."""
        return bool(self.per_step_state)

    def validate_compatibility(
        self,
        config: Any,
        *,
        dataset_ref: str,
        dataset_fingerprint: str | None,
        timesteps: Sequence[int] | torch.Tensor,
        scheduler_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Reject an artifact that does not match a resolved editing run.

        ``timesteps`` must come from the actual scheduler, not just its step
        count. Pass ``scheduler_config`` when the editor has constructed its
        scheduler to also compare the complete scheduler configuration.
        """
        mismatches: list[str] = []
        if self.model_id != config.model.model_id:
            mismatches.append("model ID")
        if self.model_revision != config.model.revision:
            mismatches.append("model revision")
        if self.dataset_ref != dataset_ref:
            mismatches.append("dataset reference")
        if self.dataset_fingerprint != dataset_fingerprint:
            mismatches.append("dataset fingerprint")
        if self.scheduler_id != "ddim":
            mismatches.append("scheduler type")
        if len(self.timesteps) != config.sampling.num_inference_steps:
            mismatches.append("step count")
        if not math.isclose(self.eta, config.sampling.eta, rel_tol=0, abs_tol=1e-9):
            mismatches.append("DDIM eta")
        if self.timesteps != _timesteps_tuple(timesteps):
            mismatches.append("timestep schedule")
        if scheduler_config is not None and self.scheduler_config != _json_object(
            scheduler_config, "scheduler_config"
        ):
            mismatches.append("scheduler configuration")
        expected_shape = (1, 4, config.model.height // 8, config.model.width // 8)
        if tuple(self.terminal_latent.shape) != expected_shape:
            mismatches.append("latent resolution")
        if mismatches:
            raise ValueError(
                "Inversion artifact is incompatible with this run: "
                + ", ".join(mismatches)
            )

    def save(self, path: str | Path) -> Path:
        """Write JSON metadata and pickle-free safetensors into ``path``."""
        directory = Path(path)
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"Artifact path is not a directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / _METADATA_FILE
        tensors_path = directory / _TENSORS_FILE
        if metadata_path.exists() or tensors_path.exists():
            raise FileExistsError(f"Artifact already exists at {directory}")

        tensors = {_TERMINAL_KEY: self.terminal_latent}
        tensors.update(
            {_STATE_PREFIX + key: tensor for key, tensor in self.per_step_state.items()}
        )
        tensors = {
            key: tensor.detach().to(device="cpu").contiguous()
            for key, tensor in tensors.items()
        }
        metadata = {
            "schema_version": _SCHEMA_VERSION,
            "method_id": self.method_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dataset_ref": self.dataset_ref,
            "dataset_fingerprint": self.dataset_fingerprint,
            "sample_uid": self.sample_uid,
            "scheduler_id": self.scheduler_id,
            "scheduler_config": self.scheduler_config,
            "eta": self.eta,
            "timesteps": list(self.timesteps),
            "tensors": _tensor_manifest(tensors),
        }
        save_file(tensors, str(tensors_path))
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, path: str | Path) -> InversionArtifact:
        """Load and validate a schema-v1 artifact without deserializing pickle."""
        directory = Path(path)
        metadata_path = directory / _METADATA_FILE
        tensors_path = directory / _TENSORS_FILE
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read inversion artifact metadata at {directory}") from exc
        if not isinstance(metadata, dict) or metadata.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported inversion artifact schema at {directory}")
        expected_fields = {
            "schema_version",
            "method_id",
            "model_id",
            "model_revision",
            "dataset_ref",
            "dataset_fingerprint",
            "sample_uid",
            "scheduler_id",
            "scheduler_config",
            "eta",
            "timesteps",
            "tensors",
        }
        if set(metadata) != expected_fields:
            raise ValueError(f"Invalid inversion artifact metadata fields at {directory}")
        try:
            tensors = load_file(str(tensors_path), device="cpu")
        except Exception as exc:
            raise ValueError(f"Cannot read inversion artifact tensors at {directory}") from exc
        manifest = metadata["tensors"]
        if not isinstance(manifest, dict) or set(manifest) != set(tensors):
            raise ValueError(f"Inversion artifact tensor manifest mismatch at {directory}")
        if manifest != _tensor_manifest(tensors) or _TERMINAL_KEY not in tensors:
            raise ValueError(f"Inversion artifact tensor shape or dtype mismatch at {directory}")
        state = {}
        for key, tensor in tensors.items():
            if key == _TERMINAL_KEY:
                continue
            if not key.startswith(_STATE_PREFIX):
                raise ValueError(f"Unexpected inversion artifact tensor key {key!r}")
            state[key[len(_STATE_PREFIX) :]] = tensor
        try:
            return cls(
                method_id=metadata["method_id"],
                model_id=metadata["model_id"],
                model_revision=metadata["model_revision"],
                dataset_ref=metadata["dataset_ref"],
                dataset_fingerprint=metadata["dataset_fingerprint"],
                sample_uid=metadata["sample_uid"],
                scheduler_id=metadata["scheduler_id"],
                scheduler_config=metadata["scheduler_config"],
                eta=metadata["eta"],
                timesteps=metadata["timesteps"],
                terminal_latent=tensors[_TERMINAL_KEY],
                per_step_state=state,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid inversion artifact at {directory}: {exc}") from exc
