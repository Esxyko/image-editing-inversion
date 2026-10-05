"""Validated, atomic storage for reusable tensor components, independent of methods."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file


TensorSpecs = Mapping[str, tuple[tuple[int, ...], torch.dtype]]


def tensor_checksum(tensor: torch.Tensor) -> str:
    """Hash tensor shape, dtype, and exact bytes, including bfloat16 tensors."""
    tensor = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


class IntermediateCache:
    """Store per-image components without exposing them as replay artifacts.

    Entries are independently published; no catalog or whole-dataset tensor
    memory is required. As with final artifacts, publication assumes one writer.
    """

    def __init__(self, root: Path = Path("data/cache/inversion")) -> None:
        self.root = root.expanduser().resolve()

    def _path(self, component: str, inputs: Mapping[str, Any]) -> Path:
        if re.fullmatch(r"[a-z][a-z0-9_]*", component) is None:
            raise ValueError("Intermediate component names must be lowercase identifiers")
        key = hashlib.sha256(_canonical(inputs).encode("utf-8")).hexdigest()
        path = self.root / component / f"{key}.safetensors"
        if path.resolve() != path or path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Intermediate cache paths must not redirect outside their location")
        return path

    @staticmethod
    def _validate(tensors: Mapping[str, torch.Tensor], specs: TensorSpecs) -> None:
        if set(tensors) != set(specs):
            raise ValueError("Intermediate tensor names do not match their component")
        for name, (shape, dtype) in specs.items():
            tensor = tensors[name]
            if (not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point()
                    or tuple(tensor.shape) != shape or tensor.dtype != dtype
                    or not torch.isfinite(tensor).all().item()):
                raise ValueError(f"Invalid intermediate tensor {name!r}")

    @staticmethod
    def _validate_producer(producer: Mapping[str, Any]) -> None:
        if (not isinstance(producer, dict) or set(producer) != {"uids", "batch_size"}
                or not isinstance(producer["uids"], list) or not producer["uids"]
                or any(not isinstance(uid, str) or not uid.strip() for uid in producer["uids"])
                or type(producer["batch_size"]) is not int
                or producer["batch_size"] != len(producer["uids"])):
            raise ValueError("Invalid intermediate producer batch metadata")

    @torch.inference_mode(False)
    @torch.no_grad()
    def load(
        self, component: str, inputs: Mapping[str, Any], *, specs: TensorSpecs,
    ) -> dict[str, torch.Tensor] | None:
        """Return independent ordinary CPU tensors; invalid entries are misses."""
        try:
            path = self._path(component, inputs)
            with safe_open(str(path), framework="pt", device="cpu") as stream:
                metadata = stream.metadata()
                if (not isinstance(metadata, dict)
                        or set(metadata) != {"schema_version", "inputs", "producer", "checksums"}
                        or metadata["schema_version"] != "1"
                        or metadata["inputs"] != _canonical(inputs)
                        or set(stream.keys()) != set(specs)):
                    return None
                producer = json.loads(metadata["producer"])
                self._validate_producer(producer)
                if inputs.get("sample_uid") not in producer["uids"]:
                    return None
                tensors: dict[str, torch.Tensor] = {}
                for name, (shape, dtype) in specs.items():
                    if tuple(stream.get_slice(name).get_shape()) != shape:
                        return None
                    tensor = stream.get_tensor(name)
                    if tensor.dtype != dtype:
                        return None
                    tensors[name] = tensor.clone().contiguous()
            self._validate(tensors, specs)
            checksums = json.loads(metadata["checksums"])
            if not isinstance(checksums, dict) or checksums != {
                name: tensor_checksum(tensor) for name, tensor in tensors.items()
            }:
                return None
            return tensors
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, SafetensorError):
            return None

    @torch.inference_mode(False)
    @torch.no_grad()
    def store(
        self, component: str, inputs: Mapping[str, Any], tensors: Mapping[str, torch.Tensor],
        *, specs: TensorSpecs, producer: Mapping[str, Any],
    ) -> None:
        """Publish a complete entry atomically, leaving an existing entry on failure."""
        self._validate(tensors, specs)
        self._validate_producer(producer)
        if inputs.get("sample_uid") not in producer["uids"]:
            raise ValueError("Intermediate sample UID must belong to its producer batch")
        path = self._path(component, inputs)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = self._path(component, inputs)
        temporary = path.parent / f".{path.stem}-{uuid4().hex}.tmp"
        try:
            copies = {name: tensor.detach().to(device="cpu").clone().contiguous()
                      for name, tensor in tensors.items()}
            save_file(copies, str(temporary), metadata={
                "schema_version": "1", "inputs": _canonical(inputs),
                "producer": _canonical(producer),
                "checksums": _canonical({name: tensor_checksum(tensor)
                                          for name, tensor in copies.items()}),
            })
            temporary.replace(path)
        except SafetensorError as exc:
            raise ValueError(f"Cannot serialize intermediate component {component!r}: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
