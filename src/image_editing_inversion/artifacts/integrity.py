"""Independent collection checksums and read-only legacy catalog verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from safetensors import safe_open
import torch

from .._project import project_paths
from .intermediates import tensor_checksum
from .inversion import ArtifactCompatibilityError


_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class _CollectionIntegrity:
    """Bind production metadata and each tensor's shape, dtype, and exact bytes."""

    def __init__(self, metadata: Mapping[str, str], keys: Sequence[str]) -> None:
        self._metadata = dict(metadata)
        self._keys = tuple(keys)
        self._checksums: dict[str, str] | None = None
        if "checksums" not in metadata:
            return
        checksums = json.loads(metadata["checksums"])
        if (not isinstance(checksums, dict) or set(checksums) != {"metadata", "tensors"}
                or not isinstance(checksums["tensors"], dict)
                or set(checksums["tensors"]) != set(keys)
                or any(not isinstance(value, str) or not _DIGEST_RE.fullmatch(value)
                       for value in [checksums["metadata"], *checksums["tensors"].values()])):
            raise ValueError("Invalid collection checksum metadata")
        if checksums["metadata"] != self._metadata_checksum(metadata):
            raise ValueError("Collection production metadata checksum mismatch")
        self._checksums = dict(checksums["tensors"])

    @staticmethod
    def _metadata_checksum(metadata: Mapping[str, str]) -> str:
        values = {key: value for key, value in metadata.items() if key != "checksums"}
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def with_checksums(
        cls, metadata: Mapping[str, str], checksums: Mapping[str, str],
    ) -> dict[str, str]:
        return {**metadata, "checksums": json.dumps({
            "metadata": cls._metadata_checksum(metadata), "tensors": dict(checksums),
        }, sort_keys=True, allow_nan=False)}

    @property
    def has_checksums(self) -> bool:
        return self._checksums is not None

    @property
    def checksums(self) -> Mapping[str, str]:
        if self._checksums is None:
            raise ValueError("Verify the legacy collection before using its tensor checksums")
        return self._checksums

    @staticmethod
    def _catalog_checksum(path: Path) -> str | None:
        """Read only the legacy file's integrity baseline, without creating a catalog."""
        root = project_paths().artifacts.resolve()
        if not path.is_relative_to(root):
            return None
        try:
            data = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
            raise ValueError("Invalid artifact catalog for legacy integrity verification")
        if data.get("schema_version") == 1:
            return None
        if data.get("schema_version") != 2 or not isinstance(data.get("groups"), dict):
            raise ValueError("Invalid artifact catalog for legacy integrity verification")
        group = data["groups"].get(path.relative_to(root).as_posix())
        if group is None:
            return None
        if (not isinstance(group, dict) or set(group) != {"checksum"}
                or not isinstance(group["checksum"], str)
                or not _DIGEST_RE.fullmatch(group["checksum"])):
            raise ValueError(f"Invalid artifact group checksum for {path}")
        return group["checksum"]

    def verify_tensor(self, key: str, tensor: torch.Tensor) -> None:
        if self._checksums is None or tensor_checksum(tensor) != self._checksums[key]:
            raise ValueError(f"Collection tensor checksum mismatch for {key!r}")

    def verify_file(self, path: Path, *, expected_checksum: str | None = None) -> None:
        """Verify every tensor using bounded memory; upgrade only trusted legacy state."""
        legacy = self._checksums is None
        if legacy:
            expected = expected_checksum if expected_checksum is not None else self._catalog_checksum(path)
            if expected is None:
                raise ArtifactCompatibilityError(
                    f"Collection {path} has no embedded or catalog checksum; regenerate it "
                    "before replay or merging"
                )
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(
                    f"Legacy collection checksum mismatch at {path}; its integrity cannot "
                    "be established from a stale catalog"
                )
        checksums = {}
        with safe_open(str(path), framework="pt", device="cpu") as saved:
            if saved.metadata() != self._metadata or set(saved.keys()) != set(self._keys):
                raise ValueError("Collection changed during integrity verification")
            for key in self._keys:
                # Do not retain mapped tensors after this call or across file replacement.
                if legacy:
                    checksums[key] = tensor_checksum(saved.get_tensor(key))
                else:
                    self.verify_tensor(key, saved.get_tensor(key))
        if legacy:
            self._checksums = checksums
