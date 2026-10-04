"""Persist and find reusable inversions independently of experiment outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

from .inversion import InversionArtifact


_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_FILES = ("artifact.json", "tensors.safetensors")


def _normalize(inputs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(inputs, Mapping) or any(not isinstance(key, str) for key in inputs):
        raise ValueError("Inversion cache inputs must be a mapping with string keys")
    try:
        return json.loads(json.dumps(dict(inputs), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Inversion cache inputs must be JSON-compatible") from exc


def _key(inputs: Mapping[str, Any]) -> str:
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ArtifactCatalog:
    """Own a schema-v1 catalog and immutable artifact directories.

    Catalog updates assume one writer. A complete artifact is published before
    its catalog entry; failed/uncataloged directories are preserved.
    """

    def __init__(self, root: Path = Path("data/artifacts")) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "catalog.json"
        self._entries: dict[str, dict[str, Any]] = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._write()
            return
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read artifact catalog at {self.path}: {exc}") from exc
        if (
            not isinstance(data, dict)
            or set(data) != {"schema_version", "entries"}
            or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
            or not isinstance(data["entries"], dict)
        ):
            raise ValueError(f"Invalid artifact catalog schema at {self.path}")
        for key, entry in data["entries"].items():
            try:
                self._validate_entry(key, entry)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid artifact catalog entry {key!r}: {exc}") from exc
        self._entries = data["entries"]

    def _artifact_path(self, relative: str) -> Path:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("Artifact directory must be a relative path")
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ValueError("Artifact directory must stay inside the catalog root")
        return path

    def _validate_entry(self, key: str, entry: Any) -> None:
        if not isinstance(key, str) or not _DIGEST_RE.fullmatch(key):
            raise ValueError("Cache key must be a SHA-256 digest")
        if not isinstance(entry, dict) or set(entry) != {"inputs", "artifact", "checksums"}:
            raise ValueError("Entry must contain inputs, artifact, and checksums")
        inputs = _normalize(entry["inputs"])
        if _key(inputs) != key:
            raise ValueError("Cache key does not match its inversion inputs")
        self._artifact_path(entry["artifact"])
        checksums = entry["checksums"]
        if (
            not isinstance(checksums, dict)
            or set(checksums) != set(_ARTIFACT_FILES)
            or any(not isinstance(value, str) or not _DIGEST_RE.fullmatch(value)
                   for value in checksums.values())
        ):
            raise ValueError("Entry must include both artifact file checksums")

    def _write(self) -> None:
        temporary = self.root / f".catalog-{uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps({"schema_version": 1, "entries": self._entries},
                       indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def lookup(self, inputs: Mapping[str, Any]) -> tuple[InversionArtifact, Path] | None:
        """Return an intact matching artifact; missing/damaged files are misses."""
        normalized = _normalize(inputs)
        entry = self._entries.get(_key(normalized))
        if entry is None or entry["inputs"] != normalized:
            return None
        try:
            path = self._artifact_path(entry["artifact"])
            if any(_file_digest(path / name) != entry["checksums"][name]
                   for name in _ARTIFACT_FILES):
                return None
            return InversionArtifact.load(path), path
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def store(
        self, artifact: InversionArtifact, inputs: Mapping[str, Any] | None = None
    ) -> Path:
        """Publish an artifact, optionally making it discoverable for reuse."""
        normalized = None if inputs is None else _normalize(inputs)
        key = None if normalized is None else _key(normalized)
        suffix = uuid4().hex
        staging = self.root / f".pending-{suffix}"
        artifact.save(staging)
        checksums = {name: _file_digest(staging / name) for name in _ARTIFACT_FILES}
        destination = self.root / f"{key[:16] if key else 'uncached'}-{suffix}"
        staging.rename(destination)
        if key is not None:
            self._entries[key] = {
                "inputs": normalized,
                "artifact": destination.relative_to(self.root).as_posix(),
                "checksums": checksums,
            }
            self._write()
        return destination
