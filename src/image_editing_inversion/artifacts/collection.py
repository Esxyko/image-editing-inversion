"""Indexed safetensors collections and bounded-memory group serialization."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import struct
from typing import Any, Sequence

from safetensors import safe_open
from safetensors.torch import save_file

from .inversion import ArtifactCompatibilityError, ArtifactProvenance, ArtifactSettings, InversionArtifact
from .layout import method_directory_name, pipeline_group_parameters


_FILENAME = "artifacts.safetensors"
_KEY_RE = re.compile(r"^artifacts/(0|[1-9][0-9]*)/(terminal_latent|state/[A-Za-z][A-Za-z0-9_.-]*)$")
_MAX_HEADER = 100_000_000


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """A sample's stable position in a published group file."""

    path: Path
    index: int
    sample_uid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        if type(self.index) is not int or self.index < 0:
            raise ValueError("Artifact index must be a non-negative integer")
        if not isinstance(self.sample_uid, str) or not self.sample_uid.strip():
            raise ValueError("Artifact sample UID must be a non-empty string")


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Missing safetensors header at {path}")
        length = struct.unpack("<Q", prefix)[0]
        if length > _MAX_HEADER:
            raise ValueError(f"Safetensors header is too large at {path}")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header at {path}")
    return header, 8 + length


def _production_metadata(ids: Sequence[str], entries: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Intern shared production inputs, retaining one reference for every UID."""
    contexts: list[dict[str, Any]] = []
    indices: dict[str, int] = {}
    references = []
    for uid, entry in zip(ids, entries, strict=True):
        if entry["sample_uid"] != uid:
            raise ValueError("Production metadata does not match its UID")
        context = {key: value for key, value in entry.items() if key != "sample_uid"}
        canonical = json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if canonical not in indices:
            indices[canonical] = len(contexts)
            contexts.append(context)
        references.append(indices[canonical])
    return {
        "schema_version": "5", "ids": json.dumps(list(ids)),
        "entries": json.dumps(references), "contexts": json.dumps(contexts, allow_nan=False),
    }


def _inspect(path: Path, steps: int) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], tuple[dict[str, Any], ...]]:
    """Validate every entry's structure without materializing its tensors."""
    try:
        header, _ = _read_header(path)
        with safe_open(str(path), framework="pt", device="cpu") as saved:
            metadata = saved.metadata()
            if not metadata or metadata.get("schema_version") != "5":
                raise ValueError("Unsupported inversion artifact schema; regenerate with diffuse for schema v5")
            if set(metadata) != {"schema_version", "ids", "entries", "contexts"}:
                raise ValueError("Collection metadata must contain schema_version, ids, entries, and contexts")
            ids = json.loads(metadata["ids"])
            if (not isinstance(ids, list) or not ids
                    or any(not isinstance(uid, str) or not uid.strip() for uid in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("Collection IDs must be a nonempty list of unique nonempty strings")
            references = json.loads(metadata["entries"])
            contexts = json.loads(metadata["contexts"])
            if not isinstance(contexts, list) or not contexts:
                raise ValueError("Collection production contexts must be a nonempty list")
            if (not isinstance(references, list) or len(references) != len(ids)
                    or any(type(index) is not int or not 0 <= index < len(contexts) for index in references)):
                raise ValueError("Collection production metadata must align with IDs")
            validated = []
            for context in contexts:
                if not isinstance(context, dict) or "sample_uid" in context:
                    raise ValueError("Invalid shared production context")
                entry = InversionArtifact.validate_metadata({**context, "sample_uid": "shared-context"})
                entry.pop("sample_uid")
                validated.append(entry)
            entries = [{**validated[index], "sample_uid": uid}
                       for uid, index in zip(ids, references, strict=True)]
            for uid, entry in zip(ids, entries, strict=True):
                if entry["sample_uid"] != uid or len(entry["timesteps"]) != steps:
                    raise ValueError("Production metadata UID or timestep count does not match its entry")
            keys: list[list[str]] = [[] for _ in ids]
            terminals: set[int] = set()
            for key in saved.keys():
                match = _KEY_RE.fullmatch(key)
                if match is None or int(match[1]) >= len(ids):
                    raise ValueError(f"Unexpected collection tensor key {key!r}")
                index = int(match[1])
                shape = saved.get_slice(key).get_shape()
                if match[2] == "terminal_latent":
                    dtype = header[key]["dtype"]
                    floating = dtype in {"F16", "BF16", "F32", "F64"} or dtype.startswith("F8")
                    if len(shape) != 4 or shape[:2] != [1, 4] or not floating:
                        raise ValueError(f"Invalid terminal latent at position {index}")
                    terminals.add(index)
                elif not shape or shape[0] != steps:
                    raise ValueError(f"Invalid per-step state at position {index}")
                keys[index].append(key)
            if terminals != set(range(len(ids))):
                raise ValueError("Every collection ID must have exactly one terminal latent")
            return tuple(ids), tuple(tuple(entry) for entry in keys), tuple(entries)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"Invalid inversion artifact collection at {path}: {exc}") from exc
    except Exception as exc:
        # The native safetensors reader raises its own exception type.
        raise ValueError(f"Cannot read inversion artifact collection at {path}: {exc}") from exc


class ArtifactCollection:
    """A group file whose ordered IDs correspond to numbered tensor entries."""

    filename = _FILENAME

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.name != self.filename:
            raise ValueError(f"Artifact collection filename must be {self.filename}")
        self.method_id = method_directory_name(self.path.parent.parent.name)
        self.parameters = pipeline_group_parameters(self.path.parent.name)
        self.ids, self._keys, self._entries = _inspect(self.path, self.parameters["num_inference_steps"])
        if any(entry["method_id"] != self.method_id for entry in self._entries):
            raise ValueError("Saved artifact methods do not match their collection directory")
        if any(entry["provenance"] is not None
               and entry["provenance"]["guidance_scale"] is not None
               and entry["provenance"]["guidance_scale"] != self.parameters["guidance_scale"]
               for entry in self._entries):
            raise ValueError("Saved artifact guidance does not match its collection directory")
        self._positions = {uid: index for index, uid in enumerate(self.ids)}
        self._identity = self._file_identity()

    def _file_identity(self) -> tuple[int, ...]:
        stat = self.path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino

    @property
    def references(self) -> tuple[ArtifactReference, ...]:
        return tuple(ArtifactReference(self.path, index, uid) for index, uid in enumerate(self.ids))

    def reference(self, sample_uid: str) -> ArtifactReference:
        try:
            return ArtifactReference(self.path, self._positions[sample_uid], sample_uid)
        except KeyError as exc:
            raise ValueError(f"Artifact ID not found: {sample_uid}") from exc

    def load(self, reference: ArtifactReference, *, settings: ArtifactSettings) -> InversionArtifact:
        if self._file_identity() != self._identity:
            raise ValueError("Artifact collection changed; reopen it before loading")
        if (reference.path != self.path or reference.index >= len(self.ids)
                or self.ids[reference.index] != reference.sample_uid):
            raise ValueError("Artifact reference does not match the collection's ordered IDs")
        prefix = f"artifacts/{reference.index}/"
        try:
            with safe_open(str(self.path), framework="pt", device="cpu") as saved:
                # Detach loaded batches from file mappings before Windows replacement.
                tensors = {key[len(prefix):]: saved.get_tensor(key).clone()
                           for key in self._keys[reference.index]}
            metadata = dict(self._entries[reference.index])
            provenance = metadata.pop("provenance")
            artifact = InversionArtifact(
                **metadata,
                provenance=None if provenance is None else ArtifactProvenance(**provenance),
                terminal_latent=tensors.pop("terminal_latent"),
                per_step_state={key[len("state/"):]: tensor for key, tensor in tensors.items()},
            )
            artifact.validate_settings(settings, steps=self.parameters["num_inference_steps"])
            return artifact
        except ArtifactCompatibilityError:
            raise
        except Exception as exc:
            raise ValueError(f"Cannot load artifact {reference.sample_uid!r} at {self.path}: {exc}") from exc

    @staticmethod
    def stage(path: Path, artifacts: Sequence[InversionArtifact]) -> tuple[ArtifactReference, ...]:
        """Serialize one batch, keeping tensor copies bounded by batch size."""
        ids = [artifact.sample_uid for artifact in artifacts]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("A staged batch must contain unique artifact IDs")
        tensors = {}
        for index, artifact in enumerate(artifacts):
            artifact.__post_init__()
            values = {"terminal_latent": artifact.terminal_latent}
            values.update({f"state/{key}": tensor for key, tensor in artifact.per_step_state.items()})
            tensors.update({f"artifacts/{index}/{key}": tensor.detach().cpu().contiguous().clone()
                            for key, tensor in values.items()})
        save_file(tensors, str(path), metadata=_production_metadata(
            ids, [artifact.metadata() for artifact in artifacts],
        ))
        return tuple(ArtifactReference(path, index, uid) for index, uid in enumerate(ids))

    @staticmethod
    def assemble(path: Path, references: Sequence[ArtifactReference], *, steps: int) -> None:
        """Stream selected tensor payloads into one new safetensors file."""
        ids = [reference.sample_uid for reference in references]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("A collection must contain unique artifact IDs")
        headers = {}
        source_keys = {}
        output: dict[str, Any] = {
            "__metadata__": {"schema_version": "5", "ids": json.dumps(ids)},
        }
        entries = []
        payloads = []
        offset = 0
        for index, reference in enumerate(references):
            if reference.path not in headers:
                source_ids, keys, source_entries = _inspect(reference.path, steps)
                header, start = _read_header(reference.path)
                headers[reference.path] = (header, start, source_ids, source_entries)
                source_keys[reference.path] = keys
            header, start, source_ids, source_entries = headers[reference.path]
            if reference.index >= len(source_ids) or source_ids[reference.index] != reference.sample_uid:
                raise ValueError("Source reference does not match its ID list")
            entries.append(source_entries[reference.index])
            prefix = f"artifacts/{reference.index}/"
            for key in sorted(source_keys[reference.path][reference.index],
                              key=lambda name: header[name]["data_offsets"][0]):
                descriptor = header[key]
                begin, end = descriptor["data_offsets"]
                length = end - begin
                output[f"artifacts/{index}/{key[len(prefix):]}"] = {
                    "dtype": descriptor["dtype"], "shape": descriptor["shape"],
                    "data_offsets": [offset, offset + length],
                }
                payloads.append((reference.path, start + begin, length))
                offset += length
        output["__metadata__"] = _production_metadata(ids, entries)
        encoded = json.dumps(output, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        if len(encoded) > _MAX_HEADER:
            raise ValueError("Combined safetensors header is too large")
        with path.open("xb") as destination:
            destination.write(struct.pack("<Q", len(encoded)))
            destination.write(encoded)
            source_path = None
            source = None
            try:
                for tensor_path, position, remaining in payloads:
                    if tensor_path != source_path:
                        if source is not None:
                            source.close()
                        source = tensor_path.open("rb")
                        source_path = tensor_path
                    source.seek(position)
                    while remaining:
                        chunk = source.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise ValueError(f"Truncated tensor payload at {tensor_path}")
                        destination.write(chunk)
                        remaining -= len(chunk)
            finally:
                if source is not None:
                    source.close()
            destination.flush()
            os.fsync(destination.fileno())
        _inspect(path, steps)
