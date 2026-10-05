"""Cache individual inversions and atomically publish complete group files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .._project import project_paths
from .collection import ArtifactCollection, ArtifactReference
from .inversion import ArtifactSettings, InversionArtifact
from .layout import ArtifactGroup, method_directory_name, pipeline_group_name, pipeline_group_parameters


_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


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
    """Own a schema-v2 cache catalog; publication assumes one active writer."""

    def __init__(self) -> None:
        self.root = project_paths().artifacts.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "catalog.json"
        self._entries: dict[str, dict[str, Any]] = {}
        self._groups: dict[str, dict[str, str]] = {}
        self._checked: dict[Path, tuple[tuple[Any, ...], ArtifactCollection | None]] = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return  # Create/replace the catalog only after successful publication.
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read artifact catalog at {self.path}: {exc}") from exc
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or not isinstance(data.get("entries"), dict)):
            raise ValueError(f"Invalid artifact catalog schema at {self.path}")
        if data["schema_version"] == 1 and set(data) == {"schema_version", "entries"}:
            return  # Old files and entries remain untouched until new publication.
        if (data["schema_version"] != 2 or set(data) != {"schema_version", "entries", "groups"}
                or not isinstance(data["groups"], dict)):
            raise ValueError(f"Unsupported artifact catalog schema at {self.path}")
        for relative, group in data["groups"].items():
            self._artifact_path(relative)
            if (not isinstance(group, dict) or set(group) != {"checksum"}
                    or not isinstance(group["checksum"], str)
                    or not _DIGEST_RE.fullmatch(group["checksum"])):
                raise ValueError(f"Invalid artifact group checksum for {relative!r}")
        positions = set()
        samples = set()
        for key, entry in data["entries"].items():
            if (not isinstance(key, str) or not _DIGEST_RE.fullmatch(key)
                    or not isinstance(entry, dict)
                    or set(entry) != {"inputs", "artifact", "index", "sample_uid"}):
                raise ValueError(f"Invalid artifact catalog entry {key!r}")
            inputs = _normalize(entry["inputs"])
            path = self._artifact_path(entry["artifact"])
            ArtifactReference(path, entry["index"], entry["sample_uid"])
            if (_key(inputs) != key or inputs.get("sample_uid") != entry["sample_uid"]
                    or inputs.get("method_id") != ArtifactGroup.from_path(path).method_id
                    or entry["artifact"] not in data["groups"]):
                raise ValueError(f"Artifact catalog entry identity mismatch for {key!r}")
            position = (entry["artifact"], entry["index"])
            sample = (entry["artifact"], entry["sample_uid"])
            if position in positions or sample in samples:
                raise ValueError(f"Duplicate artifact catalog position or UID for {key!r}")
            positions.add(position)
            samples.add(sample)
        self._entries = data["entries"]
        self._groups = data["groups"]

    def _artifact_path(self, relative: str) -> Path:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("Artifact file must be a relative path")
        parts = Path(relative).parts
        if len(parts) != 3 or parts[2] != ArtifactCollection.filename:
            raise ValueError("Artifact file must use method/steps-N_guidance-G/artifacts.safetensors")
        if relative != "/".join(parts):
            raise ValueError("Artifact paths must use canonical relative forward-slash paths")
        path = (self.root / relative).resolve()
        if self.root not in path.parents or path != self.root / relative:
            raise ValueError("Artifact file must stay inside the catalog root without redirects")
        ArtifactGroup.from_path(path)
        return path

    def _write(self) -> None:
        temporary = self.root / f".catalog-{uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps({"schema_version": 2, "entries": self._entries, "groups": self._groups},
                           indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
            )
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _collection(self, path: Path) -> ArtifactCollection | None:
        """Check a group checksum once per unchanged file, then read lazily."""
        relative = path.relative_to(self.root).as_posix()
        expected = self._groups.get(relative, {}).get("checksum")
        try:
            stat = path.stat()
            identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, expected)
            checked = self._checked.get(path)
            if checked is not None and checked[0] == identity:
                return checked[1]
            collection = (None if expected is not None and _file_digest(path) != expected
                          else ArtifactCollection(path))
            self._checked[path] = (identity, collection)
            return collection
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def lookup(
        self, inputs: Mapping[str, Any], *, settings: ArtifactSettings,
    ) -> tuple[InversionArtifact, ArtifactReference] | None:
        normalized = _normalize(inputs)
        entry = self._entries.get(_key(normalized))
        if entry is None or entry["inputs"] != normalized:
            return None
        try:
            path = self._artifact_path(entry["artifact"])
            collection = self._collection(path)
            if collection is None:
                return None
            reference = ArtifactReference(path, entry["index"], entry["sample_uid"])
            artifact = collection.load(reference, settings=settings)
            if (artifact.provenance is None
                    or artifact.provenance.method_parameters != normalized.get("method_parameters")):
                return None
            return artifact, reference
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def group(self, method_id: str, *, pipeline_h_params: Mapping[str, Any]) -> ArtifactGroupPublication:
        """Start a transaction that stages batches and publishes once."""
        return ArtifactGroupPublication(self, method_id, pipeline_h_params)

    def store_group(
        self,
        artifacts: Sequence[InversionArtifact],
        inputs: Sequence[Mapping[str, Any] | None] | None = None,
        *,
        pipeline_h_params: Mapping[str, Any],
    ) -> tuple[ArtifactReference, ...]:
        """Merge ordered artifacts and aligned cache inputs into one group."""
        if not artifacts:
            raise ValueError("At least one inversion artifact is required")
        cache_inputs = [None] * len(artifacts) if inputs is None else inputs
        with self.group(artifacts[0].method_id, pipeline_h_params=pipeline_h_params) as publication:
            publication.stage(artifacts, cache_inputs)
            publication.publish()
            return tuple(publication.reference(artifact.sample_uid) for artifact in artifacts)


class ArtifactGroupPublication:
    """Stage an ordered merge without changing the published group on failure."""

    def __init__(self, catalog: ArtifactCatalog, method_id: str, parameters: Mapping[str, Any]) -> None:
        self.catalog = catalog
        self.method_id = method_directory_name(method_id)
        group_name = pipeline_group_name(parameters)
        self.parameters = pipeline_group_parameters(group_name)
        self.path = catalog._artifact_path(f"{self.method_id}/{group_name}/{ArtifactCollection.filename}")
        self._base = catalog._collection(self.path)
        self._sources: dict[str, ArtifactReference] = {} if self._base is None else {
            reference.sample_uid: reference for reference in self._base.references
        }
        self._inputs: dict[str, dict[str, Any] | None] = {uid: None for uid in self._sources}
        relative = self.path.relative_to(catalog.root).as_posix()
        if self._base is not None:
            for entry in catalog._entries.values():
                if entry["artifact"] == relative:
                    reference = self._sources.get(entry["sample_uid"])
                    if reference is not None and reference.index == entry["index"]:
                        self._inputs[reference.sample_uid] = entry["inputs"]
        self._temporary: TemporaryDirectory | None = None
        self._changed = False
        self._published = False

    def __enter__(self) -> ArtifactGroupPublication:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = self.path.parent / "h-params.json"
        if descriptor.exists() and json.loads(descriptor.read_text(encoding="utf-8")) != self.parameters:
            raise ValueError(f"Artifact group parameters do not match at {self.path.parent}")
        self._temporary = TemporaryDirectory(prefix=".pending-", dir=self.path.parent)
        return self

    def __exit__(self, exc_type: Any, error: BaseException | None, traceback: Any) -> None:
        if self._temporary is not None:
            try:
                self._temporary.cleanup()
            except BaseException as cleanup:
                if error is None:
                    raise
                error.add_note(f"Artifact staging cleanup failed: {cleanup}")
            finally:
                self._temporary = None

    def stage(
        self, artifacts: Sequence[InversionArtifact | ArtifactReference],
        inputs: Sequence[Mapping[str, Any] | None],
    ) -> None:
        if self._temporary is None or self._published:
            raise RuntimeError("Artifact publication is not active")
        if len(artifacts) != len(inputs):
            raise ValueError("Artifacts and cache inputs must have the same length")
        normalized = [None if value is None else _normalize(value) for value in inputs]
        fresh = []
        ids = set()
        for artifact, cache_inputs in zip(artifacts, normalized, strict=True):
            uid = artifact.sample_uid
            if uid in ids:
                raise ValueError("A staged batch must contain unique artifact IDs")
            ids.add(uid)
            if cache_inputs is not None and (
                cache_inputs.get("sample_uid") != uid or cache_inputs.get("method_id") != self.method_id
            ):
                raise ValueError("Cache inputs must match the artifact's sample UID and method")
            if isinstance(artifact, InversionArtifact):
                if artifact.method_id != self.method_id or len(artifact.timesteps) != self.parameters["num_inference_steps"]:
                    raise ValueError("Artifact method and steps must match its group")
                if (artifact.provenance is not None and artifact.provenance.guidance_scale is not None
                        and artifact.provenance.guidance_scale != self.parameters["guidance_scale"]):
                    raise ValueError("Artifact production guidance must match its group")
                fresh.append(artifact)
            elif not isinstance(artifact, ArtifactReference) or self._sources.get(uid) != artifact:
                raise ValueError("Reused artifact reference must belong to the current published group")
        staged = {}
        if fresh:
            temporary_path = Path(self._temporary.name) / f"batch-{uuid4().hex}.safetensors"
            staged = {reference.sample_uid: reference
                      for reference in ArtifactCollection.stage(temporary_path, fresh)}
            self._changed = True
        # Dict replacement retains existing positions; additions follow traversal order.
        for artifact, cache_inputs in zip(artifacts, normalized, strict=True):
            uid = artifact.sample_uid
            self._sources[uid] = staged[uid] if isinstance(artifact, InversionArtifact) else artifact
            self._inputs[uid] = cache_inputs

    def reference(self, sample_uid: str) -> ArtifactReference:
        if not self._published:
            raise RuntimeError("Artifact group has not been published")
        return self._sources[sample_uid]

    def publish(self) -> tuple[ArtifactReference, ...]:
        if self._temporary is None or self._published:
            raise RuntimeError("Artifact publication is not active")
        if not self._sources:
            raise ValueError("Cannot publish an empty artifact group")
        relative = self.path.relative_to(self.catalog.root).as_posix()
        if self._changed:
            combined = Path(self._temporary.name) / "combined.safetensors"
            ArtifactCollection.assemble(combined, tuple(self._sources.values()),
                                        steps=self.parameters["num_inference_steps"])
            checksum = _file_digest(combined)
            # Publish the descriptor before the file; existing descriptors are unchanged.
            descriptor = self.path.parent / "h-params.json"
            if not descriptor.exists():
                staged_descriptor = Path(self._temporary.name) / "h-params.json"
                staged_descriptor.write_text(json.dumps(self.parameters, indent=2) + "\n", encoding="utf-8")
                staged_descriptor.replace(descriptor)
            combined.replace(self.path)
        else:
            checksum = self.catalog._groups.get(relative, {}).get("checksum") or _file_digest(self.path)
        references = tuple(ArtifactReference(self.path, index, uid) for index, uid in enumerate(self._sources))
        entries = {key: entry for key, entry in self.catalog._entries.items() if entry["artifact"] != relative}
        for reference in references:
            cache_inputs = self._inputs[reference.sample_uid]
            if cache_inputs is not None:
                entries[_key(cache_inputs)] = {
                    "inputs": cache_inputs, "artifact": relative,
                    "index": reference.index, "sample_uid": reference.sample_uid,
                }
        if (entries != self.catalog._entries
                or self.catalog._groups.get(relative) != {"checksum": checksum}):
            self.catalog._entries = entries
            self.catalog._groups[relative] = {"checksum": checksum}
            self.catalog._write()
        if self._changed:
            self.catalog._checked.pop(self.path, None)
        self._sources = {reference.sample_uid: reference for reference in references}
        self._published = True
        return references
