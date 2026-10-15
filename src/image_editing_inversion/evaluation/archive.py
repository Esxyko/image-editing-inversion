"""Validate archived manifests and stream one saved RGB tensor batch at a time."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, TYPE_CHECKING
from zipfile import ZipFile

from .models import MethodOutput, require

if TYPE_CHECKING:
    import torch
    from ..data import DatasetRepository


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


class ArchiveReader:
    """Own one read-only ZIP and its validated, complete method outputs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.hashes: dict[str, str] = {}
        self.methods: tuple[MethodOutput, ...] = ()
        self._zip: ZipFile | None = None

    def __enter__(self) -> ArchiveReader:
        status = self.path.stat()
        self._snapshot = (status.st_size, status.st_mtime_ns)
        self._zip = ZipFile(self.path)
        try:
            entries = self._zip.infolist()
            names = [entry.filename for entry in entries]
            require(len(names) == len(set(names)), f"Duplicate ZIP members: {self.path}")
            for name in names:
                self._relative(name.rstrip("/"))
            self._names = {entry.filename for entry in entries if not entry.is_dir()}
            manifests = [name for name in self._names if PurePosixPath(name).name == "sweep.json"]
            require(len(manifests) == 1, "Archive must contain exactly one sweep.json")
            self._root = PurePosixPath(manifests[0]).parent
            self.manifest = self._object(manifests[0])
            require(self.manifest.get("schema_version") in (1, 2), "Unsupported sweep schema")
            require(self.manifest.get("status") == "ok", "Evaluation requires a completed, successful run")
            return self
        except BaseException:
            self._zip.close()
            self._zip = None
            raise

    def __exit__(self, *args: Any) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    @staticmethod
    def _relative(value: str) -> PurePosixPath:
        require(isinstance(value, str) and bool(value), "Expected a nonempty relative ZIP path")
        path = PurePosixPath(value)
        require(
            not path.is_absolute() and ".." not in path.parts
            and "\\" not in value and ":" not in value and "\x00" not in value
            and path.as_posix() == value and value != ".",
            f"Invalid ZIP path: {value!r}",
        )
        return path

    def _member(self, directory: PurePosixPath | str, relative: str) -> str:
        return (PurePosixPath(directory) / self._relative(relative)).as_posix()

    def _read(self, member: str) -> bytes:
        require(member in self._names, f"Missing ZIP member: {member}")
        if self._zip is None:
            raise RuntimeError("Archive reader is closed")
        content = self._zip.read(member)
        self.hashes[member] = hashlib.sha256(content).hexdigest()
        return content

    def _object(self, member: str) -> dict[str, Any]:
        value = json.loads(self._read(member), parse_constant=_invalid_constant)
        require(isinstance(value, dict), f"Expected a JSON object: {member}")
        return value

    def _records(self, member: str) -> tuple[dict[str, Any], ...]:
        values = tuple(json.loads(line, parse_constant=_invalid_constant)
                       for line in self._read(member).splitlines() if line.strip())
        require(all(isinstance(value, dict) for value in values), f"Expected JSONL objects: {member}")
        return values

    def prepare(self, repository: DatasetRepository) -> None:
        """Validate metadata and sample identity before loading evaluation weights."""
        parameters = self.manifest.get("parameter_files")
        require(isinstance(parameters, list) and bool(parameters), "Run has no parameter files")
        methods: list[MethodOutput] = []
        seen_parameters: set[str] = set()
        for parameter in parameters:
            require(isinstance(parameter, dict), "Invalid parameter manifest entry")
            filename = parameter.get("pipeline_h_params_file")
            require(isinstance(filename, str), "Missing pipeline parameter filename")
            require(len(self._relative(filename).parts) == 1, "Parameter filename must be a basename")
            require(filename not in seen_parameters, f"Duplicate parameter file: {filename}")
            seen_parameters.add(filename)
            require(
                parameter.get("directory") == filename and parameter.get("status") == "ok",
                "Incomplete or mismatched parameter directory",
            )
            entries = parameter.get("methods")
            require(isinstance(entries, list) and bool(entries), f"No methods for {filename}")
            seen_methods: set[str] = set()
            reference_uids: tuple[str, ...] | None = None
            for entry in entries:
                require(isinstance(entry, dict), "Invalid method manifest entry")
                method_id = entry.get("method_id")
                require(isinstance(method_id, str) and bool(method_id), "Missing method ID")
                require(method_id not in seen_methods, f"Duplicate method: {filename}/{method_id}")
                seen_methods.add(method_id)
                require(entry.get("status") == "ok", f"Incomplete method: {filename}/{method_id}")
                directory = entry.get("directory")
                require(isinstance(directory, str), "Missing method directory")
                require(
                    self._relative(directory).parts == (filename, method_id),
                    "Method directory does not match its parameter/method",
                )
                directory = self._member(self._root, directory)
                configuration = self._object(self._member(directory, "resolved-config.json"))
                records = self._records(self._member(directory, "results.jsonl"))
                batches = self._records(self._member(directory, "batches.jsonl"))
                require(bool(records), f"Method has no results: {directory}")
                require(entry.get("counts") == {"ok": len(records), "skipped": 0, "error": 0},
                        "Method manifest counts do not match complete results")
                self._identity(configuration, filename, method_id, repository, records[0].get("uid"))
                require(isinstance(configuration.get("config"), dict), "Missing resolved configuration")
                model = configuration["config"].get("model")
                require(isinstance(model, dict), "Missing archived model settings")
                require(all(type(model.get(key)) is int and model[key] > 0 for key in ("height", "width")),
                        "Missing image dimensions in archived configuration")
                uids: list[str] = []
                for record in records:
                    require(record.get("status") == "ok", f"Incomplete result in {directory}")
                    self._identity(record, filename, method_id, repository, record.get("uid"))
                    require(record.get("settings_file") == "resolved-config.json", "Unexpected result settings file")
                    relative = record.get("tensor_file")
                    require(
                        isinstance(relative, str) and self._relative(relative).parts[0] == "tensors",
                        "Expected archived RGB tensor output",
                    )
                    require(self._member(directory, relative) in self._names, "Missing result tensor file")
                    require(type(record.get("tensor_index")) is int and record["tensor_index"] >= 0,
                            "Invalid tensor index")
                    uids.append(record["uid"])
                require(len(uids) == len(set(uids)), f"Duplicate result UID in {directory}")
                if reference_uids is None:
                    reference_uids = tuple(uids)
                require(tuple(uids) == reference_uids, f"Methods use different UID sequences for {filename}")
                for batch in batches:
                    require(
                        batch.get("method_id") == method_id and batch.get("pipeline_h_params_file") == filename,
                        "Batch configuration identity mismatch",
                    )
                methods.append(MethodOutput(filename, method_id, directory, configuration, records, batches))
        self.methods = tuple(methods)

    @staticmethod
    def _identity(value: dict[str, Any], filename: str, method_id: str,
                  repository: DatasetRepository, uid: str) -> None:
        require(value.get("pipeline_h_params_file") == filename and value.get("method_id") == method_id,
                "Result/configuration identity mismatch")
        require(isinstance(uid, str) and bool(uid), "Missing result UID")
        repository.validate_identity(value.get("dataset_ref"), value.get("dataset_fingerprint"), uid)

    def iter_images(
        self, method: MethodOutput, batch_size: int,
    ) -> Iterator[tuple[list[dict[str, Any]], torch.Tensor]]:
        """Coalesce saved batches up to the evaluation batch size, retaining UID order."""
        import torch
        from safetensors import SafetensorError
        from safetensors.torch import load

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in method.records:
            grouped[record["tensor_file"]].append(record)
        model = method.configuration["config"]["model"]
        height, width = model["height"], model["width"]
        pending_records: list[dict[str, Any]] = []
        pending_pixels: list[torch.Tensor] = []
        for relative, records in grouped.items():
            member = self._member(method.directory, relative)
            content = self._read(member)
            header_size = int.from_bytes(content[:8], "little")
            require(0 < header_size <= len(content) - 8, f"Invalid safetensors header: {member}")
            header = json.loads(content[8:8 + header_size], parse_constant=_invalid_constant)
            require(isinstance(header, dict), f"Invalid tensor header: {member}")
            metadata = header.get("__metadata__", {})
            require(isinstance(metadata, dict), "Missing tensor metadata")
            require(all(metadata.get(key) == value for key, value in {
                "schema_version": "1", "color_space": "RGB", "layout": "NCHW",
                "value_range": "0..255", "method_id": method.method_id,
                "pipeline_h_params_file": method.parameter_file,
            }.items()), f"Tensor metadata mismatch: {member}")
            ordered = json.loads(metadata.get("uids", "null"))
            require(isinstance(ordered, list) and all(isinstance(uid, str) for uid in ordered)
                    and len(ordered) == len(set(ordered)), "Invalid tensor UID list")
            require(
                sorted(record["tensor_index"] for record in records) == list(range(len(ordered))),
                "Incomplete or duplicate tensor index coverage",
            )
            require(all(record.get("edit_batch_id") == metadata.get("edit_batch_id")
                        and ordered[record["tensor_index"]] == record["uid"] for record in records),
                    "Tensor batch ID or UID/index mismatch")
            try:
                pixels = load(content).get("edited")
            except SafetensorError as exc:
                raise ValueError(f"Invalid safetensors payload in {member}: {exc}") from exc
            require(isinstance(pixels, torch.Tensor) and pixels.dtype == torch.uint8
                    and tuple(pixels.shape) == (len(ordered), 3, height, width), "Invalid edited RGB tensor")
            for record in records:
                pending_records.append(record)
                pending_pixels.append(pixels[record["tensor_index"]])
                if len(pending_records) == batch_size:
                    yield pending_records, torch.stack(pending_pixels)
                    pending_records, pending_pixels = [], []
        if pending_records:
            yield pending_records, torch.stack(pending_pixels)

    def provenance(self) -> dict[str, Any]:
        status = self.path.stat()
        require((status.st_size, status.st_mtime_ns) == self._snapshot, "Archive changed during evaluation")
        return {"path": str(self.path), "filename": self.path.name, "size_bytes": status.st_size,
                "modified_ns": status.st_mtime_ns, "sweep": self.manifest,
                "member_sha256": dict(self.hashes)}
