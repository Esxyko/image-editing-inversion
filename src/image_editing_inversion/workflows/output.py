"""Persist sweep manifests and per-parameter/method snapshots, images, and records."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence
from uuid import uuid4

from .._project import project_paths
from ..artifacts.layout import method_directory_name
from ..config import ExperimentConfig
from .timing import BatchTiming

if TYPE_CHECKING:
    from PIL import Image


def _write_json(path: Path, values: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(values), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except BaseException as exc:
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup:
            exc.add_note(f"Temporary output cleanup failed: {cleanup}")
        raise


class SweepOutput:
    """Own one invocation's directory and parameter/method manifest."""

    def __init__(self, parameter_files: Sequence[Path], *, dataset: str | None = None) -> None:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        self.output_dir = project_paths().output.resolve() / run_id
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._manifest: dict[str, Any] = {
            "schema_version": 2,
            "dataset": dataset,
            "status": "pending",
            "parameter_files": [
                {"pipeline_h_params_file": path.name, "directory": path.name,
                 "status": "pending", "counts": {"ok": 0, "skipped": 0, "error": 0},
                 "methods": []}
                for path in parameter_files
            ],
        }
        self._write()

    def _write(self) -> None:
        _write_json(self.output_dir / "sweep.json", self._manifest)

    @property
    def status(self) -> str:
        return self._manifest["status"]

    def fail(
        self, error: str, *, filename: str | None = None, method_id: str | None = None,
    ) -> None:
        """Finalize affected pending/running combinations after a setup failure."""
        for parameter in self._manifest["parameter_files"]:
            if filename is not None and parameter["pipeline_h_params_file"] != filename:
                continue
            affected = False
            for method in parameter["methods"]:
                if method_id is not None and method["method_id"] != method_id:
                    continue
                if method["status"] in {"pending", "running"}:
                    method.update(status="error", error=error)
                    affected = True
            if affected:
                parameter.update(status="error", error=error)
        self.finish("error", error)

    def set_methods(self, method_ids: Sequence[str]) -> None:
        """Declare every combination before any method starts running."""
        names = [method_directory_name(method_id) for method_id in method_ids]
        for entry in self._manifest["parameter_files"]:
            entry["methods"] = [
                {"method_id": method_id,
                 "directory": (Path(entry["directory"]) / method_id).as_posix(),
                 "status": "pending", "counts": {"ok": 0, "skipped": 0, "error": 0}}
                for method_id in names
            ]
        self._write()

    def update_method(
        self, filename: str, method_id: str, status: str,
        counts: Mapping[str, int], error: str | None = None,
    ) -> None:
        parameter = next(item for item in self._manifest["parameter_files"]
                         if item["pipeline_h_params_file"] == filename)
        entry = next(item for item in parameter["methods"] if item["method_id"] == method_id)
        entry.update(status=status, counts=dict(counts))
        if error is not None:
            entry["error"] = error
        parameter["counts"] = {
            key: sum(item["counts"][key] for item in parameter["methods"])
            for key in ("ok", "skipped", "error")
        }
        parameter["status"] = "running"
        self._manifest["status"] = "running"
        self._write()

    def update(
        self, filename: str, status: str, counts: Mapping[str, int], error: str | None = None
    ) -> None:
        entry = next(item for item in self._manifest["parameter_files"]
                     if item["pipeline_h_params_file"] == filename)
        entry.update(status=status, counts=dict(counts))
        if error is not None:
            entry["error"] = error
        self._manifest["status"] = "running"
        self._write()

    def finish(self, status: str, error: str | None = None) -> None:
        self._manifest["status"] = status
        if error is not None:
            self._manifest["error"] = error
        self._write()


class RunOutput:
    """Own snapshots, images, and records for one parameter/method combination."""

    def __init__(
        self,
        output_dir: Path,
        config: ExperimentConfig,
        dataset_ref: str,
        dataset_fingerprint: str,
        dataset_location: str,
        pipeline_h_params_file: Path,
        method_id: str,
    ) -> None:
        self.dataset_ref = dataset_ref
        self.dataset_fingerprint = dataset_fingerprint
        self.dataset_location = dataset_location
        self.pipeline_h_params_file = pipeline_h_params_file.name
        self.method_id = method_directory_name(method_id)
        self.counts = {"ok": 0, "skipped": 0, "error": 0}
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.results_path = self.output_dir / "results.jsonl"
        self.results_path.touch(exist_ok=False)
        self.batches_path = self.output_dir / "batches.jsonl"
        self.batches_path.touch(exist_ok=False)
        snapshot = {
            "config": config.to_dict(),
            "pipeline_h_params_file": self.pipeline_h_params_file,
            "method_id": self.method_id,
            "dataset_ref": self.dataset_ref,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_location": self.dataset_location,
        }
        _write_json(self.output_dir / "resolved-config.json", snapshot)

    def record(self, values: Mapping[str, Any]) -> None:
        if values.get("method_id", self.method_id) != self.method_id:
            raise ValueError("Result method must match its output directory")
        record = {
            "dataset_ref": self.dataset_ref,
            "dataset_fingerprint": self.dataset_fingerprint,
            "settings_file": "resolved-config.json",
            "pipeline_h_params_file": self.pipeline_h_params_file,
            "method_id": self.method_id,
            "inversion_batch_id": None,
            "edit_batch_id": None,
            **values,
        }
        with self.results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        status = record.get("status")
        if status in self.counts:
            self.counts[status] += 1

    def record_error(self, values: Mapping[str, Any], error: BaseException) -> None:
        """Attempt an error record without replacing the failure being reported."""
        try:
            self.record({**values, "status": "error", "error": str(error)})
        except BaseException as recording:
            error.add_note(f"Could not record result for {values.get('uid')!r}: {recording}")

    @contextmanager
    def batch(self, phase: str, sample_count: int) -> Iterator[BatchTiming]:
        """Persist each attempted batch once without changing sample counts."""
        timing = BatchTiming(phase, sample_count)
        try:
            yield timing
        except BaseException as exc:
            try:
                self.record_batch(timing.values(exc))
            except BaseException as recording:
                exc.add_note(f"Could not record batch timing: {recording}")
            raise
        else:
            self.record_batch(timing.values())

    def record_batch(self, values: Mapping[str, Any]) -> None:
        record = {
            "method_id": self.method_id,
            "pipeline_h_params_file": self.pipeline_h_params_file,
            **values,
        }
        with self.batches_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    @staticmethod
    def _sample_name(uid: str) -> str:
        """Use the project UID with colons replaced for Windows paths."""
        return uid.replace(":", "_")

    def save_images(
        self,
        method_id: str,
        uid: str,
        reconstructed: Image.Image,
        edited: Image.Image,
    ) -> tuple[Path, Path]:
        if method_id != self.method_id:
            raise ValueError("Image method must match its output directory")
        image_dir = self.output_dir / "images" / self._sample_name(uid)
        image_dir.mkdir(parents=True, exist_ok=False)
        reconstructed_path = image_dir / "reconstructed.png"
        edited_path = image_dir / "edited.png"
        reconstructed.save(reconstructed_path, format="PNG")
        edited.save(edited_path, format="PNG")
        return reconstructed_path, edited_path
