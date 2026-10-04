"""Persist sweep manifests, parameter-file snapshots, images, and records."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence
from uuid import uuid4

from ..config import ExperimentConfig

if TYPE_CHECKING:
    from PIL import Image


def _write_json(path: Path, values: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(values), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class SweepOutput:
    """Own one invocation's parent directory and parameter-file manifest."""

    def __init__(self, output_root: Path, parameter_files: Sequence[Path]) -> None:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        self.output_dir = output_root.expanduser().resolve() / run_id
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "pending",
            "parameter_files": [
                {"pipeline_h_params_file": path.name, "directory": path.name,
                 "status": "pending", "counts": {"ok": 0, "skipped": 0, "error": 0}}
                for path in parameter_files
            ],
        }
        self._write()

    def _write(self) -> None:
        _write_json(self.output_dir / "sweep.json", self._manifest)

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
    """Own snapshots, images, and records for one parameter-file child."""

    def __init__(
        self,
        output_dir: Path,
        config: ExperimentConfig,
        dataset_ref: str,
        dataset_fingerprint: str,
        dataset_location: str,
        pipeline_h_params_file: Path,
    ) -> None:
        self.dataset_ref = dataset_ref
        self.dataset_fingerprint = dataset_fingerprint
        self.dataset_location = dataset_location
        self.pipeline_h_params_file = pipeline_h_params_file.name
        self.counts = {"ok": 0, "skipped": 0, "error": 0}
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.results_path = self.output_dir / "results.jsonl"
        self.results_path.touch(exist_ok=False)
        snapshot = {
            "config": config.to_dict(),
            "pipeline_h_params_file": self.pipeline_h_params_file,
            "dataset_ref": self.dataset_ref,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_location": self.dataset_location,
        }
        _write_json(self.output_dir / "resolved-config.json", snapshot)

    def record(self, values: Mapping[str, Any]) -> None:
        record = {
            "dataset_ref": self.dataset_ref,
            "dataset_fingerprint": self.dataset_fingerprint,
            "settings_file": "resolved-config.json",
            "pipeline_h_params_file": self.pipeline_h_params_file,
            **values,
        }
        with self.results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        status = record.get("status")
        if status in self.counts:
            self.counts[status] += 1

    @staticmethod
    def _sample_name(method_id: str, uid: str, ordinal: int) -> str:
        digest = hashlib.sha256(f"{method_id}\0{uid}".encode("utf-8")).hexdigest()[:12]
        return f"{ordinal:06d}-{digest}"

    def save_images(
        self,
        method_id: str,
        uid: str,
        ordinal: int,
        reconstructed: Image.Image,
        edited: Image.Image,
    ) -> tuple[Path, Path]:
        image_dir = self.output_dir / "images" / self._sample_name(method_id, uid, ordinal)
        image_dir.mkdir(parents=True, exist_ok=False)
        reconstructed_path = image_dir / "reconstructed.png"
        edited_path = image_dir / "edited.png"
        reconstructed.save(reconstructed_path, format="PNG")
        edited.save(edited_path, format="PNG")
        return reconstructed_path, edited_path
