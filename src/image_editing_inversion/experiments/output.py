"""Persist run snapshots, inversion artifact paths, images, and records."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

from ..config import ExperimentConfig

if TYPE_CHECKING:
    from PIL import Image


class RunOutput:
    """Own the directory layout and writes for one experiment run."""

    def __init__(
        self,
        output_root: Path,
        config: ExperimentConfig,
        dataset_ref: str,
        dataset_fingerprint: str,
        dataset_location: str,
    ) -> None:
        self.dataset_ref = dataset_ref
        self.dataset_fingerprint = dataset_fingerprint
        self.dataset_location = dataset_location
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        self.output_dir = output_root.expanduser().resolve() / run_id
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.results_path = self.output_dir / "results.jsonl"
        snapshot = {
            "config": config.to_dict(),
            "dataset_ref": self.dataset_ref,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_location": self.dataset_location,
        }
        (self.output_dir / "resolved-config.json").write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def record(self, values: Mapping[str, Any]) -> None:
        record = {
            "dataset_ref": self.dataset_ref,
            "dataset_fingerprint": self.dataset_fingerprint,
            "settings_file": "resolved-config.json",
            **values,
        }
        with self.results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _sample_name(method_id: str, uid: str, ordinal: int) -> str:
        digest = hashlib.sha256(f"{method_id}\0{uid}".encode("utf-8")).hexdigest()[:12]
        return f"{ordinal:06d}-{digest}"

    def artifact_path(self, method_id: str, uid: str, ordinal: int) -> Path:
        return self.output_dir / "artifacts" / self._sample_name(method_id, uid, ordinal)

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
