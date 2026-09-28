"""Coordinate dataset samples, inversion artifacts, and shared editing runs."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from datasets import Dataset, load_from_disk

from .artifact import InversionArtifact, InversionMethod, discover_methods, get_method
from .config import ExperimentConfig, load_config
from .dataset import load_project_dataset
from .editing import Editor, PromptToPrompt


_HUB_DATASET_REF = "beatle-ju1ce/image-editing-inversion"
_LOCAL_DATASET_REF = "local:image-editing"
_REQUIRED_COLUMNS = {
    "uid",
    "source_img",
    "mask_img",
    "source_prompt",
    "target_prompt",
}


@dataclass(slots=True)
class _WorkItem:
    artifact: InversionArtifact
    artifact_path: Path
    sample: Mapping[str, Any]
    ordinal: int
    inversion_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class InversionContext:
    """Shared model runtime and dataset identity passed to method adapters."""

    editor: Editor
    config: ExperimentConfig
    dataset_ref: str
    dataset_fingerprint: str


def _chunks(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class ExperimentRunner:
    """Run a fixed experiment configuration over selected dataset records."""

    def __init__(
        self,
        config_path: Path,
        dataset_path: Path | None,
        output_root: Path,
    ) -> None:
        self.config: ExperimentConfig = load_config(config_path)
        self.config.validate_hardware()
        if dataset_path is None:
            self.dataset = load_project_dataset()
            self.dataset_ref = _HUB_DATASET_REF
            self.dataset_location = _HUB_DATASET_REF
        else:
            resolved_path = dataset_path.expanduser().resolve()
            self.dataset = load_from_disk(str(resolved_path))
            self.dataset_ref = _LOCAL_DATASET_REF
            self.dataset_location = resolved_path.as_posix()
        if not isinstance(self.dataset, Dataset):
            raise ValueError("The dataset source must contain one Hugging Face Dataset.")
        if set(self.dataset.column_names) != _REQUIRED_COLUMNS:
            raise ValueError("The dataset does not have the project's five expected columns.")
        self.dataset_fingerprint = str(getattr(self.dataset, "_fingerprint", ""))
        if not self.dataset_fingerprint:
            raise ValueError("The dataset has no fingerprint for artifact validation.")
        uids = self.dataset["uid"]
        self._uid_to_index = {uid: index for index, uid in enumerate(uids)}
        if len(self._uid_to_index) != len(uids):
            raise ValueError("The dataset contains duplicate UIDs.")

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        self.output_dir = output_root.expanduser().resolve() / run_id
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.results_path = self.output_dir / "results.jsonl"
        snapshot = {
            "config": self.config.to_dict(),
            "dataset_ref": self.dataset_ref,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_location": self.dataset_location,
        }
        (self.output_dir / "resolved-config.json").write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._editor: Editor | None = None
        self._next_ordinal = 0

    @property
    def editor(self) -> Editor:
        if self._editor is None:
            self._editor = Editor(self.config)
        return self._editor

    @property
    def inversion_context(self) -> InversionContext:
        return InversionContext(
            editor=self.editor,
            config=self.config,
            dataset_ref=self.dataset_ref,
            dataset_fingerprint=self.dataset_fingerprint,
        )

    def _allocate_ordinal(self) -> int:
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        return ordinal

    def _sample(self, uid: str) -> Mapping[str, Any]:
        try:
            return self.dataset[self._uid_to_index[uid]]
        except KeyError as exc:
            raise ValueError(f"Dataset UID not found: {uid}") from exc

    def _fetch_many(self, function: Any, values: Sequence[Any]) -> list[Any]:
        """Use the configured number of threads for dataset/artifact reads."""
        if self.config.runtime.num_workers == 0:
            return [function(value) for value in values]
        with ThreadPoolExecutor(max_workers=self.config.runtime.num_workers) as pool:
            return list(pool.map(function, values))

    def _load_artifact(self, path: Path) -> tuple[InversionArtifact, Path, Mapping[str, Any]]:
        resolved = path.expanduser().resolve()
        artifact = InversionArtifact.load(resolved)
        self._check_artifact_identity(artifact)
        return artifact, resolved, self._sample(artifact.sample_uid)

    def _load_artifact_result(
        self, path: Path
    ) -> tuple[InversionArtifact, Path, Mapping[str, Any]] | Exception:
        try:
            return self._load_artifact(path)
        except Exception as exc:
            return exc

    def _check_artifact_identity(self, artifact: InversionArtifact) -> None:
        if artifact.dataset_ref != self.dataset_ref:
            raise ValueError(
                f"Artifact dataset {artifact.dataset_ref!r} differs from "
                f"selected dataset {self.dataset_ref!r}."
            )
        if artifact.dataset_fingerprint != self.dataset_fingerprint:
            raise ValueError("Artifact dataset fingerprint differs from selected dataset.")
        if artifact.sample_uid not in self._uid_to_index:
            raise ValueError(f"Artifact UID not found in dataset: {artifact.sample_uid}")

    def _record(self, values: Mapping[str, Any]) -> None:
        record = {
            "dataset_ref": self.dataset_ref,
            "dataset_fingerprint": self.dataset_fingerprint,
            "settings_file": "resolved-config.json",
            **values,
        }
        with self.results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _image_dir(self, item: _WorkItem) -> Path:
        digest = hashlib.sha256(
            f"{item.artifact.method_id}\0{item.artifact.sample_uid}".encode("utf-8")
        ).hexdigest()[:12]
        return self.output_dir / "images" / f"{item.ordinal:06d}-{digest}"

    def _method_for(self, artifact: InversionArtifact) -> InversionMethod:
        try:
            return get_method(artifact.method_id)
        except KeyError as exc:
            raise ValueError(
                f"Artifact method {artifact.method_id!r} is not registered; "
                "install its adapter before replay"
            ) from exc

    def _prompt_to_prompt_for(
        self, method: InversionMethod
    ) -> tuple[type[PromptToPrompt], dict[str, Any]]:
        selected = method.prompt_to_prompt_class
        policy_class = PromptToPrompt if selected is None else selected
        if not isinstance(policy_class, type) or not issubclass(
            policy_class, PromptToPrompt
        ):
            raise TypeError(
                f"Method {method.method_id!r} prompt_to_prompt_class must subclass PromptToPrompt"
            )
        try:
            settings = policy_class.validate_method_settings(
                self.config.method_prompt_to_prompt(method.method_id)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid Prompt-to-Prompt settings for method {method.method_id!r}: {exc}"
            ) from exc
        if not isinstance(settings, Mapping) or any(
            not isinstance(key, str) for key in settings
        ):
            raise TypeError(
                f"Method {method.method_id!r} returned invalid Prompt-to-Prompt settings"
            )
        try:
            normalized = json.loads(json.dumps(dict(settings), sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Method {method.method_id!r} Prompt-to-Prompt settings must be JSON-compatible"
            ) from exc
        return policy_class, normalized

    def _hook_for(self, method: InversionMethod, artifact: InversionArtifact) -> Any:
        self._check_method_batch_size(method)
        hook = method.create_denoising_hook(artifact)
        if artifact.per_step_state and hook is None:
            raise ValueError(
                f"Artifact method {artifact.method_id!r} needs a denoising hook."
            )
        return hook

    def _check_method_batch_size(self, method: InversionMethod) -> None:
        limit = method.max_edit_batch_size
        if limit is None:
            return
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(
                f"Method {method.method_id!r} has an invalid max_edit_batch_size"
            )
        if self.config.runtime.batch_size > limit:
            raise ValueError(
                f"Method {method.method_id!r} supports edit batches of at most "
                f"{limit}; configured batch_size is {self.config.runtime.batch_size}"
            )

    def _edit_batch(self, items: list[_WorkItem]) -> None:
        started = time.perf_counter()
        try:
            discover_methods()
            methods = [self._method_for(item.artifact) for item in items]
            selections = [self._prompt_to_prompt_for(method) for method in methods]
            policy_class, settings = selections[0]
            if any(
                selected_class is not policy_class or selected_settings != settings
                for selected_class, selected_settings in selections[1:]
            ):
                raise ValueError(
                    "A GPU edit batch cannot mix Prompt-to-Prompt subclasses or method settings"
                )
            hooks = [
                self._hook_for(method, item.artifact)
                for method, item in zip(methods, items, strict=True)
            ]
            edited = self.editor.edit_batch(
                [item.sample for item in items],
                [item.artifact for item in items],
                hooks=hooks,
                prompt_to_prompt_class=policy_class,
                method_settings=settings,
            )
            if len(edited) != len(items):
                raise RuntimeError("The editor returned a different number of results.")
            batch_seconds = time.perf_counter() - started
            records: list[dict[str, Any]] = []
            for item, result in zip(items, edited, strict=True):
                image_dir = self._image_dir(item)
                image_dir.mkdir(parents=True, exist_ok=False)
                reconstructed_path = image_dir / "reconstructed.png"
                edited_path = image_dir / "edited.png"
                result.reconstructed.save(reconstructed_path, format="PNG")
                result.edited.save(edited_path, format="PNG")
                records.append(
                    {
                        "uid": item.artifact.sample_uid,
                        "method_id": item.artifact.method_id,
                        "artifact": str(item.artifact_path),
                        "reconstructed_image": str(reconstructed_path.relative_to(self.output_dir)),
                        "edited_image": str(edited_path.relative_to(self.output_dir)),
                        "inversion_seconds": item.inversion_seconds,
                        "edit_batch_seconds": batch_seconds,
                        "status": "ok",
                    }
                )
            for record in records:
                self._record(record)
        except Exception as exc:
            batch_seconds = time.perf_counter() - started
            for item in items:
                self._record(
                    {
                        "uid": item.artifact.sample_uid,
                        "method_id": item.artifact.method_id,
                        "artifact": str(item.artifact_path),
                        "inversion_seconds": item.inversion_seconds,
                        "edit_batch_seconds": batch_seconds,
                        "status": "error",
                        "error": str(exc),
                    }
                )
            raise

    def edit_artifacts(self, artifact_paths: Sequence[Path]) -> Path:
        """Replay saved artifacts in config-sized GPU batches."""
        for path_batch in _chunks(artifact_paths, self.config.runtime.batch_size):
            items: list[_WorkItem] = []
            loaded_batch = self._fetch_many(self._load_artifact_result, path_batch)
            for requested_path, loaded in zip(path_batch, loaded_batch, strict=True):
                if isinstance(loaded, Exception):
                    self._record(
                        {
                            "uid": None,
                            "method_id": None,
                            "artifact": str(requested_path.expanduser().resolve()),
                            "status": "error",
                            "error": str(loaded),
                        }
                    )
                    raise loaded
                artifact, artifact_path, sample = loaded
                items.append(
                    _WorkItem(
                        artifact=artifact,
                        artifact_path=artifact_path,
                        sample=sample,
                        ordinal=self._allocate_ordinal(),
                    )
                )
            self._edit_batch(items)
        return self.output_dir

    def run_methods(self, method_ids: Sequence[str], uids: Sequence[str]) -> Path:
        """Invert selected records with future registered methods, then edit them."""
        discover_methods()
        methods = [(method_id, get_method(method_id)) for method_id in method_ids]
        for _, method in methods:
            self._check_method_batch_size(method)
            self._prompt_to_prompt_for(method)
        for method_id, method in methods:
            for uid_batch in _chunks(uids, self.config.runtime.batch_size):
                items: list[_WorkItem] = []
                samples = self._fetch_many(self._sample, uid_batch)
                for uid, sample in zip(uid_batch, samples, strict=True):
                    started = time.perf_counter()
                    artifact_path: Path | None = None
                    try:
                        artifact = method.invert(sample, self.inversion_context)
                        if not isinstance(artifact, InversionArtifact):
                            raise TypeError(
                                f"Method {method_id!r} must return InversionArtifact"
                            )
                        if artifact.method_id != method_id or artifact.sample_uid != uid:
                            raise ValueError(
                                f"Method {method_id!r} returned an artifact for a "
                                "different method or UID."
                            )
                        self._check_artifact_identity(artifact)
                        digest = hashlib.sha256(
                            f"{method_id}\0{uid}".encode()
                        ).hexdigest()[:12]
                        artifact_path = (
                            self.output_dir
                            / "artifacts"
                            / f"{self._next_ordinal:06d}-{digest}"
                        )
                        artifact.save(artifact_path)
                    except Exception as exc:
                        self._record(
                            {
                                "uid": uid,
                                "method_id": method_id,
                                "artifact": str(artifact_path) if artifact_path else None,
                                "inversion_seconds": time.perf_counter() - started,
                                "status": "error",
                                "error": str(exc),
                            }
                        )
                        raise
                    inversion_seconds = time.perf_counter() - started
                    items.append(
                        _WorkItem(
                            artifact=artifact,
                            artifact_path=artifact_path,
                            sample=sample,
                            ordinal=self._allocate_ordinal(),
                            inversion_seconds=inversion_seconds,
                        )
                    )
                self._edit_batch(items)
        return self.output_dir


def edit_artifacts(
    config_path: Path,
    dataset_path: Path | None,
    artifact_paths: Sequence[Path],
    output_root: Path,
) -> Path:
    return ExperimentRunner(config_path, dataset_path, output_root).edit_artifacts(
        artifact_paths
    )


def run_methods(
    config_path: Path,
    dataset_path: Path | None,
    method_ids: Sequence[str],
    uids: Sequence[str],
    output_root: Path,
) -> Path:
    discover_methods()
    for method_id in method_ids:
        get_method(method_id)
    return ExperimentRunner(config_path, dataset_path, output_root).run_methods(
        method_ids, uids
    )
