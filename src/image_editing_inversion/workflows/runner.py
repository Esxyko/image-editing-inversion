"""Coordinate dataset samples, inversion artifacts, and shared editing runs."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import inspect
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..artifacts import ArtifactCatalog, ArtifactCompatibilityError, InversionArtifact
from ..config import ExperimentConfig, discover_pipeline_h_params, load_config
from ..data import DatasetRepository
from ..generation import Editor, PromptToPrompt
from ..inversion import InversionContext, InversionMethod, discover_methods, get_method
from .output import RunOutput, SweepOutput


@dataclass(slots=True)
class _WorkItem:
    artifact: InversionArtifact
    artifact_path: Path
    sample: Mapping[str, Any]
    ordinal: int
    inversion_seconds: float | None = None
    artifact_reused: bool = True


def _chunks(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class WorkflowRunner:
    """Run parameter files sequentially with one dataset and model runtime."""

    def __init__(
        self,
        dataset_path: Path | None,
        output_root: Path,
        *,
        pipeline_h_params_file: str | Path | None = None,
    ) -> None:
        self._parameter_configs = tuple(
            (path, load_config(path))
            for path in discover_pipeline_h_params(pipeline_h_params_file)
        )
        self.config: ExperimentConfig = self._parameter_configs[0][1]
        for _, config in self._parameter_configs:
            config.validate_hardware()
        self._repository = DatasetRepository(dataset_path)
        self.dataset = self._repository.dataset
        self.dataset_ref = self._repository.dataset_ref
        self.dataset_location = self._repository.dataset_location
        self.dataset_fingerprint = self._repository.dataset_fingerprint
        self._sweep_output = SweepOutput(
            output_root, [path for path, _ in self._parameter_configs]
        )
        self.output_dir = self._sweep_output.output_dir
        self.results_path = self.output_dir / self._parameter_configs[0][0].name / "results.jsonl"
        self._output: RunOutput | None = None
        self._editor: Editor | None = None
        self._catalog: ArtifactCatalog | None = None
        self._fingerprints: dict[str, dict[str, Any]] = {}
        self._next_ordinal = 0

    @property
    def _run_output(self) -> RunOutput:
        if self._output is None:
            raise RuntimeError("No parameter-file run is active")
        return self._output

    def _activate_configuration(self, path: Path, config: ExperimentConfig) -> None:
        self._output = RunOutput(
            self.output_dir / path.name, config, self.dataset_ref,
            self.dataset_fingerprint, self.dataset_location, path,
        )
        self.results_path = self._output.results_path
        self._next_ordinal = 0
        if self._editor is not None:
            self._editor.reconfigure(config)
        self.config = config

    def _sweep(self, action: Callable[[], None]) -> Path:
        completed = 0
        for path, config in self._parameter_configs:
            try:
                self._activate_configuration(path, config)
                self._sweep_output.update(path.name, "running", self._run_output.counts)
                action()
            except Exception as exc:
                counts = (self._output.counts if self._output is not None
                          and self._output.pipeline_h_params_file == path.name
                          else {"ok": 0, "skipped": 0, "error": 0})
                self._sweep_output.update(path.name, "error", counts, str(exc))
                self._sweep_output.finish("error", str(exc))
                raise
            counts = self._run_output.counts
            completed += counts["ok"]
            self._sweep_output.update(
                path.name, "ok" if counts["ok"] else "skipped", counts
            )
        if not completed:
            message = f"No compatible artifact/file pairs produced images; see {self.output_dir}"
            self._sweep_output.finish("error", message)
            raise ValueError(message)
        self._sweep_output.finish("ok")
        return self.output_dir

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
        return self._repository.sample(uid)

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
        self._repository.validate_identity(
            artifact.dataset_ref, artifact.dataset_fingerprint, artifact.sample_uid
        )

    def _record(self, values: Mapping[str, Any]) -> None:
        self._run_output.record(values)

    def _cache_inputs(self, method: InversionMethod, uid: str) -> dict[str, Any] | None:
        parameters = method.inversion_cache_parameters(self.inversion_context)
        if parameters is None:
            return None
        if not isinstance(parameters, Mapping):
            raise TypeError("inversion_cache_parameters must return a mapping or None")
        if method.method_id not in self._fingerprints:
            generation_dir = Path(inspect.getfile(Editor)).parent
            sources = {
                "method": Path(inspect.getfile(type(method))),
                "base": Path(inspect.getfile(InversionMethod)),
                "context": Path(inspect.getfile(InversionContext)),
                "hooks": Path(inspect.getfile(InversionMethod)).with_name("hooks.py"),
                "editor": generation_dir / "editor.py",
                "runtime": generation_dir / "runtime.py",
                "artifact": Path(inspect.getfile(InversionArtifact)),
            }
            self._fingerprints[method.method_id] = {
                "method_class": f"{type(method).__module__}.{type(method).__qualname__}",
                "sources": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in sources.items()
                },
                "dependencies": {
                    package: version(package)
                    for package in ("torch", "diffusers", "transformers", "accelerate",
                                    "pillow", "safetensors")
                },
            }
        return {
            "cache_version": 1,
            "dataset_ref": self.dataset_ref,
            "dataset_fingerprint": self.dataset_fingerprint,
            "sample_uid": uid,
            "method_id": method.method_id,
            "model": asdict(self.config.model),
            "scheduler": {
                "id": "ddim", "config": self.editor.scheduler_config,
                "timesteps": list(self.editor.expected_timesteps),
                "eta": self.config.sampling.eta,
            },
            "guidance_scale": self.config.sampling.guidance_scale,
            "method_parameters": dict(parameters),
            "numerics": self.editor.inversion_cache_settings(),
            "implementation": self._fingerprints[method.method_id],
        }

    def _validate_inversion(
        self, artifact: InversionArtifact, method: InversionMethod, uid: str
    ) -> None:
        if not isinstance(artifact, InversionArtifact):
            raise TypeError(f"Method {method.method_id!r} must return InversionArtifact")
        if artifact.method_id != method.method_id or artifact.sample_uid != uid:
            raise ValueError(
                f"Method {method.method_id!r} returned an artifact for a different method or UID"
            )
        self._check_artifact_identity(artifact)
        method.validate_replay(artifact, self.inversion_context)
        self.editor.validate_artifact(artifact)

    def _invert_or_reuse(
        self, method: InversionMethod, sample: Mapping[str, Any], uid: str
    ) -> tuple[InversionArtifact, Path, bool]:
        if self._catalog is None:
            raise RuntimeError("The inversion artifact catalog is not initialized")
        inputs = self._cache_inputs(method, uid)
        cached = None if inputs is None else self._catalog.lookup(inputs)
        if cached is not None:
            artifact, path = cached
            try:
                self._validate_inversion(artifact, method, uid)
            except (ValueError, TypeError, KeyError):
                pass
            else:
                return artifact, path, True
        artifact = method.invert(sample, self.inversion_context)
        self._validate_inversion(artifact, method, uid)
        return artifact, self._catalog.store(artifact, inputs), False

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
        method.validate_replay(artifact, self.inversion_context)
        self.editor.validate_artifact(artifact)
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
        if not items:
            return
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
                reconstructed_path, edited_path = self._run_output.save_images(
                    item.artifact.method_id,
                    item.artifact.sample_uid,
                    item.ordinal,
                    result.reconstructed,
                    result.edited,
                )
                records.append(
                    {
                        "uid": item.artifact.sample_uid,
                        "method_id": item.artifact.method_id,
                        "artifact": str(item.artifact_path),
                        "reconstructed_image": str(reconstructed_path.relative_to(self._run_output.output_dir)),
                        "edited_image": str(edited_path.relative_to(self._run_output.output_dir)),
                        "artifact_reused": item.artifact_reused,
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
                        "artifact_reused": item.artifact_reused,
                        "inversion_seconds": item.inversion_seconds,
                        "edit_batch_seconds": batch_seconds,
                        "status": "error",
                        "error": str(exc),
                    }
                )
            raise

    def edit_artifacts(self, artifact_paths: Sequence[Path]) -> Path:
        """Replay each parameter file, recording incompatible pairs as skipped."""
        discover_methods()
        return self._sweep(lambda: self._edit_artifacts_for_config(artifact_paths))

    def _edit_artifacts_for_config(self, artifact_paths: Sequence[Path]) -> None:
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
                            "artifact_reused": True,
                            "status": "error",
                            "error": str(loaded),
                        }
                    )
                    raise loaded
                artifact, artifact_path, sample = loaded
                try:
                    method = self._method_for(artifact)
                    self._check_method_batch_size(method)
                    self._prompt_to_prompt_for(method)
                    method.validate_replay(artifact, self.inversion_context)
                    self.editor.validate_artifact(artifact)
                except ArtifactCompatibilityError as exc:
                    self._record({
                        "uid": artifact.sample_uid,
                        "method_id": artifact.method_id,
                        "artifact": str(artifact_path),
                        "artifact_reused": True,
                        "status": "skipped",
                        "reason": str(exc),
                    })
                    continue
                except Exception as exc:
                    self._record({
                        "uid": artifact.sample_uid,
                        "method_id": artifact.method_id,
                        "artifact": str(artifact_path),
                        "artifact_reused": True,
                        "status": "error",
                        "error": str(exc),
                    })
                    raise
                items.append(
                    _WorkItem(
                        artifact=artifact,
                        artifact_path=artifact_path,
                        sample=sample,
                        ordinal=self._allocate_ordinal(),
                    )
                )
            self._edit_batch(items)

    def run_methods(self, method_ids: Sequence[str], uids: Sequence[str]) -> Path:
        """Invert/reuse and edit all requested records for each parameter file."""
        discover_methods()
        methods = [(method_id, get_method(method_id)) for method_id in method_ids]
        for _, method in methods:
            self._check_method_batch_size(method)
            self._prompt_to_prompt_for(method)
        if self._catalog is None:
            self._catalog = ArtifactCatalog()
        return self._sweep(lambda: self._run_methods_for_config(methods, uids))

    def _run_methods_for_config(
        self, methods: Sequence[tuple[str, InversionMethod]], uids: Sequence[str]
    ) -> None:
        for method_id, method in methods:
            for uid_batch in _chunks(uids, self.config.runtime.batch_size):
                items: list[_WorkItem] = []
                samples = self._fetch_many(self._sample, uid_batch)
                for uid, sample in zip(uid_batch, samples, strict=True):
                    started = time.perf_counter()
                    artifact_path: Path | None = None
                    artifact_reused = False
                    try:
                        artifact, artifact_path, artifact_reused = self._invert_or_reuse(
                            method, sample, uid
                        )
                    except Exception as exc:
                        self._record(
                            {
                                "uid": uid,
                                "method_id": method_id,
                                "artifact": str(artifact_path) if artifact_path else None,
                                "artifact_reused": artifact_reused,
                                "inversion_seconds": time.perf_counter() - started,
                                "status": "error",
                                "error": str(exc),
                            }
                        )
                        raise
                    inversion_seconds = (0.0 if artifact_reused
                                         else time.perf_counter() - started)
                    items.append(
                        _WorkItem(
                            artifact=artifact,
                            artifact_path=artifact_path,
                            sample=sample,
                            ordinal=self._allocate_ordinal(),
                            inversion_seconds=inversion_seconds,
                            artifact_reused=artifact_reused,
                        )
                    )
                self._edit_batch(items)


def edit_artifacts(
    dataset_path: Path | None,
    artifact_paths: Sequence[Path],
    output_root: Path,
    *,
    pipeline_h_params_file: str | Path | None = None,
) -> Path:
    return WorkflowRunner(
        dataset_path, output_root, pipeline_h_params_file=pipeline_h_params_file
    ).edit_artifacts(
        artifact_paths
    )


def run_methods(
    dataset_path: Path | None,
    method_ids: Sequence[str],
    uids: Sequence[str],
    output_root: Path,
    *,
    pipeline_h_params_file: str | Path | None = None,
) -> Path:
    discover_methods()
    for method_id in method_ids:
        get_method(method_id)
    return WorkflowRunner(
        dataset_path, output_root, pipeline_h_params_file=pipeline_h_params_file
    ).run_methods(
        method_ids, uids
    )
