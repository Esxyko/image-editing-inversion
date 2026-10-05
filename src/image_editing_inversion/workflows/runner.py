"""Schedule parameter sweeps, inversion batches, and prepared editing batches."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..artifacts import (
    ArtifactCatalog, ArtifactCompatibilityError, ArtifactReference,
    ArtifactRepository, ArtifactSettings, IntermediateCache,
)
from ..artifacts.layout import method_directory_name
from ..config import ConfigError, ExperimentConfig, discover_pipeline_h_params, load_config
from ..data import DatasetRepository
from ..generation import Editor
from ..inversion import InversionContext, InversionMethod, discover_methods, get_method
from ..inversion.methods.common import SharedInversionComponents
from .inversion import InversionCoordinator
from .models import InversionRecord, WorkItem
from .output import RunOutput, SweepOutput
from .replay import ReplayPreparer
from .timing import BatchTiming


def _chunks(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _resolve_method_ids(method_ids: Sequence[str]) -> tuple[str, ...]:
    """Expand ALL and validate each unique method before workflow setup."""
    if not method_ids:
        raise ValueError("At least one inversion method is required.")
    discovered = discover_methods()
    resolved: dict[str, None] = {}
    for selector in method_ids:
        selected = discovered if selector == "ALL" else (selector,)
        for method_id in selected:
            method_directory_name(method_id)
            get_method(method_id)
            resolved[method_id] = None
    if not resolved:
        raise ValueError("No inversion methods are registered.")
    return tuple(resolved)


class WorkflowRunner:
    """Run parameter files sequentially with one dataset and model runtime."""

    def __init__(
        self,
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
        self._repository = DatasetRepository(None)
        self.dataset = self._repository.dataset
        self.dataset_ref = self._repository.dataset_ref
        self.dataset_location = self._repository.dataset_location
        self.dataset_fingerprint = self._repository.dataset_fingerprint
        self._sweep_output = SweepOutput(
            Path("data/output"), [path for path, _ in self._parameter_configs]
        )
        self.output_dir = self._sweep_output.output_dir
        self.results_path: Path | None = None
        self._outputs: dict[str, RunOutput] = {}
        self._finished_methods: set[str] = set()
        self._active_method_id: str | None = None
        self._parameter_path: Path | None = None
        self._editor: Editor | None = None
        self._coordinator: InversionCoordinator | None = None
        self._components: SharedInversionComponents | None = None
        self._replay = ReplayPreparer(self._repository)

    def _activate_configuration(self, path: Path, config: ExperimentConfig) -> None:
        self._outputs = {}
        self._finished_methods = set()
        self._active_method_id = None
        self._parameter_path = path
        self.results_path = None
        if self._editor is not None:
            self._editor.reconfigure(config)
        self.config = config

    def _output_for(self, method_id: str) -> RunOutput:
        method_id = method_directory_name(method_id)
        if self._parameter_path is None:
            raise RuntimeError("No parameter-file run is active")
        self._active_method_id = method_id
        if method_id not in self._outputs:
            self._sweep_output.update_method(
                self._parameter_path.name, method_id, "running",
                {"ok": 0, "skipped": 0, "error": 0},
            )
            self._outputs[method_id] = RunOutput(
                self.output_dir / self._parameter_path.name / method_id,
                self.config, self.dataset_ref, self.dataset_fingerprint,
                self.dataset_location, self._parameter_path, method_id,
            )
        output = self._outputs[method_id]
        self.results_path = output.results_path
        return output

    def _finish_method(self, method_id: str) -> None:
        self._active_method_id = method_id
        output = self._outputs[method_id]
        self._sweep_output.update_method(
            output.pipeline_h_params_file, method_id,
            "ok" if output.counts["ok"] else "skipped", output.counts,
        )
        self._finished_methods.add(method_id)

    def _counts(self) -> dict[str, int]:
        return {key: sum(output.counts[key] for output in self._outputs.values())
                for key in ("ok", "skipped", "error")}

    def _sweep(self, action: Callable[[], None], method_ids: Sequence[str]) -> Path:
        self._sweep_output.set_methods(method_ids)
        completed = 0
        for path, config in self._parameter_configs:
            try:
                self._activate_configuration(path, config)
                self._sweep_output.update(path.name, "running", self._counts())
                action()
                for method_id in self._outputs:
                    if method_id not in self._finished_methods:
                        self._finish_method(method_id)
            except BaseException as exc:
                for method_id, output in self._outputs.items():
                    if output.counts["error"] or method_id == self._active_method_id:
                        self._sweep_output.update_method(
                            path.name, method_id, "error", output.counts, str(exc)
                        )
                    elif method_id not in self._finished_methods:
                        self._sweep_output.update_method(
                            path.name, method_id, "running", output.counts
                        )
                if self._active_method_id is not None and self._active_method_id not in self._outputs:
                    self._sweep_output.update_method(
                        path.name, self._active_method_id, "error",
                        {"ok": 0, "skipped": 0, "error": 0}, str(exc),
                    )
                self._sweep_output.update(path.name, "error", self._counts(), str(exc))
                self._sweep_output.finish("error", str(exc))
                raise
            counts = self._counts()
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
            components=self._components,
            dataset_content=self._repository.content_digest,
        )

    def _sample(self, uid: str) -> Mapping[str, Any]:
        return self._repository.sample(uid)

    def _fetch_many(self, function: Any, values: Sequence[Any]) -> list[Any]:
        """Use the configured number of threads for dataset/artifact reads."""
        if self.config.runtime.num_workers == 0:
            return [function(value) for value in values]
        with ThreadPoolExecutor(max_workers=self.config.runtime.num_workers) as pool:
            return list(pool.map(function, values))

    def _record(self, values: Mapping[str, Any]) -> None:
        self._output_for(values["method_id"]).record(values)

    @property
    def artifact_settings(self) -> ArtifactSettings:
        """Supply independent expectations for persisted production identity."""
        return ArtifactSettings(
            model_id=self.config.model.model_id,
            model_revision=self.config.model.revision,
            dataset_ref=self.dataset_ref,
            dataset_fingerprint=self.dataset_fingerprint,
            scheduler_config=self.editor.scheduler_config,
            eta=self.config.sampling.eta,
            timesteps_for_steps=self.editor.timesteps_for_steps,
            model_content=self.editor.model_content_identity,
            dataset_content=self._repository.content_digest,
            numerics=self.editor.inversion_cache_settings(),
        )

    def _preflight(self, methods: Sequence[InversionMethod]) -> None:
        """Validate every inversion combination before any sample is processed."""
        self._sweep_output.set_methods([method.method_id for method in methods])
        errors: list[tuple[Path, str, str]] = []
        for path, config in self._parameter_configs:
            for method in methods:
                for check in (
                    lambda: method.validate_inversion_config(config),
                    lambda: self._replay.check_batch_size(method, config),
                    lambda: self._replay.policy_for(method, config),
                ):
                    try:
                        check()
                    except Exception as exc:
                        errors.append((path, method.method_id, str(exc)))
        if errors:
            self._fail_preflight(errors)

        # Only the second phase constructs a model; all contexts share its weights.
        try:
            _ = self.editor
        except Exception as exc:
            self._fail_preflight([
                (path, method.method_id, str(exc))
                for path, _ in self._parameter_configs for method in methods
            ])
        for path, config in self._parameter_configs:
            try:
                self._activate_configuration(path, config)
                context = self.inversion_context
            except Exception as exc:
                errors.extend((path, method.method_id, str(exc)) for method in methods)
                continue
            for method in methods:
                try:
                    method.validate_inversion_context(context)
                except Exception as exc:
                    errors.append((path, method.method_id, str(exc)))
        if errors:
            self._fail_preflight(errors)
        first_path, first_config = self._parameter_configs[0]
        try:
            self._activate_configuration(first_path, first_config)
        except Exception as exc:
            self._fail_preflight([(first_path, method.method_id, str(exc)) for method in methods])

    def _fail_preflight(self, errors: Sequence[tuple[Path, str, str]]) -> None:
        reasons: dict[tuple[str, str], list[str]] = {}
        for path, method_id, reason in errors:
            reasons.setdefault((path.name, method_id), []).append(reason)
        counts = {"ok": 0, "skipped": 0, "error": 0}
        for (filename, method_id), messages in reasons.items():
            self._sweep_output.update_method(filename, method_id, "error", counts, "; ".join(messages))
        message = "Inversion preflight failed:\n" + "\n".join(
            f"- {filename} / {method_id}: {'; '.join(messages)}"
            for (filename, method_id), messages in reasons.items()
        )
        for filename in dict.fromkeys(filename for filename, _ in reasons):
            self._sweep_output.update(filename, "error", counts, message)
        self._sweep_output.finish("error", message)
        raise ConfigError(message)

    @staticmethod
    def _reference_values(
        reference: ArtifactReference, inversion: InversionRecord | None = None,
    ) -> dict[str, Any]:
        return {
            "uid": reference.sample_uid,
            "method_id": reference.path.parent.parent.name,
            "artifact": str(reference.path),
            "artifact_index": reference.index,
            "artifact_reused": True if inversion is None else inversion.artifact_reused,
            "inversion_batch_id": None if inversion is None else inversion.batch_id,
            "inversion_seconds": None if inversion is None else inversion.inversion_seconds,
            "inversion_batch_seconds": None if inversion is None else inversion.inversion_batch_seconds,
            "inversion_batch_size": None if inversion is None else inversion.inversion_batch_size,
        }

    def _edit_batch(self, items: list[WorkItem], timing: BatchTiming) -> None:
        if not items:
            timing.status = "skipped"
            return
        policy = items[0].prepared
        output = self._output_for(policy.loaded.artifact.method_id)
        batch_seconds = timing.durations.get("preparation", 0.0)
        try:
            if any(
                item.prepared.policy_class is not policy.policy_class
                or item.prepared.settings != policy.settings
                or item.prepared.method.method_id != policy.method.method_id
                for item in items[1:]
            ):
                raise ValueError("A GPU edit batch cannot mix methods, P2P subclasses or settings")
            with timing.measure("editor", device=self.editor.device):
                edited = self.editor.edit_batch(
                    [item.prepared.loaded.sample for item in items],
                    [item.prepared.loaded.artifact for item in items],
                    hooks=[item.prepared.hook for item in items],
                    prompt_to_prompt_class=policy.policy_class,
                    method_settings=policy.settings,
                )
            batch_seconds += timing.durations["editor"]
            if len(edited) != len(items):
                raise RuntimeError("The editor returned a different number of results.")
            records: list[dict[str, Any]] = []
            with timing.measure("image_write"):
                for item, result in zip(items, edited, strict=True):
                    artifact = item.prepared.loaded.artifact
                    reconstructed_path, edited_path = output.save_images(
                        artifact.method_id, artifact.sample_uid,
                        result.reconstructed, result.edited,
                    )
                    records.append({
                        **item.result_values(),
                        "reconstructed_image": str(reconstructed_path.relative_to(output.output_dir)),
                        "edited_image": str(edited_path.relative_to(output.output_dir)),
                        "edit_batch_id": timing.batch_id,
                        "edit_batch_seconds": batch_seconds,
                        "status": "ok",
                    })
            for record in records:
                self._record(record)
        except Exception as exc:
            batch_seconds = timing.durations.get("preparation", 0.0) + timing.durations.get("editor", 0.0)
            for item in items:
                self._record({
                    **item.result_values(),
                    "edit_batch_id": timing.batch_id,
                    "edit_batch_seconds": batch_seconds,
                    "status": "error", "error": str(exc),
                })
            raise

    def _process_edit_references(
        self, references: Sequence[ArtifactReference], *, allow_skip: bool,
        inversions: Mapping[str, InversionRecord] | None = None,
    ) -> None:
        """Read and prepare one method segment of an existing editing chunk."""
        output = self._output_for(references[0].path.parent.parent.name)
        settings = self.artifact_settings
        context = self.inversion_context
        with output.batch("edit", len(references)) as timing:
            timing.extra["edited_count"] = 0
            with timing.measure("input_read"):
                results = self._fetch_many(
                    lambda reference: self._replay.load_result(reference, settings), references
                )
            items: list[WorkItem] = []
            with timing.measure("preparation"):
                for result in results:
                    reference = result.reference
                    inversion = None if inversions is None else inversions[reference.sample_uid]
                    values = self._reference_values(reference, inversion)
                    try:
                        if result.error is not None:
                            raise result.error
                        if result.loaded is None:
                            raise RuntimeError("Artifact loading returned neither a value nor an error")
                        prepared = self._replay.prepare(result.loaded, context)
                    except ArtifactCompatibilityError as exc:
                        if allow_skip:
                            self._record({**values, "status": "skipped", "reason": str(exc)})
                            continue
                        self._record({
                            **values, "edit_batch_id": timing.batch_id,
                            "status": "error", "error": str(exc),
                        })
                        raise
                    except Exception as exc:
                        self._record({
                            **values, "edit_batch_id": timing.batch_id,
                            "status": "error", "error": str(exc),
                        })
                        raise
                    items.append(WorkItem(prepared, inversion))
                    timing.extra["edited_count"] = len(items)
            self._edit_batch(items, timing)

    def edit_artifacts(self, artifact_references: Sequence[ArtifactReference]) -> Path:
        """Replay each parameter file, recording incompatible pairs as skipped."""
        if not artifact_references:
            raise ValueError("At least one saved inversion artifact is required.")
        discover_methods()
        method_ids = tuple(dict.fromkeys(
            method_directory_name(reference.path.parent.parent.name)
            for reference in artifact_references
        ))
        return self._sweep(
            lambda: self._edit_artifacts_for_config(artifact_references), method_ids
        )

    def _edit_artifacts_for_config(self, artifact_references: Sequence[ArtifactReference]) -> None:
        remaining = Counter(reference.path.parent.parent.name for reference in artifact_references)
        self._output_for(artifact_references[0].path.parent.parent.name)
        _ = self.editor
        for path_batch in _chunks(artifact_references, self.config.runtime.batch_size):
            # Preserve chunk and method boundaries while timing each method's reads once.
            for method_id, group in groupby(path_batch, key=lambda ref: ref.path.parent.parent.name):
                references = list(group)
                self._process_edit_references(references, allow_skip=True)
                remaining[method_id] -= len(references)
                if not remaining[method_id] and method_id not in self._finished_methods:
                    self._finish_method(method_id)

    def run_methods(self, method_ids: Sequence[str]) -> Path:
        """Preflight the complete sweep, then invert/reuse and edit the dataset."""
        method_ids = _resolve_method_ids(method_ids)
        uids = self._repository.uids
        if not uids:
            raise ValueError("The project dataset is empty.")
        methods = [get_method(method_id) for method_id in method_ids]
        self._preflight(methods)
        try:
            if self._components is None:
                self._components = SharedInversionComponents(IntermediateCache())
            if self._coordinator is None:
                self._coordinator = InversionCoordinator(ArtifactCatalog(), self._replay)
        except Exception as exc:
            self._sweep_output.finish("error", str(exc))
            raise
        return self._sweep(lambda: self._run_methods_for_config(methods, uids), method_ids)

    def _run_methods_for_config(
        self, methods: Sequence[InversionMethod], uids: Sequence[str],
    ) -> None:
        if self._coordinator is None:
            raise RuntimeError("The inversion coordinator is not initialized")
        context = self.inversion_context
        settings = self.artifact_settings
        for method in methods:
            output = self._output_for(method.method_id)
            timings: list[InversionRecord] = []
            with ExitStack() as resources:
                publication = None
                for uid_batch in _chunks(uids, self.config.runtime.inversion_batch_size):
                    with output.batch("inversion", len(uid_batch)) as timing:
                        timing.extra.update(cache_hits=0, uncached_count=0)
                        if publication is None:
                            # The first lookup includes the published group's integrity check.
                            with timing.measure("cache_lookup"):
                                publication = resources.enter_context(self._coordinator.group(method, context))
                        timings.extend(self._coordinator.invert_batch(
                            method, uid_batch, publication, context, settings, output, timing,
                            read_samples=lambda missing_uids: self._fetch_many(self._sample, missing_uids),
                        ))
                if publication is None:
                    raise RuntimeError("An inversion group requires at least one batch")
                self._coordinator.publish(publication, output, len(uids))
                references = {record.uid: publication.reference(record.uid) for record in timings}
            # Release staging files before the separately sized editing pass.
            for timing_batch in _chunks(timings, self.config.runtime.batch_size):
                self._process_edit_references(
                    [references[record.uid] for record in timing_batch], allow_skip=False,
                    inversions={record.uid: record for record in timing_batch},
                )
            self._finish_method(method.method_id)


def edit_artifacts(artifact_id: str | None = None) -> Path:
    """Replay one artifact ID or all saved artifacts across every parameter file."""
    repository = ArtifactRepository()
    artifact_references = (
        repository.discover() if artifact_id is None else repository.resolve_all(artifact_id)
    )
    return WorkflowRunner().edit_artifacts(artifact_references)


def run_methods(
    method_ids: Sequence[str],
    *,
    pipeline_h_params_file: str | Path | None = None,
) -> Path:
    """Invert and edit with unique requested methods; ALL selects every adapter."""
    method_ids = _resolve_method_ids(method_ids)
    return WorkflowRunner(
        pipeline_h_params_file=pipeline_h_params_file
    ).run_methods(method_ids)
