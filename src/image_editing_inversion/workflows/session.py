"""Own one workflow invocation's dataset, model, adapters, and sweep lifecycle."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, Sequence

from ..artifacts import ArtifactCatalog, ArtifactReference, ArtifactSettings, IntermediateCache
from ..artifacts.layout import method_directory_name
from ..config import ExperimentConfig, discover_pipeline_h_params
from ..config.loader import ConfigurationLoader
from ..data import DatasetRepository
from ..inversion import InversionContext, InversionMethod, get_method
from ..inversion.components import SharedInversionComponents
from .editing import EditingCoordinator
from .inversion import InversionCoordinator
from .models import InversionRecord
from .output import RunOutput, SweepOutput
from .preflight import SweepPreflight
from .replay import ReplayPreparer

if TYPE_CHECKING:
    from ..generation import Editor


def _chunks(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class WorkflowSession:
    """Keep execution state private and discard it after each invocation."""

    def __init__(self, pipeline_h_params_file: str | Path | None) -> None:
        loader = ConfigurationLoader()
        self._parameter_configs = tuple(
            (path, loader.load(path)) for path in discover_pipeline_h_params(pipeline_h_params_file)
        )
        self.config = self._parameter_configs[0][1]
        for _, config in self._parameter_configs:
            config.validate_hardware()
        self._sweep_output = SweepOutput([path for path, _ in self._parameter_configs])
        self.output_dir = self._sweep_output.output_dir
        self._outputs: dict[str, RunOutput] = {}
        self._finished_methods: set[str] = set()
        self._active_method_id: str | None = None
        self._parameter_path: Path | None = None
        self._methods: dict[str, InversionMethod] = {}
        self._repository: DatasetRepository | None = None
        self._editor: Editor | None = None
        self._replay: ReplayPreparer | None = None
        self._editing: EditingCoordinator | None = None
        self._coordinator: InversionCoordinator | None = None
        self._components: SharedInversionComponents | None = None

    @property
    def repository(self) -> DatasetRepository:
        if self._repository is None:
            raise RuntimeError("The workflow dataset is not initialized")
        return self._repository

    @property
    def dataset_ref(self) -> str:
        return self.repository.dataset_ref

    @property
    def dataset_fingerprint(self) -> str:
        return self.repository.dataset_fingerprint

    @property
    def dataset_location(self) -> str:
        return self.repository.dataset_location

    def _setup(self, method_ids: Sequence[str]) -> None:
        for method_id in method_ids:
            self._active_method_id = method_id
            self._methods[method_id] = get_method(method_id)
        self._active_method_id = None

    def _load_dataset(self) -> None:
        self._repository = DatasetRepository()
        self._replay = ReplayPreparer(self._repository, self._methods)
        self._editing = EditingCoordinator(self._replay)

    @contextmanager
    def _execution(self, method_ids: Sequence[str]) -> Iterator[None]:
        try:
            self._sweep_output.set_methods(method_ids)
            yield
        except BaseException as exc:
            if self._sweep_output.status != "error":
                try:
                    self._fail(exc)
                except BaseException as recording:
                    exc.add_note(f"Could not finalize sweep manifest: {recording}")
            raise

    def _fail(self, error: BaseException) -> None:
        filename = None if self._parameter_path is None else self._parameter_path.name
        if filename is not None:
            for method_id, output in self._outputs.items():
                if output.counts["error"] or method_id == self._active_method_id:
                    status = "error"
                elif method_id in self._finished_methods:
                    status = "ok" if output.counts["ok"] else "skipped"
                else:
                    status = "running"
                self._sweep_output.update_method(
                    filename, method_id, status, output.counts,
                    str(error) if status == "error" else None,
                )
        self._sweep_output.fail(str(error), filename=filename, method_id=self._active_method_id)
        if filename is not None:
            self._sweep_output.update(filename, "error", self._counts(), str(error))
            self._sweep_output.finish("error", str(error))

    def _sweep(self, action: Callable[[], None]) -> Path:
        completed = 0
        for path, config in self._parameter_configs:
            self._activate_configuration(path, config)
            self._sweep_output.update(path.name, "running", self._counts())
            action()
            for method_id in self._outputs:
                if method_id not in self._finished_methods:
                    self._finish_method(method_id)
            counts = self._counts()
            completed += counts["ok"]
            self._sweep_output.update(path.name, "ok" if counts["ok"] else "skipped", counts)
        if not completed:
            message = f"No compatible artifact/file pairs produced images; see {self.output_dir}"
            self._sweep_output.finish("error", message)
            raise ValueError(message)
        self._sweep_output.finish("ok")
        return self.output_dir

    def run_methods(self, method_ids: Sequence[str]) -> Path:
        with self._execution(method_ids):
            self._setup(method_ids)
            methods = list(self._methods.values())
            preflight = SweepPreflight(self._parameter_configs, methods, self._sweep_output)
            preflight.configuration()
            self._load_dataset()
            uids = self.repository.uids
            if not uids:
                raise ValueError("The project dataset is empty.")
            preflight.runtime(self._activate_configuration, lambda: self.inversion_context)
            self._components = SharedInversionComponents(IntermediateCache())
            if self._replay is None:
                raise RuntimeError("Replay preparation is not initialized")
            self._coordinator = InversionCoordinator(ArtifactCatalog(), self._replay)
            return self._sweep(lambda: self._run_methods_for_config(methods, uids))

    def edit_artifacts(self, references: Sequence[ArtifactReference]) -> Path:
        method_ids = tuple(dict.fromkeys(reference.group.method_id for reference in references))
        with self._execution(method_ids):
            self._setup(method_ids)
            self._load_dataset()
            return self._sweep(lambda: self._edit_artifacts_for_config(references))

    def _process_edit_references(
        self, references: Sequence[ArtifactReference], *, allow_skip: bool,
        inversions: Mapping[str, InversionRecord] | None = None,
    ) -> None:
        if self._editing is None:
            raise RuntimeError("Editing coordination is not initialized")
        output = self._output_for(references[0].group.method_id)
        self._editing.process(
            references, self.inversion_context, self.artifact_settings, output,
            allow_skip=allow_skip, inversions=inversions, read=self._fetch_many,
        )

    def _activate_configuration(self, path: Path, config: ExperimentConfig) -> None:
        self._outputs = {}
        self._finished_methods = set()
        self._active_method_id = None
        self._parameter_path = path
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

    @property
    def editor(self) -> Editor:
        if self._editor is None:
            from ..generation import Editor

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
            dataset_content=self.repository.content_digest,
        )

    def _sample(self, uid: str) -> Mapping[str, Any]:
        return self.repository.sample(uid)

    def _fetch_many(self, function: Any, values: Sequence[Any]) -> list[Any]:
        """Use the configured number of threads for dataset/artifact reads."""
        if self.config.runtime.num_workers == 0:
            return [function(value) for value in values]
        with ThreadPoolExecutor(max_workers=self.config.runtime.num_workers) as pool:
            return list(pool.map(function, values))

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
            dataset_content=self.repository.content_digest,
            numerics=self.editor.inversion_cache_settings(),
        )

    def _edit_artifacts_for_config(self, artifact_references: Sequence[ArtifactReference]) -> None:
        remaining = Counter(reference.group.method_id for reference in artifact_references)
        self._output_for(artifact_references[0].group.method_id)
        _ = self.editor
        for path_batch in _chunks(artifact_references, self.config.runtime.batch_size):
            # Preserve chunk and method boundaries while timing each method's reads once.
            for method_id, group in groupby(path_batch, key=lambda ref: ref.group.method_id):
                references = list(group)
                self._process_edit_references(references, allow_skip=True)
                remaining[method_id] -= len(references)
                if not remaining[method_id] and method_id not in self._finished_methods:
                    self._finish_method(method_id)

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
