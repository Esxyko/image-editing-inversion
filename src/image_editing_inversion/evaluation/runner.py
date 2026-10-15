"""Public archive-evaluation facade and one-invocation resource lifecycle."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import UTC, datetime
import logging
from pathlib import Path
import time
from typing import Any, Sequence
from zipfile import BadZipFile

from .._project import project_paths
from .archive import ArchiveReader
from .models import EvaluationSettings, require
from .output import EvaluationWriter
from .timing import TimingAggregator

logger = logging.getLogger(__name__)


class ArchiveEvaluator:
    """Evaluate multiple ZIPs against one dataset and one set of metric models."""

    def __init__(self, archives: Sequence[str | Path] | None, settings: EvaluationSettings) -> None:
        self.settings = settings
        self.paths = project_paths()
        if archives is None:
            candidates = sorted(self.paths.output.glob("*.zip"))
        else:
            require(not isinstance(archives, (str, bytes, Path)), "archives must be a sequence of ZIP paths")
            candidates = [Path(path).expanduser() for path in archives]
        self.archives = tuple(dict.fromkeys(path.resolve(strict=True) for path in candidates))
        require(bool(self.archives), "No ZIP archives selected; pass ZIP paths or populate data/output")
        require(all(path.is_file() and path.suffix.lower() == ".zip" for path in self.archives),
                "Every input must be a ZIP file")
        require(len({path.stem.casefold() for path in self.archives}) == len(self.archives),
                "Input ZIP basenames must be distinct")

    def run(self) -> Path:
        # Keep optional model dependencies outside normal CLI/package imports.
        from ..data import DatasetRepository
        from ..data.schema import SOURCE_DATASETS
        from .metrics import MetricCalculator

        invocation_started = datetime.now(UTC).isoformat()
        invocation_clock = time.perf_counter()
        repository = DatasetRepository()
        dataset_identity = {"reference": repository.dataset_ref, "fingerprint": repository.dataset_fingerprint,
                            "content_digest": repository.content_digest, "split": "eval",
                            "source_assignment": "DatasetRepository.select_uids"}
        sources = {uid: source for source in SOURCE_DATASETS for uid in repository.select_uids(source)}
        writer = EvaluationWriter(self.paths.metrics)
        with ExitStack() as stack:
            readers = []
            timings = {}
            for path in self.archives:
                try:
                    reader = stack.enter_context(ArchiveReader(path))
                    reader.prepare(repository)
                    for method in reader.methods:
                        timings[(path, method.parameter_file, method.method_id)] = TimingAggregator(method)
                except (BadZipFile, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid evaluation archive {path}: {exc}") from exc
                readers.append(reader)
            calculator = stack.enter_context(MetricCalculator(self.settings, self.paths.evaluation_cache))
            for reader in readers:
                started = datetime.now(UTC).isoformat()
                clock = time.perf_counter()
                parameters: dict[str, Any] = {}
                for method in reader.methods:
                    method_clock = time.perf_counter()
                    timing = timings[(reader.path, method.parameter_file, method.method_id)]
                    scored: dict[str, dict[str, Any]] = {}
                    logger.info("Evaluating %s / %s / %s", reader.path.name, method.parameter_file, method.method_id)
                    try:
                        for records, pixels in reader.iter_images(method, self.settings.batch_size):
                            samples = [repository.sample(record["uid"]) for record in records]
                            for record, row in zip(records, calculator.score(pixels, samples), strict=True):
                                uid = row["uid"]
                                require(uid == record["uid"] and uid not in scored,
                                        "Duplicate or misaligned evaluated UID")
                                row.update(source_dataset=sources[uid], timing=timing.samples[uid],
                                           result_metadata=record)
                                scored[uid] = row
                            if len(scored) % (self.settings.batch_size * 10) == 0:
                                logger.info("  %d/%d images", len(scored), len(method.records))
                        require(set(scored) == {record["uid"] for record in method.records},
                                "Incomplete evaluated UID coverage")
                    except (BadZipFile, KeyError, TypeError, ValueError, RuntimeError) as exc:
                        raise RuntimeError(
                            f"Evaluation failed for {reader.path.name}/"
                            f"{method.parameter_file}/{method.method_id}: {exc}"
                        ) from exc
                    rows = [scored[record["uid"]] for record in method.records]
                    parameter = parameters.setdefault(method.parameter_file, {"methods": {}})
                    parameter["methods"][method.method_id] = {
                        "configuration": method.configuration,
                        "evaluation_elapsed_seconds": time.perf_counter() - method_clock,
                        "groups": writer.summarize(rows), "scores": rows,
                    }
                document = {
                    "schema_version": 1, "status": "complete", "archive": reader.provenance(),
                    "dataset": dataset_identity,
                    "evaluation": {
                        "invocation_started_at": invocation_started, "started_at": started,
                        "finished_at": datetime.now(UTC).isoformat(),
                        "archive_elapsed_seconds": time.perf_counter() - clock,
                        "invocation_elapsed_seconds_at_completion": time.perf_counter() - invocation_clock,
                        "elapsed_scope": (
                            "archive scoring includes reads and metric computation; invocation also includes "
                            "dataset/model setup and earlier archives; current report serialization is excluded"
                        ),
                        **calculator.metadata(),
                        "timing_protocol": {
                            "inversion": (
                                "batches.jsonl durations_seconds.method, uniformly divided among uncached "
                                "participants; cache hits receive zero"
                            ),
                            "editing": (
                                "batches.jsonl durations_seconds.editor, uniformly divided among edited participants"
                            ),
                            "grouping": (
                                "mean of per-image allocated core times, including cached zeros; "
                                "cross-source batches allocated before grouping"
                            ),
                            "replay": "missing inversion batch ID means unknown time and cache state (null), not zero",
                            "limitations": (
                                "batch-amortized observed core times, not individually measured latency or original "
                                "cache-generation time; IO, preparation, publication and model loading excluded"
                            ),
                        },
                    },
                    "parameter_files": parameters,
                }
                destination = writer.publish(reader.path.stem, document)
                logger.info("Published %s", destination)
        return writer.directory


def evaluate_archives(
    archives: Sequence[str | Path] | None = None, *, device: str = "cuda", batch_size: int = 16,
) -> Path:
    """Write one nested JSON per ZIP; omitted inputs select data/output/*.zip.

    Explicit paths resolve from the caller's working directory. The returned
    directory is a fresh invocation under the fixed data/metrics report root.
    """
    return ArchiveEvaluator(archives, EvaluationSettings(device, batch_size)).run()
