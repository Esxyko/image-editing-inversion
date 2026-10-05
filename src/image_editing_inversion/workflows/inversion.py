"""Coordinate inversion reuse, validation, staging, and group publication."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Mapping, Sequence

from ..artifacts import (
    ArtifactCatalog, ArtifactGroupPublication, ArtifactReference,
    ArtifactProvenance, ArtifactSettings, InversionArtifact,
)
from ..inversion import InversionContext, InversionMethod
from .models import InversionRecord
from .output import RunOutput
from .replay import ReplayPreparer
from .timing import BatchTiming


class InversionCoordinator:
    """Own inversion cache identity and publication without owning a workflow runner."""

    def __init__(self, catalog: ArtifactCatalog, replay: ReplayPreparer) -> None:
        self._catalog = catalog
        self._replay = replay
        self._identity_context: InversionContext | None = None
        self._identity_method: InversionMethod | None = None
        self._identity_inputs: dict[str, Any] | None = None

    def group(self, method: InversionMethod, context: InversionContext) -> ArtifactGroupPublication:
        return self._catalog.group(method.method_id, pipeline_h_params={
            "num_inference_steps": context.config.sampling.num_inference_steps,
            "guidance_scale": context.config.sampling.guidance_scale,
        })

    def cache_inputs(
        self, method: InversionMethod, uid: str, context: InversionContext,
    ) -> dict[str, Any] | None:
        if self._identity_context is context and self._identity_method is method:
            return None if self._identity_inputs is None else {**self._identity_inputs, "sample_uid": uid}
        parameters = method.inversion_cache_parameters(context)
        if parameters is None:
            self._identity_context = context
            self._identity_method = method
            self._identity_inputs = None
            return None
        if not isinstance(parameters, Mapping):
            raise TypeError("inversion_cache_parameters must return a mapping or None")
        inputs = {
            "cache_version": 2,
            **context.source_identity(),
            "method_id": method.method_id,
            "scheduler": {
                "id": "ddim", "config": context.editor.scheduler_config,
                "timesteps": list(context.editor.expected_timesteps),
                "eta": context.config.sampling.eta,
            },
            "guidance_scale": context.config.sampling.guidance_scale,
            "method_parameters": dict(parameters),
        }
        self._identity_inputs = json.loads(json.dumps(inputs, sort_keys=True, allow_nan=False))
        self._identity_context = context
        self._identity_method = method
        return {**self._identity_inputs, "sample_uid": uid}

    def validate(
        self, artifact: InversionArtifact, method: InversionMethod,
        uid: str, context: InversionContext,
        *, cache_inputs: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(artifact, InversionArtifact):
            raise TypeError(f"Method {method.method_id!r} must return InversionArtifact")
        if artifact.method_id != method.method_id or artifact.sample_uid != uid:
            raise ValueError(
                f"Method {method.method_id!r} returned an artifact for a different method or UID"
            )
        if cache_inputs is not None and (
            artifact.provenance is None
            or artifact.provenance.method_parameters != cache_inputs["method_parameters"]
        ):
            raise ValueError("Artifact production parameters do not match its cache inputs")
        self._replay.validate(artifact, method, context)

    def invert_batch(
        self, method: InversionMethod, uids: Sequence[str], publication: ArtifactGroupPublication,
        context: InversionContext, settings: ArtifactSettings,
        output: RunOutput, timing: BatchTiming,
        *, read_samples: Callable[[Sequence[str]], Sequence[Mapping[str, Any]]],
    ) -> list[InversionRecord]:
        """Resolve cached UIDs before reading samples for inversion misses."""
        inputs: list[dict[str, Any] | None] = []
        resolved: list[InversionArtifact | ArtifactReference | None] = []
        missing: list[int] = []
        timing.extra.update(cache_hits=0, uncached_count=0)
        if context.components is not None:
            context.components.reset_diagnostics()
            timing.extra.update(context.components.diagnostics)
        for index, uid in enumerate(uids):
            started = time.perf_counter()
            try:
                with timing.measure("cache_lookup"):
                    cache_inputs = self.cache_inputs(method, uid, context)
                    cached = None if cache_inputs is None else self._catalog.lookup(
                        cache_inputs, settings=settings
                    )
                    if cached is not None:
                        artifact, reference = cached
                        try:
                            self.validate(artifact, method, uid, context, cache_inputs=cache_inputs)
                        except (ValueError, TypeError, KeyError):
                            cached = None
                    inputs.append(cache_inputs)
                    if cached is None:
                        missing.append(index)
                        resolved.append(None)
                    else:
                        resolved.append(reference)
                    timing.extra.update(cache_hits=len(resolved) - len(missing), uncached_count=len(missing))
            except Exception as exc:
                output.record_error({
                    "uid": uid, "artifact": None, "artifact_index": None,
                    "artifact_reused": False,
                    "inversion_batch_id": timing.batch_id,
                    "inversion_seconds": time.perf_counter() - started,
                    "inversion_batch_seconds": None, "inversion_batch_size": 0,
                    "status": "error", "error": str(exc),
                }, exc)
                raise

        if missing:
            missing_uids = [uids[index] for index in missing]
            try:
                with timing.measure("input_read"):
                    samples = read_samples(missing_uids)
            except Exception as exc:
                for uid in missing_uids:
                    output.record_error({
                        "uid": uid, "artifact": None, "artifact_index": None,
                        "artifact_reused": False, "inversion_batch_id": timing.batch_id,
                        "status": "error", "error": str(exc),
                    }, exc)
                raise

        started = time.perf_counter()
        try:
            if missing:
                try:
                    with timing.measure("method", device=context.editor.device):
                        artifacts = method.invert_batch(samples, context)
                finally:
                    if context.components is not None:
                        timing.extra.update(context.components.diagnostics)
                with timing.measure("validation"):
                    if len(artifacts) != len(missing):
                        raise RuntimeError("The inversion method returned a different number of artifacts")
                    for index, artifact in zip(missing, artifacts, strict=True):
                        if not isinstance(artifact, InversionArtifact):
                            raise TypeError(f"Method {method.method_id!r} must return InversionArtifact")
                        if artifact.provenance is None:
                            artifact.provenance = ArtifactProvenance(
                                model_content=context.editor.model_content_identity,
                                dataset_content=context.dataset_content,
                                numerics=context.editor.inversion_cache_settings(),
                                method_parameters=None if inputs[index] is None else inputs[index]["method_parameters"],
                                guidance_scale=context.config.sampling.guidance_scale,
                                model_commit=context.editor.model_commit,
                            )
                        self.validate(artifact, method, uids[index], context, cache_inputs=inputs[index])
                        resolved[index] = artifact
            with timing.measure("staging"):
                if any(value is None for value in resolved):
                    raise RuntimeError("An inversion batch member was not resolved")
                publication.stage([value for value in resolved if value is not None], inputs)
            batch_seconds = time.perf_counter() - started if missing else 0.0
        except Exception as exc:
            batch_seconds = time.perf_counter() - started
            for index in missing or range(len(uids)):
                previous = resolved[index]
                output.record_error({
                    "uid": uids[index],
                    "artifact": str(previous.path) if isinstance(previous, ArtifactReference) else None,
                    "artifact_index": previous.index if isinstance(previous, ArtifactReference) else None,
                    "artifact_reused": not missing,
                    "inversion_batch_id": timing.batch_id,
                    "inversion_seconds": batch_seconds / len(missing) if missing else 0.0,
                    "inversion_batch_seconds": batch_seconds if missing else 0.0,
                    "inversion_batch_size": len(missing),
                    "status": "error", "error": str(exc),
                }, exc)
            raise
        missed = set(missing)
        return [InversionRecord(
            uid=uid,
            inversion_seconds=batch_seconds / len(missing) if index in missed else 0.0,
            inversion_batch_seconds=batch_seconds if index in missed else 0.0,
            inversion_batch_size=len(missing) if index in missed else 0,
            artifact_reused=index not in missed,
            batch_id=timing.batch_id,
        ) for index, uid in enumerate(uids)]

    def publish(
        self, publication: ArtifactGroupPublication, output: RunOutput, sample_count: int,
    ) -> None:
        with output.batch("publication", sample_count) as timing:
            timing.extra["artifact"] = str(publication.path)
            try:
                with timing.measure("publication"):
                    publication.publish()
            except Exception as exc:
                output.record_error({
                    "uid": None, "artifact": str(publication.path), "artifact_index": None,
                    "status": "error", "error": str(exc),
                }, exc)
                raise
            self._replay.invalidate(publication.path)
