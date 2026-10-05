"""Coordinate reference preparation, GPU editing, images, and result records."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ..artifacts import ArtifactCompatibilityError, ArtifactReference, ArtifactSettings
from ..inversion import InversionContext
from .models import InversionRecord, WorkItem, reference_values
from .output import RunOutput
from .replay import ReplayPreparer
from .timing import BatchTiming


class EditingCoordinator:
    """Edit one method segment using explicit session dependencies."""

    def __init__(self, replay: ReplayPreparer) -> None:
        self._replay = replay

    def process(
        self, references: Sequence[ArtifactReference], context: InversionContext,
        settings: ArtifactSettings, output: RunOutput, *, allow_skip: bool,
        read: Callable[[Callable[..., Any], Sequence[Any]], list[Any]],
        inversions: Mapping[str, InversionRecord] | None = None,
    ) -> None:
        """Read and prepare one method segment of an existing editing chunk."""
        with output.batch("edit", len(references)) as timing:
            timing.extra["edited_count"] = 0
            with timing.measure("input_read"):
                results = read(
                    lambda reference: self._replay.load_result(reference, settings), references
                )
            items: list[WorkItem] = []
            with timing.measure("preparation"):
                for result in results:
                    reference = result.reference
                    inversion = None if inversions is None else inversions[reference.sample_uid]
                    values = reference_values(reference, inversion)
                    try:
                        if result.error is not None:
                            raise result.error
                        if result.loaded is None:
                            raise RuntimeError("Artifact loading returned neither a value nor an error")
                        prepared = self._replay.prepare(result.loaded, context)
                    except ArtifactCompatibilityError as exc:
                        if allow_skip:
                            output.record({**values, "status": "skipped", "reason": str(exc)})
                            continue
                        output.record_error({
                            **values, "edit_batch_id": timing.batch_id,
                            "status": "error", "error": str(exc),
                        }, exc)
                        raise
                    except Exception as exc:
                        output.record_error({
                            **values, "edit_batch_id": timing.batch_id,
                            "status": "error", "error": str(exc),
                        }, exc)
                        raise
                    items.append(WorkItem(prepared, inversion))
                    timing.extra["edited_count"] = len(items)
            self._edit_batch(items, timing, context, output)

    def _edit_batch(
        self, items: list[WorkItem], timing: BatchTiming,
        context: InversionContext, output: RunOutput,
    ) -> None:
        if not items:
            timing.status = "skipped"
            return
        policy = items[0].prepared
        batch_seconds = timing.durations.get("preparation", 0.0)
        try:
            if any(
                item.prepared.policy_class is not policy.policy_class
                or item.prepared.settings != policy.settings
                or item.prepared.method.method_id != policy.method.method_id
                for item in items[1:]
            ):
                raise ValueError("A GPU edit batch cannot mix methods, P2P subclasses or settings")
            with timing.measure("editor", device=context.editor.device):
                edited = context.editor.edit_batch(
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
                output.record(record)
        except Exception as exc:
            batch_seconds = timing.durations.get("preparation", 0.0) + timing.durations.get("editor", 0.0)
            for item in items:
                output.record_error({
                    **item.result_values(),
                    "edit_batch_id": timing.batch_id,
                    "edit_batch_seconds": batch_seconds,
                    "status": "error", "error": str(exc),
                }, exc)
            raise
