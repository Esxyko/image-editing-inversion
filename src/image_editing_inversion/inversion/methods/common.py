"""Bundled-method validation, artifact construction, and execution cleanup."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

from ...artifacts import ArtifactCompatibilityError, InversionArtifact
from ..context import InversionContext
from ..validation import require_finite

GUIDANCE_KEY = "guidance_scale"


def validate_ddim_scheduler(config: Mapping[str, Any], label: str) -> None:
    for name, required in (
        ("prediction_type", "epsilon"), ("timestep_spacing", "leading"),
        ("clip_sample", False), ("thresholding", False),
    ):
        if config.get(name) != required:
            raise ValueError(f"{label} inversion requires DDIM {name} == {required!r}")


def guidance_state(context: InversionContext) -> torch.Tensor:
    return torch.full(
        (len(context.editor.expected_timesteps),), context.config.sampling.guidance_scale,
        dtype=torch.float64, device="cpu",
    )


def validate_guidance_state(
    artifact: InversionArtifact, label: str, *, null_text: bool = False,
) -> float:
    guidance = artifact.per_step_state[GUIDANCE_KEY]
    if (not isinstance(guidance, torch.Tensor) or not guidance.is_floating_point()
            or tuple(guidance.shape) != (len(artifact.timesteps),) or not artifact.timesteps):
        raise ValueError(f"{label} guidance_scale must be a floating [timesteps] tensor")
    require_finite(guidance, "saved guidance scale", label.removesuffix(" inversion"))
    scale = float(guidance[0].item())
    if (scale <= 1 if null_text else scale < 0) or not torch.all(guidance == guidance[0]).item():
        bound = "greater than 1" if null_text else "nonnegative"
        raise ValueError(f"{label} guidance_scale must be constant and {bound}")
    return scale


def validate_guidance_compatibility(
    guidance: float, context: InversionContext, label: str,
) -> None:
    if not math.isclose(guidance, context.config.sampling.guidance_scale, rel_tol=0, abs_tol=1e-9):
        raise ArtifactCompatibilityError(f"{label} artifact guidance scale does not match this run")


def build_artifact(
    method_id: str, context: InversionContext, uid: str, terminal_latent: torch.Tensor,
    per_step_state: Mapping[str, torch.Tensor] | None = None,
) -> InversionArtifact:
    def copy(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(device="cpu").clone().contiguous()

    return InversionArtifact(
        method_id=method_id, model_id=context.config.model.model_id,
        model_revision=context.config.model.revision,
        dataset_ref=context.dataset_ref, dataset_fingerprint=context.dataset_fingerprint,
        sample_uid=uid, scheduler_id="ddim", scheduler_config=context.editor.scheduler_config,
        eta=context.config.sampling.eta, timesteps=context.editor.expected_timesteps,
        terminal_latent=copy(terminal_latent),
        per_step_state={name: copy(tensor) for name, tensor in (per_step_state or {}).items()},
    )


@contextmanager
def model_execution(
    pipeline: Any, *, restorations: Sequence[Callable[[], Any]] = (),
) -> Iterator[None]:
    """Attempt every cleanup and retain the original execution exception."""
    execution_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        execution_error = exc
        raise
    finally:
        cleanup_error: BaseException | None = None
        for cleanup in (*restorations, pipeline.maybe_free_model_hooks):
            try:
                cleanup()
            except BaseException as exc:
                primary = execution_error if execution_error is not None else cleanup_error
                if primary is None:
                    cleanup_error = exc
                else:
                    primary.add_note(f"Inversion cleanup failed: {exc}")
        if execution_error is None and cleanup_error is not None:
            raise cleanup_error
