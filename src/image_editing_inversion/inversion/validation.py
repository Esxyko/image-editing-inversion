"""Shared inversion preflight helpers; model libraries are imported on execution."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

from ..config import ExperimentConfig
from .context import InversionContext

if TYPE_CHECKING:
    import torch
    from diffusers import DDIMInverseScheduler


def validate_sampling(config: ExperimentConfig, label: str, *, null_text: bool = False) -> None:
    if config.sampling.eta != 0:
        raise ValueError(f"{label} inversion requires sampling.eta == 0")
    scale = config.inversion_guidance_scale
    if not math.isfinite(scale) or (scale < 1 if null_text else scale < 0):
        requirement = ">= 1" if null_text else "finite and nonnegative"
        raise ValueError(f"{label} inversion requires method_h_params.inversion_guidance_scale {requirement}")
    if null_text and (not math.isfinite(config.editing_guidance_scale) or config.editing_guidance_scale < 1):
        raise ValueError(f"{label} optimization requires sampling.editing_guidance_scale >= 1")


def inverse_scheduler(context: InversionContext, label: str) -> DDIMInverseScheduler:
    """Construct and validate the reverse of the editor's actual DDIM schedule."""
    from diffusers import DDIMInverseScheduler, DDIMScheduler

    editor = context.editor
    if context.config != editor.config:
        raise ValueError(f"{label} inversion context must use the editor's configuration")
    if not isinstance(editor.scheduler, DDIMScheduler):
        raise ValueError(f"{label} inversion requires the editor's DDIM scheduler")
    try:
        scheduler = DDIMInverseScheduler.from_config(editor.scheduler_config)
        scheduler.set_timesteps(context.config.sampling.num_inference_steps, device=editor.device)
    except (ValueError, NotImplementedError) as exc:
        raise ValueError(f"{label} inversion cannot use the editor's scheduler settings: {exc}") from exc
    timesteps = tuple(int(step) for step in scheduler.timesteps.tolist())
    if timesteps != tuple(reversed(editor.expected_timesteps)):
        raise ValueError(f"{label} inverse timesteps must reverse the editor's schedule")
    return scheduler


def require_finite(tensor: torch.Tensor, name: str, label: str) -> None:
    import torch

    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{label} inversion produced non-finite {name}")


def require_finite_many(tensors: Sequence[torch.Tensor], name: str, label: str) -> None:
    """Check same-device tensors with one accelerator-to-host synchronization."""
    import torch

    if tensors and not torch.stack([torch.isfinite(tensor).all() for tensor in tensors]).all().item():
        raise ValueError(f"{label} inversion produced non-finite {name}")
