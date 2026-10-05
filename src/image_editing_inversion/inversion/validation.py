"""Shared inversion preflight helpers; model libraries are imported on execution."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..config import ExperimentConfig
from .context import InversionContext

if TYPE_CHECKING:
    import torch
    from diffusers import DDIMInverseScheduler


def validate_sampling(config: ExperimentConfig, label: str, *, null_text: bool = False) -> None:
    if config.sampling.eta != 0:
        raise ValueError(f"{label} inversion requires sampling.eta == 0")
    scale = config.sampling.guidance_scale
    if not math.isfinite(scale) or (scale <= 1 if null_text else scale < 0):
        requirement = "> 1" if null_text else "finite and nonnegative"
        raise ValueError(f"{label} inversion requires sampling.guidance_scale {requirement}")


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
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{label} inversion produced non-finite {name}")
