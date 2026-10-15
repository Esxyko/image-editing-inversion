"""Internal evaluation settings and archived method records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CLIP_MODEL = "openai/clip-vit-large-patch14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
METRICS = (
    "clip_text_similarity", "edited_region_clip", "masked_psnr_db",
    "masked_ssim", "masked_lpips",
)
GROUPS = ("all", "pie-bench", "magic-brush")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    device: str = "cuda"
    batch_size: int = 16

    def __post_init__(self) -> None:
        require(isinstance(self.device, str) and bool(self.device.strip()), "device must be a nonempty string")
        require(type(self.batch_size) is int and self.batch_size > 0, "batch_size must be a positive integer")


@dataclass(frozen=True, slots=True)
class MethodOutput:
    parameter_file: str
    method_id: str
    directory: str
    configuration: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    batches: tuple[dict[str, Any], ...]
