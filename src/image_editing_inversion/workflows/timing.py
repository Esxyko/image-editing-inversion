"""Wall-clock stage timing with synchronization at model-call boundaries."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, Iterator, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class BatchTiming:
    phase: str
    sample_count: int
    batch_id: str = field(default_factory=lambda: uuid4().hex)
    durations: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def __post_init__(self) -> None:
        stages = {
            "inversion": ("input_read", "cache_lookup", "method", "validation", "staging"),
            "edit": ("input_read", "preparation", "editor", "image_write"),
            "publication": ("publication",),
        }
        for stage in stages[self.phase]:
            self.durations.setdefault(stage, 0.0)

    @staticmethod
    def _synchronize(device: torch.device | None) -> None:
        if device is None or device.type == "cpu":
            return
        import torch

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    @contextmanager
    def measure(self, stage: str, *, device: torch.device | None = None) -> Iterator[None]:
        self._synchronize(device)
        started = time.perf_counter()
        try:
            yield
        except BaseException as exc:
            try:
                self._synchronize(device)
            except BaseException as cleanup:
                exc.add_note(f"Timing synchronization failed: {cleanup}")
            raise
        else:
            self._synchronize(device)
        finally:
            self.durations[stage] = self.durations.get(stage, 0.0) + time.perf_counter() - started

    def values(self, error: BaseException | None = None) -> dict[str, Any]:
        values = {
            "schema_version": 1,
            "batch_id": self.batch_id,
            "phase": self.phase,
            "sample_count": self.sample_count,
            "status": "error" if error is not None else self.status,
            "durations_seconds": dict(self.durations),
            **self.extra,
        }
        if error is not None:
            values["error"] = str(error)
        return values
