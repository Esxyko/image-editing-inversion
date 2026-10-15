"""Aggregate by source and publish one atomic JSON document per input ZIP."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence
from uuid import uuid4

from .models import GROUPS, METRICS, require


class EvaluationWriter:
    """Keep method statistics and per-image observations nested in one report."""

    def __init__(self, root: Path) -> None:
        self.directory = root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
        self._created = False

    @staticmethod
    def _statistics(values: Sequence[float | str | None]) -> dict[str, Any]:
        present = [value for value in values if value is not None]
        finite = [value for value in present if value != "+inf"]
        require(all(type(value) in (float, int) and math.isfinite(value) for value in finite),
                "Cannot summarize nonfinite metric values")
        infinite = len(finite) != len(present)
        return {"n": len(present),
                "mean": "+inf" if infinite else statistics.fmean(finite) if finite else None,
                "std": statistics.stdev(finite) if not infinite and len(finite) > 1 else None}

    @classmethod
    def summarize(cls, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        groups = {}
        for group in GROUPS:
            selected = [row for row in rows if group == "all" or row["source_dataset"] == group]
            known_cache = [row["timing"]["inversion_cache_hit"] for row in selected
                           if row["timing"]["inversion_cache_hit"] is not None]
            timing = {}
            for phase in ("inversion", "editing"):
                seconds = [row["timing"][f"{phase}_core_seconds"] for row in selected
                           if row["timing"][f"{phase}_core_seconds"] is not None]
                timing[phase] = {
                    "n": len(seconds), "total_core_seconds": math.fsum(seconds) if seconds else None,
                    "mean_core_seconds_per_image": statistics.fmean(seconds) if seconds else None,
                }
            groups[group] = {
                "sample_count": len(selected),
                "metrics": {metric: cls._statistics([row[metric] for row in selected]) for metric in METRICS},
                "timing": timing,
                "inversion_cache": {"known_count": len(known_cache), "hit_count": sum(known_cache),
                                    "hit_rate": sum(known_cache) / len(known_cache) if known_cache else None},
            }
        return groups

    def publish(self, stem: str, document: dict[str, Any]) -> Path:
        if not self._created:
            self.directory.mkdir(parents=True, exist_ok=False)
            self._created = True
        destination = self.directory / f"{stem}.json"
        require(not destination.exists(), f"Report already exists: {destination}")
        temporary = self.directory / f".{stem}-{uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            temporary.replace(destination)
        except BaseException as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup:
                exc.add_note(f"Could not remove temporary evaluation report: {cleanup}")
            raise
        return destination
