"""Allocate each core batch duration once to its actual participating images."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

from .models import MethodOutput, require


class TimingAggregator:
    """Recover batch-amortized core times; replay-only inversions remain unknown."""

    def __init__(self, method: MethodOutput) -> None:
        self.samples: dict[str, dict[str, Any]] = {
            record["uid"]: {"inversion_core_seconds": None, "editing_core_seconds": None,
                            "inversion_cache_hit": None}
            for record in method.records
        }
        batches: dict[str, dict[str, Any]] = {}
        for batch in method.batches:
            batch_id = batch.get("batch_id")
            require(isinstance(batch_id, str) and bool(batch_id) and batch_id not in batches,
                    "Invalid or duplicate timing batch ID")
            require(batch.get("schema_version") in (1, 2), "Unsupported batch timing schema")
            require(batch.get("phase") in ("inversion", "edit", "publication"), "Unknown timing phase")
            require(batch.get("status") == "ok", "Evaluation requires successful timing batches")
            batches[batch_id] = batch
        used: set[str] = set()
        for phase, field, duration, output in (
            ("inversion", "inversion_batch_id", "method", "inversion_core_seconds"),
            ("edit", "edit_batch_id", "editor", "editing_core_seconds"),
        ):
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in method.records:
                batch_id = record.get(field)
                if phase == "inversion" and batch_id is None:
                    continue
                require(isinstance(batch_id, str) and bool(batch_id), f"Missing {field}")
                grouped[batch_id].append(record)
            for batch_id, records in grouped.items():
                require(batch_id in batches and batches[batch_id]["phase"] == phase,
                        f"Missing or wrong-phase timing batch: {batch_id}")
                require(batch_id not in used, f"Timing batch reused across phases: {batch_id}")
                used.add(batch_id)
                batch = batches[batch_id]
                require(type(batch.get("sample_count")) is int and batch["sample_count"] == len(records),
                        "Timing batch sample count does not match results")
                durations = batch.get("durations_seconds")
                require(isinstance(durations, dict), "Missing batch durations")
                seconds = durations.get(duration)
                require(type(seconds) in (float, int) and math.isfinite(seconds) and seconds >= 0,
                        f"Missing or invalid {duration} core time")
                if phase == "inversion":
                    require(all(type(record.get("artifact_reused")) is bool for record in records),
                            "Missing inversion cache flags")
                    fresh = [record for record in records if not record["artifact_reused"]]
                    hits = len(records) - len(fresh)
                    require(type(batch.get("uncached_count")) is int and batch["uncached_count"] == len(fresh)
                            and type(batch.get("cache_hits")) is int and batch["cache_hits"] == hits,
                            "Inversion cache counts do not match results")
                    require(bool(fresh) or seconds == 0, "Cached-only batch has nonzero core inversion time")
                    for record in records:
                        self.samples[record["uid"]].update({
                            output: 0.0 if record["artifact_reused"] else seconds / len(fresh),
                            "inversion_cache_hit": record["artifact_reused"],
                        })
                else:
                    require(type(batch.get("edited_count")) is int and batch["edited_count"] == len(records),
                            "Editing batch count does not match results")
                    for record in records:
                        self.samples[record["uid"]][output] = seconds / len(records)
        require(all(batch["phase"] == "publication" or batch_id in used for batch_id, batch in batches.items()),
                "Unreferenced core timing batch")
