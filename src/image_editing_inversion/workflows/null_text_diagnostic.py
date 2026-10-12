"""Temporary Null-text observers and session; reuse the ordinary workflow and math."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ..artifacts import InversionArtifact, normalize_scheduler_config
from ..inversion import InversionContext
from ..inversion.hooks import BatchedDenoisingHook, DenoisingStepState
from ..inversion.methods.null_text import NullTextInversion, _NullTextBatchHook, _NullTextSettings
from .output import _write_json
from .session import WorkflowSession

if TYPE_CHECKING:
    from diffusers import DDIMInverseScheduler


def _json_metrics(value: Any) -> Any:
    """Keep overflowed diagnostic metrics visible without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_metrics(item) for item in value]
    return value


class _DiagnosticOutput:
    """Stream completed steps beside the ordinary method outputs."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._configured = False

    def configure(
        self, context: InversionContext, inverse: DDIMInverseScheduler, settings: Mapping[str, Any],
    ) -> None:
        if self._configured:
            return
        package = Path(__file__).resolve().parents[1]
        sources = (
            "diagnostic.py", "workflows/null_text_diagnostic.py",
            "inversion/methods/null_text.py", "inversion/components.py", "inversion/validation.py",
            "generation/runtime.py", "generation/editor.py", "generation/denoising.py",
            "generation/processor.py", "generation/prompt_to_prompt.py", "generation/alignment.py",
        )
        _write_json(self.directory / "diagnostic-config.json", {
            "schema_version": 1,
            "method_id": "null-text",
            "pipeline_h_params_file": self.directory.parent.name,
            "config": context.config.to_dict(),
            "method_h_params": dict(settings),
            "dataset_ref": context.dataset_ref,
            "dataset_fingerprint": context.dataset_fingerprint,
            "dataset_content": context.dataset_content,
            "model_commit": context.editor.model_commit,
            "model_content": dict(context.editor.model_content_identity),
            "device": str(context.editor.device),
            "model_dtype": str(context.editor.dtype),
            "embedding_parameter_dtype": "torch.float32",
            "forward_timesteps": list(context.editor.expected_timesteps),
            "inverse_timesteps": [int(t) for t in inverse.timesteps.tolist()],
            "forward_scheduler_config": context.editor.scheduler_config,
            "inverse_scheduler_config": normalize_scheduler_config(inverse.config),
            "inversion_cache_enabled": False,
            "intermediate_cache_enabled": False,
            "python_version": sys.version,
            "package_versions": {name: version(name) for name in ("torch", "diffusers", "transformers")},
            "source_sha256": {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in sources},
            "loss_space": "float32 MSE of scaled VAE latents; not decoded RGB pixels",
            "inner_losses_phase": "before each Adam update; loss_after is measured after the last update",
            "early_stop_phase": "before the last Adam update, matching ordinary Null-text",
            "nonfinite_metric_encoding": "string: nan, inf, or -inf",
        })
        for filename in ("optimization.jsonl", "reconstruction.jsonl"):
            (self.directory / filename).touch(exist_ok=False)
        self._configured = True

    def append(self, filename: str, rows: Sequence[Mapping[str, Any]]) -> None:
        with (self.directory / filename).open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(_json_metrics(dict(row)), sort_keys=True, allow_nan=False) + "\n")


class _OptimizationTrace:
    """Measure updates on detached tensors, transferring small metrics once per step."""

    def __init__(
        self, output: _DiagnosticOutput, samples: Sequence[Mapping[str, Any]],
        timesteps: Sequence[int], reconstruction_targets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self._output = output
        self._samples = [
            {key: sample[key] for key in ("uid", "source_prompt", "target_prompt")}
            for sample in samples
        ]
        self._timesteps = tuple(timesteps)
        self._reconstruction_targets = reconstruction_targets
        self.source_latents: torch.Tensor | None = None
        self._before: torch.Tensor | None = None
        self._updates: list[tuple[tuple[int, ...], torch.Tensor]] = []

    def start_step(self, step_index: int, timestep: int, learning_rate: float, tolerance: float) -> None:
        self._step_index = step_index
        self._timestep = timestep
        self._learning_rate = learning_rate
        self._tolerance = tolerance
        self._updates = []

    @torch.no_grad()
    def before_update(
        self, active: Sequence[int], losses: torch.Tensor,
        gradients: Sequence[torch.Tensor], parameters: Sequence[torch.Tensor], dtype: torch.dtype,
    ) -> None:
        self._before = torch.stack([parameters[index].detach().clone() for index in active])
        norms = torch.stack([gradient.detach().float().norm() for gradient in gradients])
        self._pending_metrics = torch.stack((losses.detach(), norms), dim=1)

    @torch.no_grad()
    def after_update(self, active: Sequence[int], parameters: Sequence[torch.Tensor], dtype: torch.dtype) -> None:
        if self._before is None:
            raise RuntimeError("Diagnostic update has no matching parameter snapshot")
        after = torch.stack([parameters[index].detach() for index in active])
        update_norms = (after - self._before).flatten(1).norm(dim=1)
        changed = (after.to(dtype=dtype) != self._before.to(dtype=dtype)).float().flatten(1).mean(1)
        metrics = torch.cat((self._pending_metrics, update_norms[:, None], changed[:, None]), dim=1)
        self._updates.append((tuple(active), metrics))
        self._before = None

    @torch.no_grad()
    def finish_step(self, errors: torch.Tensor, counts: Sequence[int], latent: torch.Tensor) -> None:
        metrics = torch.cat([values for _, values in self._updates]).cpu().tolist()
        traces = [
            {key: [] for key in (
                "inner_losses", "gradient_norms", "embedding_update_norms", "effective_embedding_change_fraction",
            )}
            for _ in self._samples
        ]
        offset = 0
        for active, _ in self._updates:
            for index, values in zip(active, metrics[offset:offset + len(active)], strict=True):
                for key, value in zip(traces[index], values, strict=True):
                    traces[index][key].append(value)
            offset += len(active)
        rows = []
        for sample, trace, loss_after, count in zip(
            self._samples, traces, errors.tolist(), counts, strict=True,
        ):
            losses = trace["inner_losses"]
            if len(losses) != count:
                raise RuntimeError("Diagnostic update counts do not match the optimizer")
            rows.append({
                "schema_version": 1, "method_id": "null-text",
                "pipeline_h_params_file": self._output.directory.parent.name,
                **sample,
                "step_index": self._step_index,
                "timestep": self._timestep,
                "target_pivot_index": len(self._timesteps) - self._step_index - 1,
                "target_timestep": (self._timesteps[self._step_index + 1]
                                    if self._step_index + 1 < len(self._timesteps) else None),
                "target_is_source_latent": self._step_index == len(self._timesteps) - 1,
                "learning_rate": self._learning_rate,
                "early_stop_threshold": self._tolerance,
                "actual_num_inner_steps": count,
                "stop_reason": "threshold_reached" if losses[-1] < self._tolerance else "max_inner_steps",
                "early_stop_loss_before_update": losses[-1],
                "loss_before": losses[0],
                "loss_after": loss_after,
                "best_loss": min(losses + [loss_after]),
                "loss_after_below_threshold": loss_after < self._tolerance,
                **trace,
            })
        self._output.append("optimization.jsonl", rows)
        self._updates = []
        if self._step_index == len(self._timesteps) - 1:
            if self.source_latents is None:
                raise RuntimeError("Diagnostic source latents were not captured")
            optimized = latent.detach().cpu()
            for index, sample in enumerate(self._samples):
                self._reconstruction_targets[sample["uid"]] = (
                    self.source_latents[index:index + 1].clone(), optimized[index:index + 1].clone(),
                )


class _DiagnosticBatchHook(_NullTextBatchHook):
    """Record final source reconstruction errors without changing denoising state."""

    def __init__(
        self, artifacts: Sequence[InversionArtifact], output: _DiagnosticOutput,
        targets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        super().__init__(artifacts)
        self._output = output
        self._targets = [targets.pop(uid) for uid in self.sample_uids]

    @torch.no_grad()
    def after_step(self, step_index: int, timestep: int, state: DenoisingStepState) -> DenoisingStepState:
        self._validate_step(step_index, timestep, state)
        if step_index == len(self._timesteps) - 1:
            source = torch.cat([pair[0] for pair in self._targets]).to(device=state.latents.device).float()
            optimized = torch.cat([pair[1] for pair in self._targets]).to(device=state.latents.device).float()
            reconstructed = state.latents[:self.batch_size].float()
            errors = torch.stack((
                (reconstructed - source).square().flatten(1).mean(1),
                (reconstructed - optimized).square().flatten(1).mean(1),
                (optimized - source).square().flatten(1).mean(1),
            ), dim=1).cpu().tolist()
            self._output.append("reconstruction.jsonl", [
                {
                    "schema_version": 1, "method_id": "null-text", "uid": uid,
                    "pipeline_h_params_file": self._output.directory.parent.name,
                    "step_index": step_index, "timestep": timestep,
                    "replay_source_latent_mse": values[0],
                    "replay_optimized_latent_mse": values[1],
                    "optimization_source_latent_mse": values[2],
                }
                for uid, values in zip(self.sample_uids, errors, strict=True)
            ])
        return state


class _DiagnosticNullTextInversion(NullTextInversion):
    """Attach observers to ordinary Null-text and bypass final inversion cache hits."""

    def __init__(self, output: _DiagnosticOutput) -> None:
        super().__init__()
        self._output = output
        self._observer: _OptimizationTrace | None = None
        self._reconstruction_targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def inversion_cache_parameters(self, context: InversionContext) -> None:
        return None

    def _prepare_inverse_scheduler(self, context: InversionContext) -> DDIMInverseScheduler:
        inverse = super()._prepare_inverse_scheduler(context)
        if self._observer is not None:
            self._output.configure(context, inverse, self.method_h_params(context.config))
        return inverse

    def invert_batch(
        self, samples: Sequence[Mapping[str, Any]], context: InversionContext,
    ) -> list[InversionArtifact]:
        self._observer = _OptimizationTrace(
            self._output, samples, context.editor.expected_timesteps, self._reconstruction_targets,
        )
        try:
            return super().invert_batch(samples, context)
        finally:
            self._observer = None

    def _optimize(
        self, pivots: list[torch.Tensor], unconditional: torch.Tensor, conditional: torch.Tensor,
        settings: _NullTextSettings, context: InversionContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._observer is None:
            raise RuntimeError("Diagnostic optimization has no observer")
        self._observer.source_latents = pivots[0].detach().cpu().clone()
        return super()._optimize(pivots, unconditional, conditional, settings, context, observer=self._observer)

    def create_batch_denoising_hook(self, artifacts: Sequence[InversionArtifact]) -> BatchedDenoisingHook:
        return _DiagnosticBatchHook(artifacts, self._output, self._reconstruction_targets)

    def result_metadata(self, artifact: InversionArtifact) -> Mapping[str, Any]:
        return {
            **super().result_metadata(artifact),
            "optimization_file": "optimization.jsonl",
            "reconstruction_diagnostics_file": "reconstruction.jsonl",
            "diagnostic_settings_file": "diagnostic-config.json",
        }


class DiagnosticWorkflowSession(WorkflowSession):
    """Use the existing Null-text run layout, selection, publication, and editing."""

    def _setup(self, method_ids: Sequence[str]) -> None:
        if tuple(method_ids) != ("null-text",) or len(self._parameter_configs) != 1:
            raise ValueError("Diagnostic experiments require null-text and one pipeline parameter file")
        path = self._parameter_configs[0][0]
        self._methods = {
            "null-text": _DiagnosticNullTextInversion(_DiagnosticOutput(self.output_dir / path.name / "null-text")),
        }

    @property
    def inversion_context(self) -> InversionContext:
        # Recompute source encodings and pivots so stale intermediate caches cannot hide a fault.
        return replace(super().inversion_context, components=None)
