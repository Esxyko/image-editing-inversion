"""Shared source encoding, conditional pivots, and per-image intermediate reuse."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ..artifacts.intermediates import IntermediateCache, TensorSpecs
from .context import InversionContext
from .validation import require_finite

if TYPE_CHECKING:
    from diffusers import DDIMInverseScheduler
    from PIL import Image


@dataclass(frozen=True, slots=True)
class SourceBatch:
    latent: torch.Tensor
    embeddings: torch.Tensor
    identities: tuple[Mapping[str, Any], ...] = ()

    @property
    def conditional(self) -> torch.Tensor:
        return self.embeddings[self.latent.shape[0]:]


def _encode_sources(
    prompts: Sequence[str], images: Sequence[Image.Image], context: InversionContext, label: str,
) -> SourceBatch:
    editor = context.editor
    editor.pipeline.maybe_free_model_hooks()
    latent = editor.encode_images(images)
    editor.pipeline.maybe_free_model_hooks()
    embeddings = editor.encode_prompts([""] * len(prompts) + list(prompts)).detach()
    require_finite(embeddings, "caption embeddings", label)
    editor.pipeline.maybe_free_model_hooks()
    return SourceBatch(latent, embeddings)


@torch.inference_mode(False)
@torch.no_grad()
def prepare_sources(
    uids: Sequence[str], prompts: Sequence[str], images: Sequence[Image.Image],
    context: InversionContext, label: str,
) -> SourceBatch:
    if context.components is None:
        return _encode_sources(prompts, images, context, label)
    return context.components.prepare_sources(uids, prompts, images, context, label)


def _conditional_pivots(
    latent: torch.Tensor, conditional: torch.Tensor, scheduler: DDIMInverseScheduler,
    context: InversionContext, label: str,
) -> list[torch.Tensor]:
    pivots = [latent.detach().to(device="cpu").clone().contiguous()]
    for timestep in scheduler.timesteps:
        model_input = scheduler.scale_model_input(latent, timestep)
        prediction = context.editor.pipeline.unet(
            model_input, timestep, encoder_hidden_states=conditional,
        ).sample
        latent = scheduler.step(prediction, timestep, latent).prev_sample
        require_finite(latent, "pivot latent", label)
        pivots.append(latent.detach().to(device="cpu").clone().contiguous())
    return pivots


@torch.inference_mode(False)
@torch.no_grad()
def conditional_pivots(
    sources: SourceBatch, uids: Sequence[str], scheduler: DDIMInverseScheduler,
    context: InversionContext, label: str,
) -> list[torch.Tensor]:
    if context.components is None:
        return _conditional_pivots(sources.latent, sources.conditional, scheduler, context, label)
    return context.components.conditional_pivots(sources, uids, scheduler, context, label)


class SharedInversionComponents:
    """Compute only missing per-image components and keep memory bounded by a batch."""

    def __init__(self, cache: IntermediateCache) -> None:
        self._cache = cache
        self.reset_diagnostics()

    def reset_diagnostics(self) -> None:
        self.diagnostics: dict[str, Any] = {
            "source_cache_hits": 0, "source_cache_misses": 0,
            "pivot_cache_hits": 0, "pivot_cache_misses": 0,
            "intermediate_cache_write_errors": [],
        }

    def _identity(self, context: InversionContext) -> dict[str, Any]:
        return {"component_version": 2, **context.source_identity(components=("vae", "text_encoder", "tokenizer"))}

    def _store(
        self, component: str, inputs: Mapping[str, Any], tensors: Mapping[str, torch.Tensor],
        specs: TensorSpecs, uids: Sequence[str],
    ) -> None:
        try:
            self._cache.store(component, inputs, tensors, specs=specs,
                              producer={"uids": list(uids), "batch_size": len(uids)})
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            self.diagnostics["intermediate_cache_write_errors"].append({
                "component": component, "uid": inputs["sample_uid"], "error": str(exc),
            })

    @torch.inference_mode(False)
    @torch.no_grad()
    def prepare_sources(
        self, uids: Sequence[str], prompts: Sequence[str], images: Sequence[Image.Image],
        context: InversionContext, label: str,
    ) -> SourceBatch:
        editor = context.editor
        identity = self._identity(context)
        model = context.config.model
        text_shape = (1, editor.pipeline.tokenizer.model_max_length,
                      editor.pipeline.text_encoder.config.hidden_size)
        specs = {
            "latent": ((1, 4, model.height // 8, model.width // 8), editor.dtype),
            "unconditional": (text_shape, editor.dtype), "conditional": (text_shape, editor.dtype),
        }
        inputs: list[dict[str, Any]] = []
        entries: list[dict[str, torch.Tensor] | None] = []
        missing: list[int] = []
        for index, (uid, prompt, image) in enumerate(zip(uids, prompts, images, strict=True)):
            rgb = image.convert("RGB")
            inputs.append({**identity, "sample_uid": uid, "source_prompt": prompt,
                           "image_size": list(rgb.size),
                           "image_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
            entry = self._cache.load("sources", inputs[-1], specs=specs)
            entries.append(entry)
            if entry is None:
                missing.append(index)
        self.diagnostics["source_cache_hits"] += len(uids) - len(missing)
        self.diagnostics["source_cache_misses"] += len(missing)
        if missing:
            fresh = _encode_sources([prompts[index] for index in missing],
                                    [images[index] for index in missing], context, label)
            unconditional, conditional = fresh.embeddings.chunk(2)
            producer = [uids[index] for index in missing]
            for position, index in enumerate(missing):
                entry = {
                    "latent": fresh.latent[position:position + 1].cpu().clone().contiguous(),
                    "unconditional": unconditional[position:position + 1].cpu().clone().contiguous(),
                    "conditional": conditional[position:position + 1].cpu().clone().contiguous(),
                }
                self._store("sources", inputs[index], entry, specs, producer)
                entries[index] = entry
        resolved = [entry for entry in entries if entry is not None]
        if len(resolved) != len(uids):
            raise RuntimeError("A source component was not resolved")
        latent = editor.to_device(torch.cat([entry["latent"] for entry in resolved]), dtype=editor.dtype)
        embeddings = editor.to_device(torch.cat(
            [entry["unconditional"] for entry in resolved] + [entry["conditional"] for entry in resolved]
        ), dtype=editor.dtype)
        return SourceBatch(latent, embeddings, tuple(inputs))

    @torch.inference_mode(False)
    @torch.no_grad()
    def conditional_pivots(
        self, sources: SourceBatch, uids: Sequence[str], scheduler: DDIMInverseScheduler,
        context: InversionContext, label: str,
    ) -> list[torch.Tensor]:
        editor = context.editor
        shape = (len(editor.expected_timesteps) + 1, 1, *sources.latent.shape[1:])
        specs = {"pivots": (shape, editor.dtype)}
        inputs: list[dict[str, Any]] = []
        entries: list[torch.Tensor | None] = []
        missing: list[int] = []
        conditional = sources.conditional
        for index, uid in enumerate(uids):
            if sources.identities[index]["sample_uid"] != uid:
                raise ValueError("Pivot source identities must match the requested sample order")
            inputs.append({
                **sources.identities[index], "scheduler_config": editor.scheduler_config,
                "unet_content": dict(editor.component_content_identity(("unet",))),
                "timesteps": list(editor.expected_timesteps),
            })
            entry = self._cache.load("conditional_pivots", inputs[-1], specs=specs)
            entries.append(None if entry is None else entry["pivots"])
            if entry is None:
                missing.append(index)
        self.diagnostics["pivot_cache_hits"] += len(uids) - len(missing)
        self.diagnostics["pivot_cache_misses"] += len(missing)
        if missing:
            indices = torch.tensor(missing, device=editor.device, dtype=torch.long)
            pivots = _conditional_pivots(sources.latent.index_select(0, indices),
                                         conditional.index_select(0, indices), scheduler, context, label)
            producer = [uids[index] for index in missing]
            for position, index in enumerate(missing):
                trajectory = torch.stack([pivot[position:position + 1] for pivot in pivots]).contiguous()
                self._store("conditional_pivots", inputs[index], {"pivots": trajectory}, specs, producer)
                entries[index] = trajectory
        resolved = [entry for entry in entries if entry is not None]
        if len(resolved) != len(uids):
            raise RuntimeError("A pivot component was not resolved")
        return [torch.cat([entry[step] for entry in resolved]).contiguous()
                for step in range(shape[0])]
