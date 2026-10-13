"""Internal DDIM execution, attention installation, and denoising hook dispatch."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from typing import Any, Sequence

from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
import torch

from ..inversion.hooks import BatchedDenoisingHook, DenoisingHook, DenoisingStepState
from .processor import _PromptToPromptProcessor
from .prompt_to_prompt import PromptToPrompt
from .runtime import ModelRuntime


class _DenoisingExecutor:
    """Execute one editing batch using a runtime supplied by the editor."""

    def __init__(self, runtime: ModelRuntime) -> None:
        self._runtime = runtime

    @staticmethod
    def _check_hook_state(
        updated: DenoisingStepState,
        original: DenoisingStepState,
    ) -> None:
        if not isinstance(updated, DenoisingStepState):
            raise TypeError("Denoising hook must return DenoisingStepState")
        for field_name in (
            "latents", "prompt_embeddings", "negative_prompt_embeddings"
        ):
            actual = getattr(updated, field_name)
            expected = getattr(original, field_name)
            if (
                not isinstance(actual, torch.Tensor)
                or actual.shape != expected.shape
                or actual.device != expected.device
                or actual.dtype != expected.dtype
            ):
                raise ValueError(
                    f"Denoising hook returned invalid {field_name}: expected "
                    f"shape {tuple(expected.shape)}, device {expected.device}, "
                    f"and dtype {expected.dtype}"
                )
        if original.variance_noise is None:
            if updated.variance_noise is not None:
                raise ValueError(
                    "Denoising hook may supply variance_noise only before a "
                    "DDIM step with eta > 0"
                )
        else:
            actual = updated.variance_noise
            expected = original.variance_noise
            if (
                not isinstance(actual, torch.Tensor)
                or actual.shape != expected.shape
                or actual.device != expected.device
                or actual.dtype != expected.dtype
            ):
                raise ValueError(
                    "Denoising hook returned invalid variance_noise: expected "
                    f"shape {tuple(expected.shape)}, device {expected.device}, "
                    f"and dtype {expected.dtype}"
                )

    def _apply_hooks(
        self,
        hooks: Sequence[DenoisingHook | None],
        stage: str,
        step_index: int,
        timestep: int,
        latents: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        negative_prompt_embeddings: torch.Tensor,
        variance_noise: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        if not any(hook is not None for hook in hooks):
            return (
                latents, prompt_embeddings, negative_prompt_embeddings, variance_noise
            )
        batch_size = len(hooks)
        changed: list[DenoisingStepState] = []
        for index, hook in enumerate(hooks):
            pair = DenoisingStepState(
                latents=torch.cat((latents[index:index + 1], latents[batch_size + index:batch_size + index + 1])),
                prompt_embeddings=torch.cat((prompt_embeddings[index:index + 1], prompt_embeddings[batch_size + index:batch_size + index + 1])),
                negative_prompt_embeddings=torch.cat((negative_prompt_embeddings[index:index + 1], negative_prompt_embeddings[batch_size + index:batch_size + index + 1])),
                variance_noise=(
                    None if variance_noise is None else torch.cat(
                        (
                            variance_noise[index:index + 1],
                            variance_noise[batch_size + index:batch_size + index + 1],
                        )
                    )
                ),
            )
            if hook is None:
                changed.append(pair)
                continue
            result = (
                hook.before_step(step_index, timestep, pair)
                if stage == "before"
                else hook.after_step(step_index, timestep, pair)
            )
            self._check_hook_state(result, pair)
            changed.append(result)
        def reassemble(field_name: str) -> torch.Tensor:
            return torch.cat(
                [getattr(item, field_name)[:1] for item in changed]
                + [getattr(item, field_name)[1:] for item in changed]
            )
        return (
            reassemble("latents"),
            reassemble("prompt_embeddings"),
            reassemble("negative_prompt_embeddings"),
            None if variance_noise is None else reassemble("variance_noise"),
        )

    @staticmethod
    def _noise_generator(seed: int, uid: str, device: torch.device) -> torch.Generator:
        digest = hashlib.sha256(uid.encode("utf-8")).digest()
        pair_seed = (seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1)
        return torch.Generator(device=device).manual_seed(pair_seed)

    def _install_processors(
        self, controller: PromptToPrompt
    ) -> dict[str, Any]:
        unet = self._runtime.pipeline.unet
        original = dict(unet.attn_processors)
        replacement = {}
        for name, processor in original.items():
            if not isinstance(processor, (AttnProcessor, AttnProcessor2_0)):
                raise ValueError(
                    f"Prompt-to-Prompt cannot control unsupported UNet attention "
                    f"processor {name}: {type(processor).__name__}"
                )
            if name.endswith(".attn1.processor"):
                is_cross = False
            elif name.endswith(".attn2.processor"):
                is_cross = True
            else:
                raise ValueError(f"Unexpected SD 1.5 UNet attention layer: {name}")
            replacement[name] = _PromptToPromptProcessor(
                processor,
                controller,
                is_cross,
                self._runtime.config.runtime.attention_query_chunk_size,
                name,
            )
        if not replacement or not any(
            name.endswith(".attn2.processor") for name in replacement
        ):
            raise ValueError("The loaded UNet has no controllable cross-attention layers")
        try:
            unet.set_attn_processor(dict(replacement))
        except BaseException as exc:
            try:
                unet.set_attn_processor(dict(original))
            except BaseException as cleanup:
                exc.add_note(f"Attention processor restoration failed: {cleanup}")
            raise
        return original

    def _apply(
        self, hooks: Sequence[DenoisingHook | None], batch_hook: BatchedDenoisingHook | None,
        stage: str, step_index: int, timestep: int, state: DenoisingStepState,
    ) -> DenoisingStepState:
        if batch_hook is not None:
            callback = batch_hook.before_step if stage == "before" else batch_hook.after_step
            updated = callback(step_index, timestep, state)
            self._check_hook_state(updated, state)
            return updated
        return DenoisingStepState(*self._apply_hooks(
            hooks, stage, step_index, timestep, state.latents,
            state.prompt_embeddings, state.negative_prompt_embeddings, state.variance_noise,
        ))

    def denoise(
        self, sample_uids: Sequence[str], latents: torch.Tensor,
        prompt_embeddings: torch.Tensor, negative_prompt_embeddings: torch.Tensor,
        hooks: Sequence[DenoisingHook | None], batch_hook: BatchedDenoisingHook | None,
        controller: PromptToPrompt,
    ) -> torch.Tensor:
        runtime = self._runtime
        pipeline = runtime.pipeline
        scheduler = pipeline.scheduler
        config = runtime.config
        # Method hooks can request the source CFG used to calibrate their state.
        # Target editing has its own CFG; tensors are ordered as all sources/targets.
        batch_size = len(sample_uids)
        source_guidance = []
        for hook in ([batch_hook] * batch_size if batch_hook is not None else hooks):
            scale = None if hook is None else hook.reconstruction_guidance_scale
            scale = config.editing_guidance_scale if scale is None else scale
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
                raise ValueError("Reconstruction guidance scale must be finite and nonnegative")
            source_guidance.append(scale)
        guidance = latents.new_tensor(
            source_guidance
            + [config.editing_guidance_scale] * batch_size
        ).reshape(2 * batch_size, 1, 1, 1)
        generators = [
            self._noise_generator(config.sampling.seed, uid, runtime.device)
            for uid in sample_uids
        ] if config.sampling.eta > 0 else []
        state = DenoisingStepState(latents, prompt_embeddings, negative_prompt_embeddings)
        original = self._install_processors(controller)
        error: BaseException | None = None
        try:
            for step_index, (step_number, timestep) in enumerate(zip(
                runtime.expected_timesteps, scheduler.timesteps, strict=True,
            )):
                controller.step_index = step_index
                variance_noise = None
                if generators:
                    source_noise = torch.stack([
                        torch.randn(
                            state.latents[index].shape, generator=generator,
                            device=runtime.device, dtype=runtime.dtype,
                        )
                        for index, generator in enumerate(generators)
                    ])
                    variance_noise = torch.cat((source_noise, source_noise))
                state = self._apply(
                    hooks, batch_hook, "before", step_index, step_number,
                    replace(state, variance_noise=variance_noise),
                )
                model_input = scheduler.scale_model_input(
                    torch.cat((state.latents, state.latents)), timestep,
                )
                embeddings = torch.cat((state.negative_prompt_embeddings, state.prompt_embeddings))
                prediction = pipeline.unet(
                    model_input, timestep, encoder_hidden_states=embeddings,
                ).sample
                unconditional, conditional = prediction.chunk(2)
                guided = unconditional + guidance * (conditional - unconditional)
                step_kwargs: dict[str, Any] = {"eta": config.sampling.eta}
                if state.variance_noise is not None:
                    step_kwargs["variance_noise"] = state.variance_noise
                updated = scheduler.step(guided, timestep, state.latents, **step_kwargs).prev_sample
                state = self._apply(
                    hooks, batch_hook, "after", step_index, step_number,
                    replace(state, latents=updated, variance_noise=None),
                )
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                pipeline.unet.set_attn_processor(dict(original))
            except BaseException as cleanup:
                if error is None:
                    raise
                error.add_note(f"Attention processor restoration failed: {cleanup}")
        return state.latents
