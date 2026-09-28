"""Shared Stable Diffusion 1.5 Prompt-to-Prompt editing backbone.

The source and target branches start from the same inverted latent. Attention
maps from the source branch control the target branch for the configured early
fraction of DDIM steps. An inversion adapter may additionally modify one pair's
latents or text embeddings through a :class:`DenoisingHook`.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
from PIL import Image
import torch
import torch.nn.functional as F

from .artifact import DenoisingHook, DenoisingStepState, InversionArtifact
from .config import ExperimentConfig, PromptToPromptConfig


@dataclass(frozen=True, slots=True)
class EditResult:
    """The reconstruction and edit for one dataset sample."""

    reconstructed: Image.Image
    edited: Image.Image


@dataclass(frozen=True, slots=True)
class PairAttentionMap:
    """Token alignment used by the default Prompt-to-Prompt policy."""

    mode: str
    replacement: torch.Tensor | None = None
    refinement_indices: torch.Tensor | None = None
    refinement_alpha: torch.Tensor | None = None


def _prompt_ids(tokenizer: Any, prompt: str, max_length: int) -> list[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(ids) > max_length:
        raise ValueError(
            f"Prompt has {len(ids)} CLIP tokens but this model accepts at most "
            f"{max_length}; shorten the dataset caption before editing"
        )
    return ids


def _word_spans(tokenizer: Any, words: list[str], full_length: int) -> list[range]:
    """Locate each whitespace word in CLIP's tokenized prompt."""
    spans: list[range] = []
    previous = 1  # CLIP's beginning-of-text token.
    for word_index in range(len(words)):
        prefix = " ".join(words[: word_index + 1])
        end = 1 + len(tokenizer.encode(prefix, add_special_tokens=False))
        if end < previous or end > full_length - 1:
            raise ValueError("Cannot align caption words to CLIP tokens")
        spans.append(range(previous, end))
        previous = end
    if previous != full_length - 1:
        raise ValueError("Cannot align caption words to CLIP tokens")
    return spans


def _pair_attention_map(
    tokenizer: Any,
    source_prompt: str,
    target_prompt: str,
    requested_mode: str,
    max_length: int,
    device: torch.device,
) -> PairAttentionMap:
    source_ids = _prompt_ids(tokenizer, source_prompt, max_length)
    target_ids = _prompt_ids(tokenizer, target_prompt, max_length)
    source_words, target_words = source_prompt.split(), target_prompt.split()
    mode = (
        "replace" if len(source_words) == len(target_words) else "refine"
    ) if requested_mode == "auto" else requested_mode

    if mode == "replace":
        if len(source_words) != len(target_words):
            raise ValueError(
                "Prompt-to-Prompt replacement requires equal source and target "
                "word counts; select refinement or auto mode"
            )
        source_spans = _word_spans(tokenizer, source_words, len(source_ids))
        target_spans = _word_spans(tokenizer, target_words, len(target_ids))
        mapper = torch.zeros((max_length, max_length), dtype=torch.float32)
        mapper[0, 0] = 1.0
        target_eos = len(target_ids) - 1
        for source_span, target_span in zip(source_spans, target_spans):
            if not target_span:
                raise ValueError("Cannot replace a word with no CLIP tokens")
            if len(source_span) == len(target_span):
                for source_index, target_index in zip(source_span, target_span):
                    mapper[source_index, target_index] = 1.0
            else:
                weight = 1.0 / len(target_span)
                for source_index in source_span:
                    mapper[source_index, target_span.start : target_span.stop] = weight
        # Map end-of-text and padding tokens to their counterparts. Every source
        # row then preserves its attention mass, including CLIP's padded tail.
        for source_index in range(len(source_ids) - 1, max_length):
            target_index = min(
                target_eos + source_index - (len(source_ids) - 1), max_length - 1
            )
            mapper[source_index, target_index] = 1.0
        if not torch.allclose(mapper.sum(dim=1), torch.ones(max_length)):
            raise ValueError("Cannot construct a complete replacement token map")
        return PairAttentionMap(mode="replace", replacement=mapper.to(device))

    if mode == "refine":
        indices = torch.zeros(max_length, dtype=torch.long)
        alpha = torch.zeros(max_length, dtype=torch.float32)
        for match in SequenceMatcher(
            a=source_ids, b=target_ids, autojunk=False
        ).get_matching_blocks():
            for offset in range(match.size):
                indices[match.b + offset] = match.a + offset
                alpha[match.b + offset] = 1.0
        for target_index in range(len(target_ids), max_length):
            indices[target_index] = min(
                len(source_ids) + target_index - len(target_ids), max_length - 1
            )
            alpha[target_index] = 1.0
        return PairAttentionMap(
            mode="refine",
            refinement_indices=indices.to(device),
            refinement_alpha=alpha.to(device),
        )

    raise ValueError(f"Unsupported Prompt-to-Prompt mode: {mode!r}")


class PromptToPrompt:
    """Subclassable attention policy for a batch of source/target pairs.

    Subclasses may override token alignment, the active-step schedule, and the
    cross- or self-attention transfer methods. ``artifacts`` and ``settings``
    are available to those methods. The editor retains the four-branch UNet
    layout, DDIM loop, model setup, and denoising hooks.
    """

    _MAX_SELF_REPLACE_TOKENS = 16**2

    @classmethod
    def validate_method_settings(cls, settings: Mapping[str, Any]) -> Mapping[str, Any]:
        """Validate and normalize method-specific YAML settings.

        The base policy accepts no extra settings. Subclasses that use settings
        must override this method and return a JSON-compatible mapping.
        """
        if not isinstance(settings, Mapping):
            raise TypeError("Prompt-to-Prompt settings must be a mapping")
        if settings:
            raise ValueError(
                f"{cls.__name__} does not accept method-specific Prompt-to-Prompt settings"
            )
        return {}

    def __init__(
        self,
        config: PromptToPromptConfig,
        tokenizer: Any,
        source_prompts: Sequence[str],
        target_prompts: Sequence[str],
        artifacts: Sequence[InversionArtifact],
        *,
        num_steps: int,
        max_length: int,
        device: torch.device,
        settings: Mapping[str, Any],
    ) -> None:
        if len(source_prompts) != len(target_prompts) or len(artifacts) != len(source_prompts):
            raise ValueError("Prompt-to-Prompt requires one artifact and two prompts per pair")
        self.config = config
        self.tokenizer = tokenizer
        self.source_prompts = tuple(source_prompts)
        self.target_prompts = tuple(target_prompts)
        self.num_steps = num_steps
        self.max_length = max_length
        self.device = device
        self.artifacts = tuple(artifacts)
        self.batch_size = len(self.artifacts)
        self.settings = dict(settings)
        self.cross_steps = math.ceil(num_steps * config.cross_replace_fraction)
        self.self_steps = math.ceil(num_steps * config.self_replace_fraction)
        self.step_index = -1
        self.pair_maps = tuple(
            self.build_pair_map(source, target)
            for source, target in zip(self.source_prompts, self.target_prompts, strict=True)
        )

    def build_pair_map(self, source_prompt: str, target_prompt: str) -> PairAttentionMap:
        """Return token alignment for one pair; override for custom mapping."""
        return _pair_attention_map(
            self.tokenizer, source_prompt, target_prompt, self.config.mode,
            self.max_length, self.device,
        )

    def active(self, is_cross: bool, query_tokens: int, layer_name: str) -> bool:
        """Return whether attention transfer applies at the current step."""
        if not is_cross and query_tokens > self._MAX_SELF_REPLACE_TOKENS:
            return False
        count = self.cross_steps if is_cross else self.self_steps
        return 0 <= self.step_index < count

    def replace_cross_attention(
        self,
        source_prob: torch.Tensor,
        target_prob: torch.Tensor,
        pair_index: int,
        layer_name: str,
    ) -> torch.Tensor:
        """Return target cross-attention probabilities for one query chunk.

        ``layer_name`` identifies the UNet processor. The result must match
        ``target_prob`` in shape, device, and dtype.
        """
        pair_map = self.pair_maps[pair_index]
        if pair_map.mode == "replace":
            assert pair_map.replacement is not None
            return torch.matmul(source_prob, pair_map.replacement)
        assert pair_map.refinement_indices is not None
        assert pair_map.refinement_alpha is not None
        aligned = source_prob.index_select(-1, pair_map.refinement_indices)
        alpha = pair_map.refinement_alpha.view(1, 1, -1)
        return alpha * aligned + (1.0 - alpha) * target_prob

    def replace_self_attention(
        self,
        source_prob: torch.Tensor,
        target_prob: torch.Tensor,
        pair_index: int,
        layer_name: str,
    ) -> torch.Tensor:
        """Return target self-attention probabilities for one query chunk."""
        return source_prob


class _PromptToPromptProcessor:
    """Diffusers attention processor that transfers conditional attention maps.

    The UNet input has four contiguous branches: unconditional source,
    unconditional target, conditional source, and conditional target. As in the
    reference implementation, self-attention sharing is limited to at most
    16 x 16 spatial tokens; higher-resolution self-attention uses its original
    processor. Active attention maps are computed in query chunks.
    """

    def __init__(
        self, original: AttnProcessor | AttnProcessor2_0,
        controller: PromptToPrompt,
        is_cross: bool,
        query_chunk_size: int,
        layer_name: str,
    ) -> None:
        self.original = original
        self.controller = controller
        self.is_cross = is_cross
        self.query_chunk_size = query_chunk_size
        self.layer_name = layer_name

    @staticmethod
    def _probabilities(
        query: torch.Tensor,
        key: torch.Tensor,
        scale: float,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = torch.bmm(query.float(), key.float().transpose(1, 2)) * scale
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -torch.inf)
            else:
                scores = scores + mask.float()
        return scores.softmax(dim=-1)

    @staticmethod
    def _mask_slice(
        mask: torch.Tensor | None, branch: int, start: int, end: int
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        branch_mask = mask[branch]
        return (
            branch_mask
            if branch_mask.shape[-2] == 1
            else branch_mask[:, start:end, :]
        )

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        query_tokens = (
            hidden_states.shape[-2] * hidden_states.shape[-1]
            if hidden_states.ndim == 4
            else hidden_states.shape[-2]
        )
        if not self.controller.active(self.is_cross, query_tokens, self.layer_name):
            return self.original(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb,
                *args, **kwargs,
            )
        if args or kwargs:
            raise ValueError(
                "This Prompt-to-Prompt processor does not support extra attention "
                "arguments; refusing to skip active attention control"
            )
        if self.is_cross != (encoder_hidden_states is not None):
            raise ValueError("Unexpected SD 1.5 attention layer type")

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channels, height * width).transpose(1, 2)
        elif input_ndim == 3:
            batch_size = hidden_states.shape[0]
        else:
            raise ValueError("Unsupported attention hidden-state rank")
        pairs = self.controller.batch_size
        if batch_size != 4 * pairs:
            raise ValueError(
                f"Prompt-to-Prompt expected {4 * pairs} UNet branches, got {batch_size}"
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        heads = attn.heads
        query = query.view(batch_size, -1, heads, query.shape[-1] // heads).transpose(1, 2)
        key = key.view(batch_size, -1, heads, key.shape[-1] // heads).transpose(1, 2)
        value = value.view(batch_size, -1, heads, value.shape[-1] // heads).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if attention_mask is not None:
            prepared_mask = attn.prepare_attention_mask(
                attention_mask, key.shape[-2], batch_size
            ).view(batch_size, heads, -1, key.shape[-2])
        else:
            prepared_mask = None
        output = torch.empty(
            (*query.shape[:-1], value.shape[-1]),
            device=query.device,
            dtype=value.dtype,
        )
        # The unconditional branches are never edited by Prompt-to-Prompt.
        output[: 2 * pairs] = F.scaled_dot_product_attention(
            query[: 2 * pairs], key[: 2 * pairs], value[: 2 * pairs],
            attn_mask=None if prepared_mask is None else prepared_mask[: 2 * pairs],
            dropout_p=0.0, is_causal=False, scale=attn.scale,
        )

        query_length = query.shape[-2]
        for pair_index in range(pairs):
            source_branch = 2 * pairs + pair_index
            target_branch = 3 * pairs + pair_index
            source_key = key[source_branch]
            target_key = key[target_branch]
            source_value = value[source_branch]
            target_value = value[target_branch]
            for start in range(0, query_length, self.query_chunk_size):
                end = min(start + self.query_chunk_size, query_length)
                source_prob = self._probabilities(
                    query[source_branch, :, start:end], source_key, attn.scale,
                    self._mask_slice(prepared_mask, source_branch, start, end),
                )
                output[source_branch, :, start:end] = torch.bmm(
                    source_prob.to(source_value.dtype), source_value
                )
                if self.is_cross or (
                    type(self.controller).replace_self_attention
                    is not PromptToPrompt.replace_self_attention
                ):
                    target_prob = self._probabilities(
                        query[target_branch, :, start:end], target_key, attn.scale,
                        self._mask_slice(prepared_mask, target_branch, start, end),
                    )
                else:
                    target_prob = source_prob
                if self.is_cross:
                    replacement_prob = self.controller.replace_cross_attention(
                        source_prob, target_prob, pair_index, self.layer_name
                    )
                else:
                    replacement_prob = self.controller.replace_self_attention(
                        source_prob, target_prob, pair_index, self.layer_name
                    )
                if (
                    not isinstance(replacement_prob, torch.Tensor)
                    or replacement_prob.shape != target_prob.shape
                    or replacement_prob.device != target_prob.device
                    or replacement_prob.dtype != target_prob.dtype
                ):
                    raise ValueError(
                        f"Prompt-to-Prompt subclass returned invalid attention at {self.layer_name}"
                    )
                output[target_branch, :, start:end] = torch.bmm(
                    replacement_prob.to(target_value.dtype), target_value
                )

        hidden_states = output.transpose(1, 2).reshape(batch_size, query_length, -1)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, channels, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class Editor:
    """Load one SD 1.5 pipeline and edit batches of compatible artifacts."""

    def __init__(self, config: ExperimentConfig) -> None:
        config.validate_hardware()
        self.config = config
        self.device = torch.device(config.runtime.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[config.runtime.dtype]
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.set_timesteps(
            config.sampling.num_inference_steps, device=self.device
        )
        if config.runtime.vae_slicing:
            pipeline.enable_vae_slicing()
        if config.runtime.vae_tiling:
            pipeline.enable_vae_tiling()
        if config.runtime.cpu_offload == "none":
            pipeline.to(self.device)
        else:
            gpu_id = self.device.index or 0
            if config.runtime.cpu_offload == "model":
                pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
            else:
                pipeline.enable_sequential_cpu_offload(gpu_id=gpu_id)
        pipeline.unet.eval()
        pipeline.text_encoder.eval()
        pipeline.vae.eval()
        self._pipeline = pipeline

    @property
    def pipeline(self) -> StableDiffusionPipeline:
        return self._pipeline

    @property
    def scheduler(self) -> DDIMScheduler:
        return self._pipeline.scheduler

    @property
    def expected_timesteps(self) -> tuple[int, ...]:
        return tuple(int(step) for step in self.scheduler.timesteps.tolist())

    @property
    def scheduler_config(self) -> dict[str, Any]:
        return json.loads(json.dumps(dict(self.scheduler.config), allow_nan=False))

    def validate_artifact(self, artifact: InversionArtifact) -> None:
        artifact.validate_compatibility(
            self.config,
            dataset_ref=artifact.dataset_ref,
            dataset_fingerprint=artifact.dataset_fingerprint,
            timesteps=self.expected_timesteps,
            scheduler_config=self.scheduler_config,
        )

    def _encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        tokenizer = self.pipeline.tokenizer
        tokens = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=False,
            return_tensors="pt",
        )
        if tokens.input_ids.shape[-1] != tokenizer.model_max_length:
            raise ValueError("Caption exceeds the model's CLIP token limit")
        return self.pipeline.text_encoder(
            tokens.input_ids.to(self.device)
        )[0]

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

    def _prepare_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda" and latent.device.type == "cpu":
            if self.config.runtime.pin_memory and not latent.is_pinned():
                latent = latent.pin_memory()
            return latent.to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=self.config.runtime.pin_memory,
            )
        return latent.to(device=self.device, dtype=self.dtype)

    def _install_processors(
        self, controller: PromptToPrompt
    ) -> dict[str, Any]:
        unet = self.pipeline.unet
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
                self.config.runtime.attention_query_chunk_size,
                name,
            )
        if not replacement or not any(
            name.endswith(".attn2.processor") for name in replacement
        ):
            raise ValueError("The loaded UNet has no controllable cross-attention layers")
        try:
            unet.set_attn_processor(dict(replacement))
        except Exception:
            unet.set_attn_processor(dict(original))
            raise
        return original

    @torch.inference_mode()
    def edit_batch(
        self,
        samples: Sequence[Mapping[str, Any]],
        artifacts: Sequence[InversionArtifact],
        hooks: Sequence[DenoisingHook | None] | None = None,
        *,
        prompt_to_prompt_class: type[PromptToPrompt] = PromptToPrompt,
        method_settings: Mapping[str, Any] | None = None,
    ) -> list[EditResult]:
        """Denoise up to ``runtime.batch_size`` source/target pairs together."""
        batch_size = len(samples)
        if not batch_size or batch_size > self.config.runtime.batch_size:
            raise ValueError(
                f"edit_batch requires 1 to {self.config.runtime.batch_size} samples"
            )
        if len(artifacts) != batch_size:
            raise ValueError("Each dataset sample needs one inversion artifact")
        if hooks is None:
            hooks = [None] * batch_size
        if len(hooks) != batch_size:
            raise ValueError("Each dataset sample needs one hook entry")
        if not isinstance(prompt_to_prompt_class, type) or not issubclass(
            prompt_to_prompt_class, PromptToPrompt
        ):
            raise TypeError("prompt_to_prompt_class must subclass PromptToPrompt")
        validated_settings = prompt_to_prompt_class.validate_method_settings(
            {} if method_settings is None else method_settings
        )
        if not isinstance(validated_settings, Mapping):
            raise TypeError("Prompt-to-Prompt settings validator must return a mapping")
        for sample, artifact, hook in zip(samples, artifacts, hooks):
            if sample.get("uid") != artifact.sample_uid:
                raise ValueError(
                    f"Artifact UID {artifact.sample_uid!r} does not match dataset sample"
                )
            self.validate_artifact(artifact)
            if artifact.requires_denoising_hook and hook is None:
                raise ValueError(
                    f"Artifact {artifact.sample_uid!r} has method-specific per-step "
                    "state but no denoising hook was provided"
                )
            if hook is not None and not isinstance(hook, DenoisingHook):
                raise TypeError("hooks must contain DenoisingHook instances or None")

        tokenizer = self.pipeline.tokenizer
        max_length = tokenizer.model_max_length
        source_prompts: list[str] = []
        target_prompts: list[str] = []
        for sample in samples:
            source = sample.get("source_prompt")
            target = sample.get("target_prompt")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Dataset source_prompt must be a nonempty string")
            if not isinstance(target, str) or not target.strip():
                raise ValueError("Dataset target_prompt must be a nonempty string")
            source_prompts.append(source)
            target_prompts.append(target)
        controller = prompt_to_prompt_class(
            self.config.prompt_to_prompt,
            tokenizer,
            source_prompts,
            target_prompts,
            artifacts,
            num_steps=len(self.expected_timesteps),
            max_length=max_length,
            device=self.device,
            settings=validated_settings,
        )
        prompts = source_prompts + target_prompts
        prompt_embeddings = self._encode_prompts(prompts)
        negative_prompt_embeddings = self._encode_prompts([""] * (2 * batch_size))
        base_latents = torch.cat(
            [self._prepare_latent(artifact.terminal_latent) for artifact in artifacts]
        )
        latents = torch.cat((base_latents, base_latents.clone()), dim=0)
        generators = [
            self._noise_generator(self.config.sampling.seed, str(sample["uid"]), self.device)
            for sample in samples
        ]

        original_processors = self._install_processors(controller)
        try:
            for step_index, timestep in enumerate(self.scheduler.timesteps):
                step_number = int(timestep.item())
                controller.step_index = step_index
                variance_noise = None
                if self.config.sampling.eta > 0:
                    source_noise = torch.stack(
                        [
                            torch.randn(
                                latents[index].shape,
                                generator=generator,
                                device=self.device,
                                dtype=self.dtype,
                            )
                            for index, generator in enumerate(generators)
                        ]
                    )
                    variance_noise = torch.cat(
                        (source_noise, source_noise), dim=0
                    )
                latents, prompt_embeddings, negative_prompt_embeddings, variance_noise = self._apply_hooks(
                    hooks, "before", step_index, step_number,
                    latents, prompt_embeddings, negative_prompt_embeddings,
                    variance_noise,
                )
                model_input = torch.cat((latents, latents), dim=0)
                model_input = self.scheduler.scale_model_input(model_input, timestep)
                embeddings = torch.cat(
                    (negative_prompt_embeddings, prompt_embeddings), dim=0
                )
                prediction = self.pipeline.unet(
                    model_input, timestep, encoder_hidden_states=embeddings
                ).sample
                unconditional, conditional = prediction.chunk(2)
                guided = unconditional + self.config.sampling.guidance_scale * (
                    conditional - unconditional
                )
                step_kwargs: dict[str, Any] = {"eta": self.config.sampling.eta}
                if variance_noise is not None:
                    step_kwargs["variance_noise"] = variance_noise
                latents = self.scheduler.step(
                    guided, timestep, latents, **step_kwargs
                ).prev_sample
                latents, prompt_embeddings, negative_prompt_embeddings, _ = self._apply_hooks(
                    hooks, "after", step_index, step_number,
                    latents, prompt_embeddings, negative_prompt_embeddings,
                    None,
                )
        finally:
            self.pipeline.unet.set_attn_processor(dict(original_processors))

        decoded = self.pipeline.vae.decode(
            latents / self.pipeline.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        images = self.pipeline.image_processor.postprocess(
            decoded, output_type="pil"
        )
        return [
            EditResult(reconstructed=images[index], edited=images[batch_size + index])
            for index in range(batch_size)
        ]
