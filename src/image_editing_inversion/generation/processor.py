"""Diffusers attention integration for source/target image pairs."""

from __future__ import annotations

from typing import Any

from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
import torch
import torch.nn.functional as F

from .prompt_to_prompt import PromptToPrompt


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
