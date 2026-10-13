"""Diffusers attention integration for source/target image pairs."""

from __future__ import annotations

from contextlib import ExitStack
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
    processor. Uncontrolled branches retain the original processor's outputs;
    active attention maps are computed in query chunks only for target transfer.
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
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
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

    def _native_attention_with_projections(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        temb: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse native Q/K/V without changing its batch or numerical path."""
        projections: dict[str, torch.Tensor] = {}

        def capture(name: str):
            def record(
                module: Any, inputs: tuple[Any, ...], output: torch.Tensor,
            ) -> None:
                projections[name] = output

            return record

        # Scope observers to one native call, including its failure path. They
        # retain the actual tensors without copying or modifying module outputs.
        with ExitStack() as observers:
            for name in ("to_q", "to_k", "to_v"):
                handle = getattr(attn, name).register_forward_hook(capture(name))
                observers.callback(handle.remove)
            original_output = self.original(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb,
            )
        if len(projections) != 3:
            raise ValueError("Native attention did not expose all Q/K/V projections")
        return (
            original_output, projections["to_q"], projections["to_k"],
            projections["to_v"],
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

        if hidden_states.ndim not in (3, 4):
            raise ValueError("Unsupported attention hidden-state rank")
        pairs = self.controller.batch_size
        batch_size = hidden_states.shape[0]
        if batch_size != 4 * pairs:
            raise ValueError(
                f"Prompt-to-Prompt expected {4 * pairs} UNet branches, got {batch_size}"
            )
        # Keep the native attention, normalization, projection and residual path
        # for both unconditional branches and the conditional source branch.
        original_output, query, key, value = self._native_attention_with_projections(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb,
        )
        residual = hidden_states[3 * pairs:]
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channels, height, width = hidden_states.shape
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
            (pairs, *query.shape[1:-1], value.shape[-1]),
            device=query.device,
            dtype=value.dtype,
        )

        query_length = query.shape[-2]
        if type(self.controller) is PromptToPrompt:
            if self.is_cross:
                self._default_cross_attention(query, key, value, prepared_mask, attn.scale, output)
            else:
                self._default_self_attention(query, key, value, prepared_mask, attn.scale, output)
        else:
            self._custom_attention(query, key, value, prepared_mask, attn.scale, output)

        hidden_states = output.transpose(1, 2).reshape(pairs, query_length, -1)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).reshape(
                pairs, channels, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        target_output = hidden_states / attn.rescale_output_factor
        return torch.cat((original_output[:3 * pairs], target_output))


    def _default_self_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        mask: torch.Tensor | None, scale: float, output: torch.Tensor,
    ) -> None:
        pairs = self.controller.batch_size
        source = slice(2 * pairs, 3 * pairs)
        source_mask = None if mask is None else mask[source]
        output[:] = F.scaled_dot_product_attention(
            query[source], key[source], value[3 * pairs:], attn_mask=source_mask,
            dropout_p=0.0, is_causal=False, scale=scale,
        )

    @staticmethod
    def _batch_mask(
        mask: torch.Tensor | None, branches: torch.Tensor,
    ) -> torch.Tensor | None:
        return None if mask is None else mask.index_select(0, branches)

    @staticmethod
    def _query_mask(
        mask: torch.Tensor | None, start: int, end: int,
    ) -> torch.Tensor | None:
        if mask is None or mask.shape[-2] == 1:
            return mask
        return mask[..., start:end, :]

    def _default_cross_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        mask: torch.Tensor | None, scale: float, output: torch.Tensor,
    ) -> None:
        pairs = self.controller.batch_size
        # Bound score tensors by the old per-pair query budget. Limit projection
        # copies to four pairs even when the configured editing batch is large.
        pair_limit = min(4, self.query_chunk_size)
        for group in self.controller._batch_maps:
            for offset in range(0, group.pair_indices.numel(), pair_limit):
                indices = group.pair_indices[offset:offset + pair_limit]
                source_branches = indices + 2 * pairs
                target_branches = indices + 3 * pairs
                source_query = query.index_select(0, source_branches)
                source_key = key.index_select(0, source_branches)
                target_value = value.index_select(0, target_branches)
                source_mask = self._batch_mask(mask, source_branches)
                mapping = group.mapping[offset:offset + pair_limit]
                if group.mode == "refine":
                    target_query = query.index_select(0, target_branches)
                    target_key = key.index_select(0, target_branches)
                    target_mask = self._batch_mask(mask, target_branches)
                    assert group.alpha is not None
                    alpha = group.alpha[offset:offset + pair_limit, None, None, :]
                chunk_size = max(1, self.query_chunk_size // indices.numel())
                for start in range(0, query.shape[-2], chunk_size):
                    end = min(start + chunk_size, query.shape[-2])
                    source_prob = self._probabilities(
                        source_query[..., start:end, :], source_key, scale,
                        self._query_mask(source_mask, start, end),
                    )
                    if group.mode == "replace":
                        replacement = torch.matmul(source_prob, mapping[:, None])
                    else:
                        target_prob = self._probabilities(
                            target_query[..., start:end, :], target_key, scale,
                            self._query_mask(target_mask, start, end),
                        )
                        gather_indices = mapping[:, None, None, :].expand(
                            source_prob.shape[0], source_prob.shape[1], end - start, -1,
                        )
                        aligned = source_prob.gather(-1, gather_indices)
                        replacement = alpha * aligned + (1.0 - alpha) * target_prob
                    output[indices, :, start:end] = torch.matmul(
                        replacement.to(target_value.dtype), target_value,
                    )

    def _custom_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        prepared_mask: torch.Tensor | None, scale: float, output: torch.Tensor,
    ) -> None:
        """Keep the original per-pair extension behavior for every policy subclass."""
        pairs = self.controller.batch_size
        query_length = query.shape[-2]
        for pair_index in range(pairs):
            source_branch = 2 * pairs + pair_index
            target_branch = 3 * pairs + pair_index
            source_key = key[source_branch]
            target_key = key[target_branch]
            target_value = value[target_branch]
            for start in range(0, query_length, self.query_chunk_size):
                end = min(start + self.query_chunk_size, query_length)
                source_prob = self._probabilities(
                    query[source_branch, :, start:end], source_key, scale,
                    self._mask_slice(prepared_mask, source_branch, start, end),
                )
                if self.is_cross or (
                    type(self.controller).replace_self_attention
                    is not PromptToPrompt.replace_self_attention
                ):
                    target_prob = self._probabilities(
                        query[target_branch, :, start:end], target_key, scale,
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
                output[pair_index, :, start:end] = torch.bmm(
                    replacement_prob.to(target_value.dtype), target_value
                )
