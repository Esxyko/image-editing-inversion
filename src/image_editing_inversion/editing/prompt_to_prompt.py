"""Subclassable Prompt-to-Prompt attention transfer policy."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from ..artifacts import InversionArtifact
from ..config import PromptToPromptConfig
from .alignment import PairAttentionMap, _pair_attention_map


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
