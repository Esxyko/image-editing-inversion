"""Token alignment for the default Prompt-to-Prompt attention policy."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import torch


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
