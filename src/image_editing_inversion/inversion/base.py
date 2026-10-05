"""Extension interface for inversion algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from PIL import Image

from ..artifacts import InversionArtifact
from .context import InversionContext
from .hooks import DenoisingHook

if TYPE_CHECKING:
    from ..config import ExperimentConfig
    from ..generation import PromptToPrompt


class InversionMethod(ABC):
    """Implement this interface to register an inversion algorithm."""

    @property
    @abstractmethod
    def method_id(self) -> str:
        """Stable ID written into artifacts and run records."""

    @property
    def max_edit_batch_size(self) -> int | None:
        """Largest shared-editor batch supported by the method's hook, if limited."""
        return None

    @property
    def prompt_to_prompt_class(self) -> type[PromptToPrompt] | None:
        """Optional P2P subclass for this method; ``None`` uses the shared base."""
        return None

    @abstractmethod
    def invert(
        self, sample: Mapping[str, Any], context: InversionContext
    ) -> InversionArtifact:
        """Invert a dataset sample using the resolved editing context."""

    def invert_batch(
        self, samples: Sequence[Mapping[str, Any]], context: InversionContext
    ) -> list[InversionArtifact]:
        """Return artifacts in input order; existing adapters execute sequentially."""
        return [self.invert(sample, context) for sample in samples]

    def _validate_batch(
        self, samples: Sequence[Mapping[str, Any]], context: InversionContext
    ) -> tuple[list[str], list[str], list[Image.Image]]:
        """Validate shared source inputs before any model calls."""
        if not 1 <= len(samples) <= context.config.runtime.inversion_batch_size:
            raise ValueError(
                f"{self.method_id} inversion requires 1 to "
                f"{context.config.runtime.inversion_batch_size} samples"
            )
        uids: list[str] = []
        prompts: list[str] = []
        images: list[Image.Image] = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise TypeError("Inversion sample must be a mapping")
            uid = sample.get("uid")
            if not isinstance(uid, str) or not uid.strip():
                raise ValueError("Dataset uid must be a nonempty string")
            prompt = sample.get("source_prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Dataset source_prompt must be a nonempty string")
            image = sample.get("source_img")
            if not isinstance(image, Image.Image):
                raise TypeError("Dataset source_img must be a Pillow image")
            uids.append(uid)
            prompts.append(prompt)
            images.append(image)
        return uids, prompts, images

    def validate_inversion_config(self, config: ExperimentConfig) -> None:
        """Optionally validate inversion settings without loading model weights."""

    def validate_inversion_context(self, context: InversionContext) -> None:
        """Optionally validate the actual scheduler before processing any samples."""

    def validate_replay(
        self, artifact: InversionArtifact, context: InversionContext
    ) -> None:
        """Optionally reject method-specific state incompatible with this run."""

    def inversion_cache_parameters(
        self, context: InversionContext
    ) -> Mapping[str, Any] | None:
        """Opt into reuse with effective inversion settings; None disables caching.

        Include any extra inputs affecting inversion, including the seed for
        stochastic methods. Shared model/scheduler/guidance inputs are supplied
        by the runner. Replay does not call this method.
        """
        return None

    def create_denoising_hook(
        self, artifact: InversionArtifact
    ) -> DenoisingHook | None:
        """Return a hook when the artifact has method-specific per-step state."""
        return None
