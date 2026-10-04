"""Extension interface for inversion algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping

from ..artifacts import InversionArtifact
from .context import InversionContext
from .hooks import DenoisingHook

if TYPE_CHECKING:
    from ..editing import PromptToPrompt


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

    def validate_replay(
        self, artifact: InversionArtifact, context: InversionContext
    ) -> None:
        """Optionally reject method-specific state incompatible with this run."""

    def create_denoising_hook(
        self, artifact: InversionArtifact
    ) -> DenoisingHook | None:
        """Return a hook when the artifact has method-specific per-step state."""
        return None
