"""Shared model runtime and dataset identity for inversion adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ..config import ExperimentConfig

if TYPE_CHECKING:
    from ..generation import Editor
    from .components import SharedInversionComponents


@dataclass(frozen=True, slots=True)
class InversionContext:
    """Shared model runtime and dataset identity passed to method adapters."""

    editor: Editor
    config: ExperimentConfig
    dataset_ref: str
    dataset_fingerprint: str
    components: SharedInversionComponents | None = None
    dataset_content: str | None = None

    def source_identity(self, *, components: tuple[str, ...] | None = None) -> dict[str, Any]:
        """Semantic inputs shared by final and intermediate inversion caches."""
        return {
            "dataset_ref": self.dataset_ref,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_content": self.dataset_content,
            "model": asdict(self.config.model),
            "model_content": dict(self.editor.model_content_identity if components is None
                                  else self.editor.component_content_identity(components)),
            "numerics": self.editor.inversion_cache_settings(),
        }
