"""Public generation results, attention policies, and shared editor.

The editor is imported on demand so attention policy adapters do not need to
import Diffusers just to define their policy subclasses.
"""

from typing import TYPE_CHECKING

from .alignment import PairAttentionMap
from .prompt_to_prompt import PromptToPrompt
from .result import EditResult

if TYPE_CHECKING:
    from .editor import Editor

__all__ = ["Editor", "EditResult", "PairAttentionMap", "PromptToPrompt"]


def __getattr__(name: str):
    if name == "Editor":
        from .editor import Editor

        globals()[name] = Editor
        return Editor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
