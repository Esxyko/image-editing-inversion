"""Reconstructed and edited images for one sample."""

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True, slots=True)
class EditResult:
    """The reconstruction and edit for one dataset sample."""

    reconstructed: Image.Image
    edited: Image.Image
