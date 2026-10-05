"""Resolve fixed project locations relative to the package's src directory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Use the source layout's fixed project root, independent of cwd."""

    root: Path = field(init=False)

    def __post_init__(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = root / "config.yaml"
        if not config.is_file():
            raise ValueError(f"Expected project configuration beside src at {config}")
        object.__setattr__(self, "root", root)

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def pipeline_h_params(self) -> Path:
        return self.root / "pipeline_h_params"

    @property
    def method_h_params(self) -> Path:
        return self.root / "method_h_params"

    @property
    def env(self) -> Path:
        return self.root / ".env"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def artifacts(self) -> Path:
        return self.data / "artifacts"

    @property
    def intermediates(self) -> Path:
        return self.data / "cache" / "inversion"

    @property
    def output(self) -> Path:
        return self.data / "output"

    def model_path(self, model_id: str) -> Path:
        path = Path(model_id).expanduser()
        return path if path.is_absolute() else self.root / path


def project_paths() -> ProjectPaths:
    return ProjectPaths()
