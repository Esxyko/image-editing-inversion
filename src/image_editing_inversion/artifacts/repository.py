"""Resolve raw sample IDs and discover indexed artifact collections."""

from pathlib import Path
from typing import Iterator

from .._project import project_paths
from .collection import ArtifactCollection, ArtifactReference
from .layout import is_pipeline_group, method_directory_name


class ArtifactRepository:
    """Select published artifacts without modifying storage or the catalog."""

    def __init__(self) -> None:
        self.root = project_paths().artifacts.resolve()

    def resolve(self, artifact_id: str) -> ArtifactReference:
        matches = self.resolve_all(artifact_id)
        if len(matches) > 1:
            raise ValueError(f"Artifact ID is ambiguous across method/parameter groups: {artifact_id}")
        return matches[0]

    def resolve_all(self, artifact_id: str) -> tuple[ArtifactReference, ...]:
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ValueError("Artifact ID must be a non-empty raw dataset UID")
        matches = tuple(collection.reference(artifact_id)
                        for collection in self._collections() if artifact_id in collection.ids)
        if not matches:
            raise ValueError(f"Artifact ID not found: {artifact_id}")
        return matches

    def discover(self) -> tuple[ArtifactReference, ...]:
        """Return group-path order, then each collection's positional order."""
        references = tuple(reference for collection in self._collections()
                           for reference in collection.references)
        if not references:
            raise ValueError(f"No schema-v5 inversion artifact collections found in {self.root}.")
        return references

    def _collections(self) -> Iterator[ArtifactCollection]:
        for method in self._children(self.root):
            try:
                method_directory_name(method.name)
            except ValueError:
                continue
            for group in self._children(method):
                path = group / ArtifactCollection.filename
                if is_pipeline_group(group.name) and path.is_file() and not path.is_symlink():
                    yield ArtifactCollection(path)

    def _children(self, parent: Path) -> Iterator[Path]:
        if not parent.is_dir():
            return
        for candidate in sorted(parent.iterdir(), key=lambda path: path.name):
            if candidate.name.startswith(".") or candidate.is_symlink() or not candidate.is_dir():
                continue
            path = candidate.resolve()
            if path.parent == parent and self.root in path.parents:
                yield path
