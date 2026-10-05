"""Select datasets and resolve samples by their stable UIDs."""

from __future__ import annotations

from typing import Any, Mapping

from datasets import Dataset

from .identity import DatasetContentIdentity
from .._project import project_paths
from .loader import load_project_dataset
from .schema import HUB_DATASET_REF, REQUIRED_COLUMNS


class DatasetRepository:
    """Own one dataset's selection, identity, and UID index."""

    def __init__(self) -> None:
        resolved_path = project_paths().data
        self.dataset = load_project_dataset()
        self.dataset_ref = HUB_DATASET_REF
        self.dataset_location = HUB_DATASET_REF
        if not isinstance(self.dataset, Dataset):
            raise ValueError("The dataset source must contain one Hugging Face Dataset.")
        if set(self.dataset.column_names) != REQUIRED_COLUMNS:
            raise ValueError("The dataset does not have the project's five expected columns.")
        self.dataset_fingerprint = str(getattr(self.dataset, "_fingerprint", ""))
        if not self.dataset_fingerprint:
            raise ValueError("The dataset has no fingerprint for artifact validation.")
        uids = self.dataset["uid"]
        if any(not isinstance(uid, str) or not uid.strip() for uid in uids):
            raise ValueError("The dataset contains invalid UIDs.")
        self._uid_to_index = {uid: index for index, uid in enumerate(uids)}
        if len(self._uid_to_index) != len(uids):
            raise ValueError("The dataset contains duplicate UIDs.")
        self._content_identity = DatasetContentIdentity(self.dataset, resolved_path)
        self._content_digest: str | None = None

    @property
    def content_digest(self) -> str:
        """Read actual dependency bytes once, sharing the digest across a sweep."""
        if self._content_digest is None:
            self._content_digest = self._content_identity.digest()
        return self._content_digest

    @property
    def uids(self) -> tuple[str, ...]:
        """Return every UID in dataset order without materializing images."""
        return tuple(self._uid_to_index)

    def sample(self, uid: str) -> Mapping[str, Any]:
        """Return a sample for a UID, rejecting identifiers outside the dataset."""
        try:
            return self.dataset[self._uid_to_index[uid]]
        except KeyError as exc:
            raise ValueError(f"Dataset UID not found: {uid}") from exc

    def validate_identity(
        self,
        dataset_ref: str,
        dataset_fingerprint: str | None,
        sample_uid: str,
    ) -> None:
        """Check artifact provenance against the selected dataset."""
        if dataset_ref != self.dataset_ref:
            raise ValueError(
                f"Artifact dataset {dataset_ref!r} differs from "
                f"selected dataset {self.dataset_ref!r}."
            )
        if dataset_fingerprint != self.dataset_fingerprint:
            raise ValueError("Artifact dataset fingerprint differs from selected dataset.")
        if sample_uid not in self._uid_to_index:
            raise ValueError(f"Artifact UID not found in dataset: {sample_uid}")
