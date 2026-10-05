"""Shared dataset identity and record columns."""

HUB_DATASET_REF = "beatle-ju1ce/image-editing-inversion"
SOURCE_DATASETS = ("pie-bench", "magic-brush")
REQUIRED_COLUMNS = frozenset({
    "uid",
    "source_img",
    "mask_img",
    "source_prompt",
    "target_prompt",
})


def validate_dataset_source(dataset: str | None) -> None:
    """Reject unsupported source selectors without loading the project dataset."""
    if dataset is not None and dataset not in SOURCE_DATASETS:
        raise ValueError(
            f"Dataset source must be one of {SOURCE_DATASETS!r} or None; got {dataset!r}."
        )
