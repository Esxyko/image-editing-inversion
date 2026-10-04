"""Shared dataset identity and record columns."""

HUB_DATASET_REF = "beatle-ju1ce/image-editing-inversion"
LOCAL_DATASET_REF = "local:image-editing"
REQUIRED_COLUMNS = frozenset({
    "uid",
    "source_img",
    "mask_img",
    "source_prompt",
    "target_prompt",
})
