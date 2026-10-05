# Dataset reference

Each record represents an edit with a source image, a binary edit mask, and a
pair of captions describing the source and intended result. The edited target
image was used to prepare MagicBrush captions but is not stored in the Hub
dataset.

## Construction

- **PIE-Bench:** All 700 entries in `mapping_file.json` are read in sorted
  sample-ID order. Each 512 × 512 source image is converted to RGB. Its
  run-length encoded edit mask is decoded to a 512 × 512 single-channel image,
  with the image boundary marked white. `original_prompt` and `editing_prompt`
  supply the caption pair after surrounding brackets are removed. The mapping
  key becomes the record UID.
- **MagicBrush:** The public train and dev records are combined in that order.
  Only edits with 1024 × 1024 source images are retained; their target images
  and masks must have matching dimensions. Source and target images are resized
  to 512 × 512 RGB. Masks are resized and thresholded to black or white; for
  RGBA masks, transparent pixels mark the edit area. A source and target caption
  pair is generated from the processed images and edit instruction, then matched
  back to the edit using its content fingerprint. Each UID is
  `<img_id>:<turn_index>`.

The combined artifact contains PIE-Bench records first, followed by the
retained MagicBrush train and dev records. Assembly requires one matching
caption pair for every retained MagicBrush edit and rejects duplicate UIDs.

## Record format

| Column | Value |
| --- | --- |
| `uid` | Non-empty source record identifier |
| `source_img` | 512 × 512 RGB image |
| `mask_img` | 512 × 512 single-channel mask; black is 0 and white is 255 |
| `source_prompt` | Non-empty caption for the source image |
| `target_prompt` | Non-empty caption for the intended edit |

## Loading and identity

```python
from image_editing_inversion.data import DatasetRepository, load_project_dataset

records = load_project_dataset()
repository = DatasetRepository()
print(repository.uids[0])
sample = repository.sample(repository.uids[0])
```

Both APIs use the fixed `data/` directory beside `src/`, independent of the
working directory. The first load downloads the private Hub train split;
subsequent loads use saved Arrow shards. Access requires `HF_TOKEN`, the root
`.env` file, or an existing Hugging Face login. Incomplete saved datasets fail
clearly; repair their dataset files while preserving caches, artifacts, and outputs.

The repository validates the five-column schema, nonempty unique UIDs, and the
Hugging Face fingerprint. Ordered UID access does not decode images. Content
identity streams saved Arrow dependencies and raw external source-image files
once per invocation. Effective features and format affect the digest; dataset
citations, descriptions, JSON formatting, and external mask contents do not.
Changes within Arrow shards can conservatively invalidate every sample, including
changes to target prompts or embedded masks. The first download is reloaded from
saved shards so first and later invocations identify the same files.

Dataset-building code is retained only in the ignored local archive. The runtime
package loads the assembled dataset; it does not rebuild captions or benchmark records.

## Source selection

`diffuse --dataset pie-bench` and `diffuse --dataset magic-brush` select records
from the combined dataset. Omitting the argument processes every record.
Python callers can pass the same values as the optional `dataset` keyword to
`run_methods` or `WorkflowRunner.run_methods`; the default is `None`.

`DatasetRepository.select_uids(dataset)` returns the selected UIDs in their
existing dataset order without decoding images. PIE-Bench UIDs contain no colon;
MagicBrush UIDs use `<img_id>:<turn_index>` and contain a colon. Unsupported
selectors raise `ValueError`. A workflow with no matching records fails before
model loading with an error identifying the selected source.

Selection applies to inversion, reconstruction, and editing across all selected
methods and parameter files. It does not modify the five-column dataset, UID
index, fingerprint, or content digest. Full and filtered runs share cache
identities; artifact publication preserves existing records from other sources.
The top-level `dataset` field in `sweep.json` records the selector, or `null` for
an unfiltered invocation. Saved-artifact replay has no source selector.

See [README](../README.md) for setup and [artifact reuse](artifacts.md#cache-reuse)
for how dataset identity affects inversion caching.
