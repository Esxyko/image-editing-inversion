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

The original combined artifact contains PIE-Bench records first, followed by the
retained MagicBrush source train and dev records. Assembly requires one matching
caption pair for every retained MagicBrush edit and rejects duplicate UIDs.

## Eval and dev splits

The assembled Hub dataset's original `train` split is partitioned into `eval`
and `dev`. Stage numbers refer to the MagicBrush UID's `turn_index` suffix;
related stages share the same `img_id`. Membership is determined from the
original assembled dataset, after its image-size filtering.

- `eval` retains all PIE-Bench records and every MagicBrush first turn.
- A MagicBrush second turn is initially eligible for `eval` only when its image
  ID has no first turn but does have a third turn in the original assembled dataset.
- Other second turns and every third turn move to `dev`.

The eligible second turns are then reduced to balance `eval` at **700 replacement
and 700 refinement pairs**. Replacement means equal source/target prompt word
counts using `str.split()`; refinement means different word counts. Rank the 554
eligible replacement and 1,088 eligible refinement second turns separately by:

1. Original instruction word count, `len(instruction.split())`, ascending.
2. Original instruction character count, `len(instruction)`, ascending, including
   spaces and punctuation.
3. Encoded `source_img` file size in bytes, descending.
4. Original assembled record position, ascending, to break any remaining ties.

Move the first **249 replacement** and **732 refinement** records into `dev`.
Instructions are matched from preserved caption-generation metadata by image ID
and source/target captions. Selection includes 242 replacement records below
(6 words, 28 characters), followed by the 7 largest source files at that length;
and 729 refinement records below (7 words, 35 characters), followed by the 3
largest source files at that length. Ranking selects records to move; each split
retains the original assembled relative order.

The final `eval` contains 1,400 records: all 700 PIE-Bench records, all 39
MagicBrush first turns, and 661 second turns (305 replacement, 356 refinement).
The final `dev` contains 3,896 records. The union still contains all 5,296 records.
The balanced split was published in
[revision f51282b](https://huggingface.co/datasets/beatle-ju1ce/image-editing-inversion/commit/f51282b798112305f810e121bfc7c0dccb4f4e62).

Every original record belongs to exactly one split. Both preserve the original
five columns, image bytes, UIDs, and relative record order. The runtime loads
only `eval`; `dev` remains available on the Hub separately.
Both `diffuse` and `experiment` use this shared loader. Their `--dataset`
argument selects PIE-Bench or MagicBrush records within `eval`.

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
working directory. The first load downloads the private Hub eval split;
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

`experiment --h-params default.yaml --nsamples N` runs every discovered method
on the first `N` records in dataset order, optionally selecting a source with
`--dataset` first. Python `run_methods` and `WorkflowRunner.run_methods` accept
the same limit through the optional `nsamples` keyword. The count must be a
positive integer; `None` selects all matching records. Counts exceeding the
available records select all of them. Each MagicBrush turn counts as one record.
This limit applies before batching and cache lookup, so cached records count
toward it. Every method processes the same prefix with the combined dataset's
identity, and artifacts outside the prefix are preserved.

See [README](../README.md) for setup and [artifact reuse](artifacts.md#cache-reuse)
for how dataset identity affects inversion caching.
