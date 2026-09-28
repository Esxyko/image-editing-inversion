# Image Editing Inversion

## Load the project dataset

The project dataset is the private Hugging Face repository
`beatle-ju1ce/image-editing-inversion`. Request access to it, then set `HF_TOKEN`
in your environment or add it to the project's ignored `.env` file. The
`.env.example` file shows the key to add. An existing Hugging Face CLI login
also works.

```python
from image_editing_inversion.dataset import load_project_dataset

records = load_project_dataset()
example = records[0]
print(example["uid"])
```

The first call downloads the repository's `train` split and saves its Hugging
Face Datasets files directly in `data/`. Later calls load that saved copy without
network access. The loader leaves the existing `data/raw/`, `data/cache/`,
`data/processed/`, and `data/final/` folders in place. The `data/` directory is
local and ignored by Git. A full first download needs several gigabytes of
free disk space for the download cache and saved dataset.

## Built dataset

The original locally built artifact is at `data/final/image-editing/`. The loader
above uses the saved dataset files at the `data/` root.
Each record represents an edit with a source image, a binary edit mask, and a
pair of captions describing the source and intended result. The edited target
image is used to prepare MagicBrush captions but is not stored in the final
artifact.

### How the records are constructed

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

### Record format

| Column | Value |
| --- | --- |
| `uid` | Non-empty source record identifier |
| `source_img` | 512 × 512 RGB image |
| `mask_img` | 512 × 512 single-channel mask; black is 0 and white is 255 |
| `source_prompt` | Non-empty caption for the source image |
| `target_prompt` | Non-empty caption for the intended edit |

Load a locally built artifact with Hugging Face Datasets:

```python
from datasets import load_from_disk

records = load_from_disk("data/final/image-editing")
example = records[0]
print(example["uid"])
```
