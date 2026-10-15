# Run outputs, archives, and timing

Generation and replay write a new directory under `data/output/` for each
invocation. Inversion artifacts are stored separately so they can be reused even
if an output run is archived or removed.

- [Run directory](#run-directory)
- [RGB tensor outputs](#rgb-tensor-outputs)
- [Results and timing](#results-and-timing)
- [Run archives](#run-archives)

## Run directory

```text
data/output/<run-id>/
|-- sweep.json
`-- <parameter-file>/<method>/
    |-- resolved-config.json
    |-- results.jsonl
    |-- batches.jsonl
    `-- tensors/
        `-- batch-<edit_batch_id>.safetensors
```

| File | Contents |
| --- | --- |
| `sweep.json` | Schema-v2 invocation manifest, selections, and completion status |
| `resolved-config.json` | Complete model, sampling, attention, method, and runtime settings |
| `results.jsonl` | Per-sample status, artifact reference, tensor position, and batch associations |
| `batches.jsonl` | Batch durations and counts; use this for aggregate timing |
| `tensors/*.safetensors` | Reconstructed and edited RGB pixel batches |

The manifest records `dataset` as `pie-bench`, `magic-brush`, or `null`. Limited
inversion runs also record the requested positive `nsamples`; unlimited and
replay runs omit that field. Each invocation returns its parent run directory.

See [workflows](workflows.md) for commands and
[Null-text diagnostics](null-text-diagnostics.md) for optional additional files.

## RGB tensor outputs

New inversion and replay runs save decoded RGB pixels instead of PNG images.
VAE decoding, image-processor denormalization and clamping still run; normalized
values are converted to float32, multiplied by 255, rounded and converted to
uint8 as in the previous PNG pixel path. `Editor.edit_batch` returns `EditResult`
objects with contiguous CPU tensors `[3, height, width]` for `reconstructed` and
`edited`, rather than PIL images. Outputs are pixels, not VAE latents or CLIP embeddings.

Each successful editing batch writes
`tensors/batch-<edit_batch_id>.safetensors` under its method output directory:

| Field | Contract |
| --- | --- |
| Tensor keys | `reconstructed`, `edited` |
| Tensor dtype and shape | uint8 `[N, 3, height, width]`, CPU RGB pixels |
| Metadata | String-valued `schema_version: "1"`, JSON `uids`, `color_space: "RGB"`, `layout: "NCHW"`, `value_range: "0..255"`, `method_id`, `pipeline_h_params_file`, `edit_batch_id` |
| Successful sample record | Raw `uid`, relative `tensor_file`, zero-based `tensor_index`, and existing artifact/batch references |

`uids[i]` identifies position `i` in both tensors. Only compatible, actually
edited samples enter that list, including when a replay batch skips other
references. The temporary batch file is atomically published before successful
sample records are appended. Write failures remove the temporary file and retain
previously published batches; a failure during later result recording may leave
a complete tensor batch whose metadata still identifies every member. Existing
batch files are never overwritten. New result records have no
`reconstructed_image` or `edited_image` fields. Historical PNG files and records
remain unchanged; consumers of new outputs must use the tensor references.

Generation performs no PNG compression or whole-run aggregation. Memory stays
bounded by the editing batch, and a tail batch stores its actual sample count. Two 512x512
RGB results use 1.5 MiB per sample, about 8.2 GiB for 1,400 samples across four methods.

Read one sample without materializing the entire output batch:

```python
import json
from pathlib import Path
from safetensors import safe_open

method_dir = Path("data/output/<run-id>/default.yaml/ddim")
uid = "435242:2"  # Use a UID present in this method's results.
with (method_dir / "results.jsonl").open(encoding="utf-8") as records:
    row = next(record for line in records
               if (record := json.loads(line))["uid"] == uid and record["status"] == "ok")
index = row["tensor_index"]
with safe_open(str(method_dir / row["tensor_file"]), framework="pt", device="cpu") as saved:
    uids = json.loads(saved.metadata()["uids"])
    if uids[index] != uid:
        raise ValueError("Tensor UID does not match the result record")
    reconstructed_chw = saved.get_slice("reconstructed")[index:index + 1].squeeze(0).clone()
    edited_chw = saved.get_slice("edited")[index:index + 1].squeeze(0).clone()
edited_hwc = edited_chw.permute(1, 2, 0).contiguous()  # RGB uint8 [H, W, 3]
edited_float = edited_chw.float() / 255.0            # RGB float32 [3, H, W] in [0, 1]
```

## Results and timing

`edit_batch_seconds` retains the preparation-plus-editor wall time, excluding input
reads and tensor writes. Both legacy batch durations repeat on member records;
use `batches.jsonl` for aggregate timing instead of summing those repeated fields.
`results.jsonl` additionally records `inversion_batch_id` and `edit_batch_id`.

### Null-text result fields

Null-text results include `actual_num_inner_steps`, the list of actual Adam update
counts in the artifact's descending timestep order, and `total_inner_steps`, their
sum. These counts describe the original inversion and are retained for cache hits
and replay. Counts are zero when optimization/editing CFG is 1. Artifacts must
carry `optimization_guidance_scale`, equal to the current editing CFG; source
reconstruction uses that saved scale too. Earlier artifacts optimized at
inversion CFG require regeneration. Null-text artifacts also carry
`pivot_guidance_scale`, equal to inversion CFG. Earlier fixed-CFG-1 pivot artifacts
without that state require regeneration; pivot-guidance versioning invalidates
their automatic cache reuse. Compatible artifacts without counts report
`null` for both fields; newly computed inversions persist counts in per-step state.
Null-text results also include `reconstruction_mse`, the per-timestep latent MSE
after all Adam updates, and `final_reconstruction_mse`, its last entry. Each error
compares the optimized trajectory latent against the corresponding target pivot
in float32; the final error compares against the source image's encoded latent.
These values describe the original inversion, not decoded-image pixel error or
a new replay measurement. They are persisted with the artifact and retained for
cache hits and replay. Artifacts with complete matching snapshots but no saved
errors remain reusable and report `null` for both error fields. Recompute
inversion to populate them.

### Batch records and associations

Cache hits retain their lookup batch ID, replay-only inversions use `null`, and
skipped samples have no edit batch ID (`null`).

Each attempted inversion batch, editing method segment, and group publication
writes exactly one schema-v2 record to `batches.jsonl`, including failed attempts.
Common fields are `batch_id`, `phase`, `sample_count`, `status`, and
`durations_seconds`, with the method ID and parameter filename identifying its
output. Errors include an `error` message. Stages not reached have zero duration.

| Phase | Duration keys | Counts |
| --- | --- | --- |
| `inversion` | `input_read`, `cache_lookup`, `method`, `validation`, `staging` | `sample_count` is the requested chunk size; `cache_hits` and `uncached_count` describe resolved lookups |
| `edit` | `input_read`, `preparation`, `editor`, `tensor_write` | `sample_count` is the requested method segment; `edited_count` is the number prepared for editing |
| `publication` | `publication` | `sample_count` is the number traversed in the inversion pass |

Cache lookup time includes cache identity construction, loading, and validation
of cached entries; the first batch also includes opening and checking the existing
published group's integrity. Inversion `input_read` measures only missing-sample
reads and stays zero for batches containing only cache hits.
The `method` duration measures the adapter call, including its
internal image encoding, validation, and device transfers; it is not pure UNet
time. `validation` covers the workflow's additional checks on fresh artifacts.
Publication covers assembly, checksum computation, and catalog updates and is
never repeated on sample records. Readers preserve existing edit chunk and method
boundaries; a segment containing only incompatible artifacts is recorded as
`skipped` with `edited_count: 0`. CUDA/MPS model stages synchronize at their timing
boundaries; CPU stages use wall-clock timing. Timing adds no per-timestep
synchronization, although numerical validation and early-stop decisions may synchronize.
Model loading and preflight are outside these batch durations.

Historical schema-v1 batch records retain `image_write`; new schema-v2 records
use `tensor_write`. The remaining duration keys and legacy per-sample duration
definitions are unchanged. RGB quantization and the transfer to CPU are part of
`editor`; batch assembly and atomic file publication are part of `tensor_write`.

Inversion batch diagnostics also include `source_cache_hits`, `source_cache_misses`,
`pivot_cache_hits`, and `pivot_cache_misses`, counted per image, plus
`intermediate_cache_write_errors` containing component, UID, and error details.
Intermediate lookup, computation, and storage are included in the `method`
duration. Final-artifact hits bypass intermediate work and leave these counts at
zero. Reusing intermediates alone does not set `artifact_reused` to `true`.

## Run archives

`workflows.compress_run(run_id: str) -> Path` archives one inactive run under the
fixed `data/output/` root. It accepts only a directory name and requires a valid
`sweep.json` with top-level status `ok` or `error`; missing, invalid, pending, and
running manifests are rejected. The source must be a direct child of output,
with only regular files and directories. Symbolic links, directory junctions,
and special files are rejected before writing.

The sibling `data/output/<run-id>.zip` contains the `<run-id>/` root and all
contents, including hidden files and empty directories. DEFLATE level 6 and
ZIP64 support are enabled; file data is streamed rather than loaded as tensors.
The archive contains outputs only, not the separate dataset, inversion artifacts,
or intermediate caches. Extract it before using the ordinary tensor-loading
examples and result records' relative paths.

Compression writes a unique hidden temporary file beside the run. Once closed,
the archive is checked for the exact member list, uncompressed sizes, directory
types, and every member's CRC. Source snapshots include names, file sizes,
modification/change times, and filesystem identities; changes abort publication.
The verified temporary file atomically replaces any existing sibling ZIP. An
existing destination must be a regular file, not a link, junction, or directory.
Before publication, failures preserve the source and any old ZIP and attempt to
remove the temporary file; cleanup errors are attached to the original exception.

After publication, source containment and the snapshot are checked again before
recursive deletion. The API returns the absolute ZIP path only after deletion
succeeds. If that final check or deletion fails, the new verified ZIP remains;
the exception identifies its path and warns that source deletion may be partial.
User interruption during cleanup carries the same information as an exception
note. A process termination may leave a temporary file or a partly removed
source, depending on the stage reached.

Callers must ensure there are no concurrent writers to the run or its ZIP. A
terminal manifest and snapshot checks are not a cross-process lock. Archiving is
explicit and does not run automatically after generation or replay.

See [archiving a finished run](workflows.md#compress-a-finished-run) for the
Python call and [evaluation](evaluation.md) for reading archived RGB batches
without extraction.
