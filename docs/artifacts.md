# Artifact storage, reuse, and replay

## Collection format

Generated inversions are stored independently in one combined file per method
and step/guidance group:

```text
data/artifacts/
|-- catalog.json
`-- <method>/steps-50_guidance-7.5/
    |-- h-params.json
    `-- artifacts.safetensors
```

Each image has a floating `artifacts/<i>/terminal_latent` tensor of shape
`[1, 4, height/8, width/8]` with positive dimensions and finite values. Optional
`artifacts/<i>/state/<name>` tensors have one entry per timestep and are interpreted
by the registered method's hook. Images, captions, and masks stay in the dataset.

Each schema-v5 file embeds JSON-encoded `ids` and aligned `entries` lists in its
safetensors metadata. `entries[i]` indexes a production record in the JSON-encoded
`contexts` list for `ids[i]`. Common production records are stored once, while
every UID retains its own source association.
The raw dataset UID at `ids[i]` identifies tensors under `artifacts/<i>/`.
UIDs such as `image:1` are stored verbatim. The method comes from the method
directory and steps/guidance from the group name. Guidance values are written
without rounding, with whole values such as `7.0` written as `7`.
`h-params.json` records `num_inference_steps` and `guidance_scale`.
Parameter filenames and attention replacement fractions do not affect grouping.
Method IDs must be lowercase portable directory names, using letters, digits,
dots, underscores, or hyphens, without reserved Windows names or a trailing dot.
Shared model, dataset, scheduler, eta, and method settings stay fixed within a
sweep. Collections preserve those original values; replay supplies independent
current expectations for compatibility checks.

`ArtifactCatalog.store_group` accepts ordered artifacts and an optional parallel
list of cache inputs. Non-null cache inputs must identify the matching `sample_uid`
and `method_id`. Its required `pipeline_h_params` contains steps and guidance.
Replacements retain their existing positions; new IDs append in traversal order.
The catalog uses schema v2 with per-ID input matching and one checksum per group
file. Matching includes dataset identity/UID, model and scheduler settings,
guidance, effective method hyperparameters, and numerical runtime settings.
Missing, damaged, or incompatible
cached entries are recomputed; malformed catalogs fail clearly.

## Publication and interruption

Diffusion first completes the inversion pass for one method/settings group,
reusing valid published inversions and batching only missing or stale IDs.
Final-artifact lookups run before dataset sample reads; only inversion misses
load samples during this pass. Cache hits load their samples in the editing pass.
New tensors are staged in hidden temporary batch files and streamed into the
combined file without keeping the whole dataset's tensors in memory. The group
file is atomically replaced once all inversions succeed, before the catalog is
updated. Editing then reads the published file using the independently configured
editing batch size. An editing failure leaves completed inversions available.
An interrupted inversion pass retains the previous published group, but new
unpublished work is recomputed on the next run. Temporary files are excluded
from discovery and cleaned up when the transaction exits normally or with an error.
Catalog updates assume one active writer. A failure between file publication and
catalog publication can leave an uncataloged file available for explicit replay;
stale cache entries will miss and require recomputation.

Only schema-v5 group files and catalog-v2 entries with cache identity version 2
are reusable. Schema-v4 files and older cache identities require regeneration;
`diffuse` treats them as misses and replaces them only after successful publication. Legacy
per-sample directories, hash-named groups, and catalog-v1 entries are ignored
without deletion. The old catalog is replaced only after successful new-format
publication. Both workflows write results to the fixed `data/output/` root.

## Cache reuse

Changing only attention fractions or editing policies reuses inversion. The
bundled deterministic methods also reuse inversion when the editing seed
changes. Changing steps or guidance creates a new group; changing other effective
inversion settings replaces the sample's artifact in its group. Method
hyperparameters are loaded for cache matching using the
same validated settings as inversion. Each run generates its images again.
Null-text matches the effective per-step stopping thresholds; an unused increment
for a one-step schedule does not invalidate inversion. ReNoise matches the
prediction interval selected at each actual timestep, omitting inactive averaging
windows, unused low-timestep caps, and predictions after the selected average.
Hyperparameter files still undergo the existing full validation. Inactive
scheduler clipping/thresholding values and unused beta defaults are also omitted.
Result records include the parameter filename, combined artifact path,
`artifact_index`, and `artifact_reused`; cache hits have `inversion_seconds: 0.0`.
Cache matching excludes inversion and editing batch sizes, device and hardware
identity, PyTorch builds, CPU thread counts, and dependency versions (including
CUDA/cuDNN versions). Changing only these values allows reuse when the remaining
inputs match. Matching retains dtype and VAE tiling, including effective tile sizes
and overlap, but excludes VAE slicing, CPU offload, TF32, deterministic-algorithm,
cuDNN, SDPA, and float32 matmul-precision switches. These differences may cause
small numerical changes, which do not invalidate inversion. Source-code hashes
and method class paths are excluded from final and
intermediate cache identities, so code changes do not automatically invalidate
cached results. Scheduler dependency-version metadata is
excluded, together with default-value provenance and class-name metadata;
effective scheduler settings are still matched.

Both cache layers also match the actual data and model contents.
`DatasetRepository.content_digest` streams the actual Arrow dependencies once
per invocation and includes effective dataset features and format. Saved state
is used to locate shards even for in-memory loading. Dataset descriptions,
citations, and JSON formatting do not affect matching. External source-image
paths are hashed as raw files without decoding images; unused external mask
contents do not enter the digest. It does not scan artifacts, cache, or
output folders. The first dataset download is reloaded from its saved copy so
first and subsequent runs identify the same files. Any dataset-file change can
conservatively invalidate every sample, including a change to target prompts
or masks. The current editor does not use masks.

`ModelRuntime` resolves a Hub revision to one snapshot before loading, records
its commit as provenance, and streams the selected UNet, VAE, and text-encoder
weight files. The resolved commit is not independently matched when content is
unchanged; configured model/dataset references remain source identity constraints.
Tokenizer assets and effective model/image/text processing settings are included:
resize/interpolation, normalization, RGB conversion, token limits and special
tokens, and VAE scaling. Content hashes are shared across all parameter files
and methods; they add one sequential read of the dependency files per run.
Built-in inversion requires eta zero and uses the VAE posterior mode, so its
seed does not affect inversion or the generated edit. Random external adapters
still declare their seeds through `inversion_cache_parameters`. Target prompts
and P2P policies are applied afresh during editing.
Cache hits are resolved individually, and only misses enter model calls.
Consequently, an actual inversion batch can be smaller than the configured size.
For fresh artifacts, `inversion_batch_seconds` measures the batch including
validation and staged batch storage; final group assembly is separate.
`inversion_batch_size` counts its uncached images, and
`inversion_seconds` is that duration divided by the count. Batch duration is
repeated on each member's record, so it should not be summed across members.
Cache hits report zero for all three values; replay-only records use `null`.

## Intermediate components

Workflow inversion also reuses intermediate components across methods, parameter
files, and runs. All bundled methods share source VAE latents and unconditional
and conditional caption embeddings. Direct and Null-text additionally share the
complete conditional DDIM pivot trajectory, including the terminal pivot. Their
residuals and optimized embeddings remain method-specific; DDIM and ReNoise
compute their own terminal latents.

Intermediate entries live separately from replay artifacts:

```text
data/cache/inversion/
|-- sources/<digest>.safetensors
`-- conditional_pivots/<digest>.safetensors
```

Source identity version 2 includes the dataset reference, fingerprint and content
digest, UID, RGB image bytes and dimensions, source prompt, model content and
processing settings, dtype, and VAE tiling. Source caching projects model
content to VAE, text encoder, and tokenizer only, so changing UNet weights alone
does not invalidate source encoding. Source-code hashes,
hardware identity, PyTorch builds, CPU thread counts, and dependency
versions are excluded. Pivot identity adds UNet content, scheduler settings,
and timesteps to
the semantic source identity; exact input-tensor bytes are not part of its key.
Payload checksums still detect damaged cache files. Source
components do not depend on sampling steps or guidance; conditional pivots depend
on the schedule but not guidance or downstream method hyperparameters.

Intermediate identity deliberately excludes batch size and composition. Cached
images can be assembled into a different batch, and only missing components enter
encoding or pivot model calls. This can introduce small numerical differences
compared with recomputing the entire batch. Producer UIDs and actual batch size
are saved as metadata. Final-artifact cache matching also excludes the configured
inversion batch size.

Each intermediate entry validates metadata, tensor names, shapes, dtype,
finiteness, and payload checksums. Invalid or unreadable entries are recomputed;
complete entries are published atomically and remain reusable after an interrupted
run. Memory remains bounded by the current batch. Cache write failures are
reported in batch diagnostics and do not discard successfully computed tensors.
There is no automatic eviction; deleting `data/cache/inversion/` only removes
intermediate reuse. Publication assumes one active writer.

`inversion.components` owns source preparation and conditional pivot reuse.
Method helpers own guidance/scheduler checks and artifact construction. `InversionContext.components` is optional and defaults to `None`.
Workflows supply a `SharedInversionComponents(IntermediateCache())` provider;
direct method calls without a provider compute locally. Cache reads and shared
computations produce detached ordinary tensors suitable for Null-text autograd.
Old schema-v4 artifacts require regeneration; new schema-v5 entries preserve
production identity across group merges. Code changes do not automatically
invalidate caches; remove affected cache entries when recomputation is needed.

## Results and timing

`edit_batch_seconds` retains the preparation-plus-editor wall time, excluding input
reads and image writes. Both legacy batch durations repeat on member records;
use `batches.jsonl` for aggregate timing instead of summing those repeated fields.
`results.jsonl` additionally records `inversion_batch_id` and `edit_batch_id`.
Cache hits retain their lookup batch ID, replay-only inversions use `null`, and
skipped samples have no edit batch ID (`null`).

Each attempted inversion batch, editing method segment, and group publication
writes exactly one schema-v1 record to `batches.jsonl`, including failed attempts.
Common fields are `batch_id`, `phase`, `sample_count`, `status`, and
`durations_seconds`, with the method ID and parameter filename identifying its
output. Errors include an `error` message. Stages not reached have zero duration.

| Phase | Duration keys | Counts |
| --- | --- | --- |
| `inversion` | `input_read`, `cache_lookup`, `method`, `validation`, `staging` | `sample_count` is the requested chunk size; `cache_hits` and `uncached_count` describe resolved lookups |
| `edit` | `input_read`, `preparation`, `editor`, `image_write` | `sample_count` is the requested method segment; `edited_count` is the number prepared for editing |
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
boundaries; CPU stages use wall-clock timing, with no per-timestep synchronization.
Model loading and preflight are outside these batch durations.

Inversion batch diagnostics also include `source_cache_hits`, `source_cache_misses`,
`pivot_cache_hits`, and `pivot_cache_misses`, counted per image, plus
`intermediate_cache_write_errors` containing component, UID, and error details.
Intermediate lookup, computation, and storage are included in the `method`
duration. Final-artifact hits bypass intermediate work and leave these counts at
zero. Reusing intermediates alone does not set `artifact_reused` to `true`.

## Replay

A replay sweep through `edit_artifacts` records incompatible artifact/file pairs as `skipped`
with a reason and continues with compatible pairs. Malformed artifacts and
operational failures stop execution. If every pair is skipped, the API
raises an error. Explicit replay continues to support existing
uncataloged artifacts and does not read method hyperparameter files.

Use `InversionArtifact` for each image's inversion result and
`ArtifactCatalog.store_group` to publish a collection. Use the same model revision,
scheduler configuration, and timestep sequence as the editor. Loading restores
the saved production fields and compares them against `ArtifactSettings`; it
does not fill in provenance from current settings. Compatibility mismatches
remain `ArtifactCompatibilityError`. The runner also validates tensor shapes
and method-specific replay requirements. Group merges retain every entry's
original production record, even when entries came from different runs.

`ArtifactCollection(path)` exposes ordered `ids`, `references`, and
`reference(sample_uid)`. Each `ArtifactReference` contains `path`, `index`, and
`sample_uid`. `collection.load(reference, settings=...)` loads only the selected
entry and verifies its UID at that index. Supply `ArtifactSettings` with the
shared model, dataset, scheduler, eta, and `timesteps_for_steps` callback (such
as `editor.timesteps_for_steps`). The optional `model_content`, `dataset_content`,
and `numerics` expectations validate `ArtifactProvenance`. Workflows provide
these settings automatically. The coordinator fills missing provenance on
new adapter results, so existing adapters do not need to change their return
constructors. Standalone artifact producers should supply provenance themselves
when their files will be loaded by a workflow.
Legacy schema-v1/v2/v3/v4 artifacts are no longer readable; regenerate with `diffuse`.
`InversionArtifact.validate_terminal_latent()` checks shape, floating dtype,
positive spatial dimensions, and finite values. Construction, staged serialization,
and editing validation share this check without loading an entire collection's
tensors. Invalid cached latents cause recomputation; explicit replay rejects
non-finite latents as malformed artifacts. Editing restores installed attention
processors and performs model-offload cleanup on success or failure, including
encoding and decoding failures. Cleanup failures do not replace an existing error.

For the project Hub dataset, an external inversion implementation can package
its computed `z_t` tensor as follows:

```python
from image_editing_inversion.artifacts import ArtifactCatalog, ArtifactProvenance, InversionArtifact
from image_editing_inversion.config import load_config
from image_editing_inversion.data import DatasetRepository
from image_editing_inversion.generation import Editor

repository = DatasetRepository()
records = repository.dataset
config = load_config("default.yaml")
editor = Editor(config)
sample = records[0]
# z_t is a [1, 4, 64, 64] tensor produced by an inversion implementation.
artifact = InversionArtifact(
    method_id="external-ddim",
    model_id=config.model.model_id,
    model_revision=config.model.revision,
    dataset_ref="beatle-ju1ce/image-editing-inversion",
    dataset_fingerprint=records._fingerprint,
    sample_uid=sample["uid"],
    scheduler_id="ddim",
    scheduler_config=editor.scheduler_config,
    eta=config.sampling.eta,
    timesteps=editor.expected_timesteps,
    terminal_latent=z_t,
    provenance=ArtifactProvenance(
        model_content=editor.model_content_identity,
        model_commit=editor.model_commit,
        dataset_content=repository.content_digest,
        numerics=editor.inversion_cache_settings(),
        guidance_scale=config.sampling.guidance_scale,
    ),
)
reference, = ArtifactCatalog().store_group(
    [artifact],
    pipeline_h_params={
        "num_inference_steps": config.sampling.num_inference_steps,
        "guidance_scale": config.sampling.guidance_scale,
    },
)
print(reference.sample_uid)  # Pass this raw UID to edit_artifacts; cache inputs are optional.
```

Before replay, install an `InversionMethod` adapter registered as
`external-ddim`, even if it uses the default P2P behavior and no denoising hook.
New collections use schema v5 with the aligned IDs embedded in safetensors.

```python
from image_editing_inversion.workflows import edit_artifacts

run_dir = edit_artifacts("PASTE_SAMPLE_ID_HERE")
print(run_dir)
```

The optional `artifact_id` argument accepts a raw dataset sample UID. Every matching
position across method/settings groups is replayed. Call `edit_artifacts()` without
an ID to replay every published schema-v5 collection, including uncached entries,
in relative group-path order followed by positional order. Hidden staging files,
legacy layouts, and symlinks are excluded. Malformed collections, missing IDs,
and an empty artifact repository produce clear errors. Replay batches are flushed
when the method changes.

Both workflows use the private project Hub dataset loader described above.
New artifacts use that dataset's shared identity and resolve their sample UID
against it.
`edit_artifacts` accepts no configuration or directory overrides and processes every pipeline
parameter file. Results are written under `data/output/<run-id>/` with a
schema-v2 `sweep.json` manifest and `<parameter-file>/<method>/` children containing
`results.jsonl`, `batches.jsonl`, `resolved-config.json`, and images per sample under `images/`.
Run
records contain UIDs, combined artifact paths, and artifact indices rather than
prompts, source images, or masks.
