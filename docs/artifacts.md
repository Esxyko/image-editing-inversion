# Artifact storage, reuse, and replay

Inversion artifacts preserve per-image latent state and provenance independently
of generated RGB outputs. This guide defines their format, publication rules,
cache identity, and replay contract. For result files and archives, see
[run outputs](run-outputs.md).

- [Collection format](#collection-format)
- [Publication and interruption](#publication-and-interruption)
- [Cache reuse](#cache-reuse)
- [Intermediate components](#intermediate-components)
- [Replay](#replay)

## Collection format

Generated inversions are stored independently in one combined file per method
and step/guidance group:

```text
data/artifacts/
|-- catalog.json
`-- <method>/steps-30_guidance-7.5/
    |-- h-params.json
    `-- artifacts.safetensors
```

Each image has a floating `artifacts/<i>/terminal_latent` tensor of shape
`[1, 4, height/8, width/8]` with positive dimensions and finite values. Optional
`artifacts/<i>/state/<name>` tensors have one entry per timestep and are interpreted
by the registered method's hook. Images, captions, and masks stay in the dataset.

Schema v2 additionally supports `artifacts/<i>/static/<name>` for nonempty finite
method-defined tensors that do not vary by timestep. All bundled methods use these for the
original source conditional and unconditional `[1, tokens, hidden]` embeddings.
They are stored once per image and reused without modifying latent updates.
DDIM has no per-step state; Null-text retains optimized unconditional embeddings;
ReNoise retains its guidance state; Direct retains its original per-step residuals.
The experimental `source_latents` / `direct_source_latents` overwrite variants
have been withdrawn. Built-in replay rejects these tensors and requires regeneration.

Each schema-v1/v2 file embeds JSON-encoded `ids` and aligned `entries` lists in its
safetensors metadata. `entries[i]` indexes a production record in the JSON-encoded
`contexts` list for `ids[i]`. Common production records are stored once, while
every UID retains its own source association.
New files also embed a JSON `checksums` field containing a SHA-256 digest of the
production metadata and a digest of every tensor's dtype, shape, and exact bytes.
Readers accept this additive field in both collection schemas. Loaded entries
are verified before replay; mismatches are malformed artifacts, even when the
damaged values remain finite. Checksums are carried through streaming assembly,
and the assembled file is verified before replacing a published group.
The raw dataset UID at `ids[i]` identifies tensors under `artifacts/<i>/`.
UIDs such as `image:1` are stored verbatim. The method comes from the method
directory and steps/guidance from the group name. Guidance values are written
without rounding, with whole values such as `7.0` written as `7`.
`h-params.json` records `num_inference_steps` and `guidance_scale`; the latter
is inversion CFG. Shared editing CFG is recorded separately as provenance's
`editing_guidance_scale` and in the resolved configuration's `sampling` section.
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
inversion guidance, pipeline editing guidance, complete method hyperparameters,
and numerical runtime settings. Cache lookup and publication require the saved
editing guidance to match the cache inputs.
Missing or incompatible cached entries are recomputed. An existing group whose
integrity cannot be verified blocks publication and remains untouched, preventing
limited or source-filtered runs from discarding unselected records. Malformed
catalogs fail clearly.

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
new files remain independently verifiable through their embedded checksums.
The next inversion invocation verifies the entire published file before merging,
preserves unselected entries, and publishes the actual current file checksum to
the catalog. Stale per-ID cache entries still undergo provenance checks and miss
when their production inputs do not match.

Older files without embedded checksums require a matching catalog-v2 group checksum.
They are verified in full before replay or merging, and merged output acquires
embedded checksums. A missing or mismatching legacy checksum cannot distinguish
catalog lag from file damage: replay skips files with no integrity baseline and
rejects checksum mismatches; inversion preserves the original file and stops.
Restore a verified copy and its catalog, or regenerate separately after preserving
the original. The workflow never creates a trusted checksum from unverified data.

Current schema-v1/v2 group files with complete matching method snapshots, matching
separate editing CFG, and catalog-v2 entries with cache identity version 4 are reusable.
Previous artifact formats (including schema v5), missing snapshots or separate editing CFG,
and older cache identities require regeneration. Legacy provenance remains readable;
editing CFG is not automatically migrated from older method snapshots.
Missing provenance fields and older identities are cache misses within verifiable collections;
unsupported or unverifiable existing group files are preserved and block replacement. Legacy
per-sample directories, hash-named groups, and catalog-v1 entries are ignored
without deletion. The old catalog is replaced only after successful new-format
publication. Both workflows write results to the fixed `data/output/` root.

Collections with static tensors write schema v2; collections without them retain
schema v1. Streaming group assembly can retain older v1 entries alongside v2
entries. All bundled methods now use `source_embedding_version: 1`, which
invalidates the withdrawn overwrite cache identities and records original source
encodings. Native legacy artifacts without static encodings remain supported by
explicit replay using the original text-encoding path when they have complete
matching method snapshots, separate editing CFG, and a verifiable integrity baseline.
Batches mixing artifacts
with and without captured encodings require regeneration. Fresh embedding state
does not depend on intermediate caches during replay.

## Cache reuse

Changing only attention fractions or editing policies reuses inversion. The
bundled deterministic methods also reuse inversion when the editing seed
changes. Changing steps or inversion CFG creates a new group; changing any other
method setting or pipeline editing CFG replaces the sample's artifact in its group. Method
hyperparameters are loaded for cache matching using the
same validated settings as inversion. Each run generates its RGB pixel tensors again.
The additional algorithm identity records effective Null-text stopping thresholds
and ReNoise prediction intervals, but complete snapshot matching also includes
unused increments, averaging windows, and low-timestep caps. Hyperparameter files
undergo full validation. Inactive
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

### Dataset and model identity

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
### Cache-hit timing

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
and conditional caption embeddings. Direct and Null-text additionally cache the
complete DDIM pivot trajectory, including the terminal pivot. Direct uses CFG 1;
Null-text uses inversion CFG. They can share pivots when the pivot CFG matches.
Direct artifacts bind these source encodings and reference latents for replay;
Null-text's optimized embeddings remain method-specific. DDIM and ReNoise compute
their own terminal latents.

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
timesteps, pivot CFG, and checksums of the actual source latent and conditional embedding
to the semantic source identity. Pivots cannot be reused with newly encoded source
tensors that differ numerically from their original inputs. Guided pivots also
include a checksum of the actual unconditional embedding.
Payload checksums still detect damaged cache files. Source
components do not depend on sampling steps or guidance; conditional pivots depend
on the schedule and pivot CFG, but not downstream optimization settings.

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
Previous artifact formats require regeneration; current schema-v1/v2 entries preserve
production identity across group merges. Code changes do not automatically
invalidate caches; remove affected cache entries when recomputation is needed.

## RGB tensor outputs

Decoded reconstructions and edits are saved as uint8 RGB batches under each run.
See [RGB tensor outputs](run-outputs.md#rgb-tensor-outputs) for the schema and loading example.

## Run archives

`compress_run` verifies and publishes a ZIP before deleting the source run.
See [run archives](run-outputs.md#run-archives) for the storage and failure contract.

## Temporary Null-text diagnostics

`experiment-diagnostic` records optimization steps, reconstruction comparisons,
and reproducibility metadata. See [Null-text diagnostics](null-text-diagnostics.md)
for commands and field definitions.

## Results and timing

Sample results reference unique inversion and editing batch IDs. Use the batch
records for aggregate durations; see [results and timing](run-outputs.md#results-and-timing).

## Replay

A replay sweep through `edit_artifacts` records incompatible artifact/file pairs as `skipped`
with a reason and continues with compatible pairs. Malformed artifacts and
operational failures stop execution. If every pair is skipped, the API
raises an error. Explicit replay continues to support existing
uncataloged artifacts with complete matching method snapshots and embedded checksums.
Legacy files need a matching catalog-v2 checksum. It reads current
method YAML files for strict parameter matching, without rerunning inversion.
Missing snapshots and differing parameters are compatibility mismatches: cache
lookup recomputes them, while explicit replay skips them with a reason.

Use `InversionArtifact` for each image's inversion result and
`ArtifactCatalog.store_group` to publish a collection. Use the same model revision,
scheduler configuration, and timestep sequence as the editor. Loading restores
the saved production fields and compares them against `ArtifactSettings`; it
does not fill in provenance from current settings. Compatibility mismatches
remain `ArtifactCompatibilityError`. The runner also validates tensor shapes
and method-specific replay requirements. Group merges retain every entry's
original production record, even when entries came from different runs.

### Reading collections

`ArtifactCollection(path)` exposes ordered `ids`, `references`, and
`reference(sample_uid)`. Each `ArtifactReference` contains `path`, `index`, and
`sample_uid`. `collection.load(reference, settings=...)` loads the selected
entry and verifies its UID and tensor checksums. Legacy files without embedded
checksums undergo one full integrity scan when first read; tensors are inspected
one at a time. `collection.validate_integrity()` verifies an entire merge base.
`with collection.reader() as reader:`
allows repeated `reader.load(reference, settings=...)` calls with one native
header parse. Cache lookup batches use `catalog.lookup_batch(settings=...)` and
its `lookup(inputs)` method. Readers clone selected tensors into independent CPU
storage and close at the batch boundary before Windows file replacement. Supply `ArtifactSettings` with the
shared model, dataset, scheduler, eta, and `timesteps_for_steps` callback (such
as `editor.timesteps_for_steps`). The optional `model_content`, `dataset_content`,
and `numerics` expectations validate `ArtifactProvenance`. Workflows provide
these settings automatically. The coordinator fills missing provenance on
new adapter results, so existing adapters do not need to change their return
constructors. Standalone artifact producers should supply provenance themselves
when their files will be loaded by a workflow.
The current indexed collection format supports schema v1 and schema v2, the
latter adding static tensors while retaining indexed production metadata. Older
pre-indexed layouts (including the historical v1/v2/v3/v4/v5 formats) require
regeneration with `diffuse`; their metadata/layout differs from current v1/v2.
`InversionArtifact.validate_terminal_latent()` checks shape, floating dtype,
positive spatial dimensions, and finite values. Construction, staged serialization,
and editing validation share this check without loading an entire collection's
tensors. Invalid cached latents cause recomputation; explicit replay rejects
non-finite latents as malformed artifacts. Editing restores installed attention
processors and performs model-offload cleanup on success or failure, including
encoding and decoding failures. Cleanup failures do not replace an existing error.

### Publishing external artifacts

For the project Hub dataset, an external inversion implementation can package
its computed `z_t` tensor as follows:

```python
from dataclasses import replace
from image_editing_inversion.artifacts import ArtifactCatalog, ArtifactProvenance, InversionArtifact
from image_editing_inversion.config import load_config
from image_editing_inversion.data import DatasetRepository
from image_editing_inversion.generation import Editor
from image_editing_inversion.inversion import get_method

repository = DatasetRepository()
records = repository.dataset
config = load_config("default.yaml")
method = get_method("external-ddim")  # Register the adapter before this example.
config = replace(config, method_h_params=dict(method.method_h_params(config)))
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
        guidance_scale=config.inversion_guidance_scale,
        editing_guidance_scale=config.editing_guidance_scale,
        method_h_params=config.require_method_h_params(),
    ),
)
reference, = ArtifactCatalog().store_group(
    [artifact],
    pipeline_h_params={
        "num_inference_steps": config.sampling.num_inference_steps,
        "guidance_scale": config.inversion_guidance_scale,
    },
)
print(reference.sample_uid)  # Pass this raw UID to edit_artifacts; cache inputs are optional.
```

Before replay, install an `InversionMethod` adapter registered as
`external-ddim`, even if it uses the default P2P behavior and no denoising hook.
Collections use schema v1, or v2 when static state is present, with aligned IDs
embedded in safetensors.

```python
from image_editing_inversion.workflows import edit_artifacts

run_dir = edit_artifacts("PASTE_SAMPLE_ID_HERE")
print(run_dir)
```

### Replay selection and outputs

The optional `artifact_id` argument accepts a raw dataset sample UID. Every matching
position across method/settings groups is replayed. Call `edit_artifacts()` without
an ID to replay every supported published collection, including uncached entries,
in relative group-path order followed by positional order. Hidden staging files,
legacy layouts, and symlinks are excluded. Malformed collections, missing IDs,
and an empty artifact repository produce clear errors. Replay batches are flushed
when the method changes.

Both workflows use the private project Hub dataset loader described above.
New artifacts use that dataset's shared identity and resolve their sample UID
against it.

For fresh inversions, `diffuse --dataset pie-bench` or `--dataset magic-brush`
limits inversion, reconstruction, and editing to that source. Python
`run_methods` and `WorkflowRunner.run_methods` accept the same optional `dataset`
keyword. Filtering retains the combined dataset's fingerprint and content digest,
so selected samples can reuse artifacts from full runs. Publication merges the
selected artifacts into the existing group and preserves other sources' entries.

`edit_artifacts` accepts no configuration or directory overrides and processes every pipeline
parameter file. Results are written under `data/output/<run-id>/` with a
schema-v2 `sweep.json` manifest and `<parameter-file>/<method>/` children containing
`results.jsonl`, `batches.jsonl`, `resolved-config.json`, and RGB batch files under `tensors/`.

### Configuration snapshots and manifests

For inversion and replay runs, `resolved-config.json` contains `method_h_params`:
the complete current method settings, including all defaults and inversion CFG.
The bound `config.method_h_params` records the same normalized snapshot. Shared
editing CFG is recorded under `sampling.editing_guidance_scale`. Null-text and
ReNoise also record all of their optimization or refinement settings. Per-image
provenance contains an optional `method_h_params` snapshot and separate optional
`editing_guidance_scale`. Legacy metadata without either field remains readable,
but cannot pass cache reuse or replay compatibility checks. Readers accept these
additive provenance fields without changing collection tensor schemas.

Final inversion cache identity version `4` includes the complete method snapshot and
separate `editing_guidance_scale` alongside algorithm `method_parameters` and
production inputs. Any parameter change, including editing CFG or a currently
unused optimization setting, prevents
reuse. Direct, Null-text and ReNoise per-step guidance and provenance's historical
`guidance_scale` field denote inversion CFG. Published directory names still use
`steps-N_guidance-G` with inversion CFG; changing other parameters republishes
selected UIDs in that group and preserves entries outside the selected prefix.

The manifest includes a top-level `dataset` field from its initial creation:
`"pie-bench"` or `"magic-brush"` for a selected source, and `null` when the
argument is omitted or the workflow replays saved artifacts. The field is
preserved in successful and failed runs; `schema_version` remains `2`.

Limited inversion runs also include a top-level `nsamples` field from initial
manifest creation, containing the requested positive count. `experiment` always
provides this limit; Python callers may pass `nsamples` to `run_methods` or
`WorkflowRunner.run_methods`. Filtering by source precedes limiting in dataset
order. If fewer records are available, all matching records are processed.
Only the selected prefix is inverted and edited; cached records count toward the
limit and publication preserves artifacts outside it. Unlimited and replay runs
omit `nsamples`. Sample selection does not change cache identity or artifact schemas.

Run records contain UIDs, combined artifact paths, and artifact indices rather than
prompts, source images, or masks.
