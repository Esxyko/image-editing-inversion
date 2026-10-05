# Image Editing Inversion

## Load the project dataset

The project dataset is the private Hugging Face repository
`beatle-ju1ce/image-editing-inversion`. Request access to it, then set `HF_TOKEN`
in your environment or add it to the project's ignored `.env` file. The
`.env.example` file shows the key to add. An existing Hugging Face CLI login
also works.

```python
from image_editing_inversion.data import load_project_dataset

records = load_project_dataset()
example = records[0]
print(example["uid"])
```

The first call downloads the repository's `train` split and saves its Hugging
Face Datasets files directly in `data/`. Later calls load that saved copy without
network access. The `data/` directory is local and ignored by Git. A full first
download needs several gigabytes of free disk space for the download cache and
saved dataset.

## Dataset contents

Each record represents an edit with a source image, a binary edit mask, and a
pair of captions describing the source and intended result. The edited target
image was used to prepare MagicBrush captions but is not stored in the Hub
dataset.

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

## Package modules

The package has six subpackages with public APIs exported from their
`__init__.py` files:

| Subpackage | Responsibility |
| --- | --- |
| `config` | Experiment settings, hardware validation, and YAML loading |
| `data` | Dataset loading, identity validation, and UID-based sample access |
| `artifacts` | Portable inversion artifact validation and serialization |
| `inversion` | Adapter interfaces, context, hooks, discovery, and algorithms in `inversion.methods` |
| `generation` | Model runtime, token alignment, Prompt-to-Prompt, and denoising |
| `workflows` | Workflow orchestration, batching, and run output |

See [`structure.md`](structure.md) for implementation responsibilities and
dependency boundaries. Import public classes and functions from these
subpackages rather than their implementation files.

The former `image_editing_inversion.dataset`, `image_editing_inversion.artifact`,
and `image_editing_inversion.runner` modules have been removed. Use `data` for
dataset loading, `artifacts` for `InversionArtifact`, `inversion` for adapter and
hook APIs, and `workflows` for `WorkflowRunner`, `edit_artifacts`, and
`run_methods`. Configuration and generation APIs are exported from `config`
and `generation`. External adapters must update moved imports; there are no
compatibility wrappers. Saved artifacts use only the schema-v5 combined format described below;
older formats must be regenerated. Sweep
outputs are grouped by pipeline parameter filename and method as described below.

## Shared editing framework

The framework bundles baseline DDIM, Null-text Inversion, Direct Inversion, and ReNoise
methods and accepts inversion artifacts produced outside the package. It uses
Stable Diffusion 1.5, DDIM sampling, and a shared Prompt-to-Prompt editor to
produce a reconstruction
with `source_prompt` and an edit with `target_prompt`. The dataset mask is reserved
for later metrics; it is not used to constrain the edit.

Install the project with `uv sync`, then edit [`config.yaml`](config.yaml) and
the YAML files in [`pipeline_h_params/`](pipeline_h_params/). Run commands from
the project root. Shared configuration is always loaded from `config.yaml` in
the working directory. Its four required sections supply shared settings;
an optional `methods` section supplies adapter settings:

- `model` selects the SD 1.5 checkpoint, revision, and image dimensions.
- `sampling` selects DDIM eta and the editing seed.
- `prompt_to_prompt` selects the edit mode. `auto` uses replacement when source and target captions
  have the same number of whitespace-separated words, and refinement otherwise.
- Optional `methods.<method_id>.prompt_to_prompt` supplies additional settings
  validated by that method's P2P subclass; it does not override the shared mode
  or replacement fractions. The settings are saved with the run.
- `runtime` selects the device, precision, separate inversion and editing batch sizes,
  attention query chunk size, CPU offload mode, VAE slicing/tiling, data-loading
  workers, and pinned memory.

The checked-in runtime values target CUDA with float16, `inversion_batch_size: 1`,
and `batch_size: 1`. `runtime.inversion_batch_size` controls source images inverted
together by all four bundled methods. It is optional and defaults to `1`;
`runtime.batch_size` independently controls reconstruction/editing pairs.
For example, the two stages can use different batch sizes:

```yaml
runtime:
  # Keep the other runtime settings from config.yaml.
  inversion_batch_size: 2
  batch_size: 1
```

Increase either batch size only when GPU memory permits; a batch of `N` edits has
`N` source and `N` target branches inside the shared editor. For a smaller GPU,
keep the batch size at one and consider `cpu_offload: model` or
`cpu_offload: sequential` and a smaller `attention_query_chunk_size`. An
unsupported device or option combination fails before model loading. A run saves its resolved
configuration in each parameter file's `resolved-config.json`.
Inversion batches use the same configured device, dtype, VAE options, and CPU
offload mode as editing. Unindexed `cuda` uses PyTorch's current CUDA device;
`cuda:<index>` selects a specific GPU. `num_workers` controls dataset/artifact
read threads. `pin_memory` pins CPU model inputs and pivot tensors before
nonblocking transfers to CUDA. `attention_query_chunk_size` applies only to
Prompt-to-Prompt editing. Larger inversion batches can introduce small
floating-point differences from singleton execution. An out-of-memory error
stops the run; batch sizes are never reduced automatically.

### Pipeline parameter files and artifact reuse

Each file in `pipeline_h_params/` must define these four settings, using the
same grouping as [`pipeline_h_params/default.yaml`](pipeline_h_params/default.yaml):

```yaml
sampling:
  num_inference_steps: 50
  guidance_scale: 7.5
prompt_to_prompt:
  cross_replace_fraction: 0.4
  self_replace_fraction: 0.6
```

Steps must be a positive integer, guidance must be finite and nonnegative,
and both fractions must be between 0 and 1. All four values are required;
unknown keys and definitions of these settings in `config.yaml` are rejected.
The resolved Python settings still use `config.sampling` and
`config.prompt_to_prompt`. Use `load_config("default.yaml")` to resolve one file.

`diffuse` inverts, reconstructs, and edits every record in the project dataset.
For a comparison, hold `config.yaml`, `method_h_params/`, and the dataset fixed
across runs. Only the files in `pipeline_h_params/` vary.
It requires at least one `--method` and `--h-params FILENAME` to select one YAML
file, or `--h-params ALL` to process every parameter file:

```powershell
uv run diffuse `
  --method ddim `
  --h-params default.yaml
```

Use `--method ALL` to run every discovered method, including installed adapters
and methods registered through the Python API, in sorted method-ID order:

```powershell
uv run diffuse --method ALL --h-params ALL
```

Repeat `--method` to select individual methods or combine them with `ALL`.
Selectors expand in command-line order, and each method runs once at its first
appearance. Only exact uppercase `ALL` is reserved; lowercase `all` is an ordinary
method ID. The public `run_methods(["ALL"], pipeline_h_params_file="default.yaml")`
API follows the same rules. Empty selections, unknown IDs, and method IDs that
are not portable directory names fail before workflow construction.

Use the exact value `--h-params ALL` to process every top-level `.yaml` or `.yml`
file in filename order, completing all requested methods over the entire dataset for one file
before the next. Missing, empty, or malformed parameter files fail before model loading.
Before processing any samples, `diffuse` preflights every selected method against
every selected parameter file. The first phase checks inversion configuration,
method hyperparameter files, editing batch limits, and P2P settings without loading
model weights. After it succeeds, the second phase reuses one model to check each
file's actual scheduler and timestep requirements. For example, a later file with
guidance <= 1 rejects a sweep containing Null-text before any earlier file is inverted.
Failures are collected with parameter filenames and method IDs; the sweep and
invalid combinations are marked `error`, with zero sample counts and no per-sample
results. Configuration errors do not skip methods in `diffuse`.
The `edit_artifacts` Python API replays saved artifacts and always sweeps all parameter files.
It does not call inversion preflight or read method hyperparameter files; valid
artifacts that are incompatible with a parameter file are still recorded as `skipped`.
The dataset and loaded model are shared across the sweep; scheduler timesteps
and resolved settings are updated for each file. An invocation returns one
parent run directory:

```text
data/output/<run-id>/
|-- sweep.json
`-- default.yaml/
    `-- <method>/
        |-- resolved-config.json
        |-- results.jsonl
        |-- batches.jsonl
        `-- images/<uid-with-colons-replaced-by-underscores>/
            |-- reconstructed.png
            `-- edited.png
```

Both workflows use this layout, including single-method runs. Each method has
its own resolved snapshot and records; snapshots include
`method_id`, and recorded image paths are relative to that method's directory.
Replay reuses the same output directory across artifact groups for a method.

Image directories use the sample UID, replacing `:` with `_` for Windows
compatibility; for example, `435242:2` becomes `images/435242_2/`. The current
dataset's 5,296 UIDs contain only digits and colons, contain no underscores,
and remain unique after this replacement. Result records and inversion artifacts
retain the original UID. An existing image directory causes an error rather than
overwriting its images. Existing run directories keep their original names.

`sweep.json` uses schema version 2. Each `parameter_files` entry retains its
status and aggregate `ok`, `skipped`, and `error` counts, and contains a `methods`
list. Each method entry records `method_id`, its directory relative to the run
root, status, counts, and an error message when applicable. All combinations
start pending. A failure marks the affected method, parameter file, and sweep
as errored and stops execution; unstarted combinations remain pending.
Existing run folders and inversion artifacts are retained unchanged.

Generated inversions are stored independently in one combined file per method
and step/guidance group:

```text
data/artifacts/
|-- catalog.json
`-- <method>/steps-50_guidance-7.5/
    |-- h-params.json
    `-- artifacts.safetensors
```

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

`inversion.methods.common` owns shared source preparation, conditional pivots,
finite-value and guidance checks, strict DDIM scheduler checks, and artifact
construction. `InversionContext.components` is optional and defaults to `None`.
Workflows supply a `SharedInversionComponents(IntermediateCache())` provider;
direct method calls without a provider compute locally. Cache reads and shared
computations produce detached ordinary tensors suitable for Null-text autograd.
Old schema-v4 artifacts require regeneration; new schema-v5 entries preserve
production identity across group merges. Code changes do not automatically
invalidate caches; remove affected cache entries when recomputation is needed.

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

Inversion batch diagnostics also include `source_cache_hits`, `source_cache_misses`,
`pivot_cache_hits`, and `pivot_cache_misses`, counted per image, plus
`intermediate_cache_write_errors` containing component, UID, and error details.
Intermediate lookup, computation, and storage are included in the `method`
duration. Final-artifact hits bypass intermediate work and leave these counts at
zero. Reusing intermediates alone does not set `artifact_reused` to `true`.

A replay sweep through `edit_artifacts` records incompatible artifact/file pairs as `skipped`
with a reason and continues with compatible pairs. Malformed artifacts and
operational failures stop execution. If every pair is skipped, the API
raises an error. Explicit replay continues to support existing
uncataloged artifacts and does not read method hyperparameter files.

### Run baseline DDIM inversion

The bundled `ddim` method converts the source image to RGB, resizes it through
the pipeline image processor to the configured dimensions, and uses the VAE
posterior mode to encode it deterministically. It then follows the ascending
DDIM noise schedule using `source_prompt`, an empty negative caption, and the
shared `sampling.guidance_scale`. The inverse schedule must exactly reverse the
editor's denoising timesteps. This baseline requires `sampling.eta: 0.0` and
rejects dynamic thresholding or incompatible inverse schedules.

```powershell
uv run diffuse --method ddim --h-params ALL
```

The runner saves or reuses the terminal latent and provenance through the
artifact catalog, then produces `reconstructed.png` and `edited.png` with the shared
Prompt-to-Prompt policy. DDIM artifacts have no per-step state and need no
denoising hook. Inversion uses `runtime.inversion_batch_size`; reconstruction and
editing use `runtime.batch_size`. Saved DDIM artifacts can be replayed with
`edit_artifacts` using a compatible configuration. DDIM inversion is approximate,
so reconstruction need not reproduce the source image exactly.

The implementation is also available as `DDIMInversion` from
`image_editing_inversion.inversion`. Adapters can share prompt encoding through
`context.editor.encode_prompts`, which validates the model's CLIP token limit.
Diffusers is imported during inversion or scheduler preflight; discovering the method does not
load a model. After updating an existing installation, run `uv sync` to refresh
the bundled method's Python entry-point metadata.

### Run Null-text Inversion

The bundled `null-text` method implements
[Null-text Inversion](https://null-text-inversion.github.io/). It deterministically
encodes the image with the VAE and builds a source-caption DDIM pivot trajectory
at guidance 1. It then optimizes one unconditional embedding per denoising step
using the shared `sampling.guidance_scale` and latent reconstruction MSE. Only
these embeddings are optimized; model weights and conditional embeddings stay
fixed. Each step starts from the previous step's optimized embedding.

Run from the project root so the adapter can find
[`method_h_params/Null_text.yaml`](method_h_params/Null_text.yaml). This file is
resolved relative to the working directory and loaded on the adapter's first
inversion preflight, direct inversion, or cache lookup; its validated settings are cached for that adapter instance. Missing
or malformed files fail clearly. Omitted keys use the defaults below, and unknown
keys are rejected. Optimization settings do not change the experiment YAML.

| Hyperparameter | Default | Meaning |
| --- | --- | --- |
| `num_inner_steps` | `10` | Maximum Adam updates per timestep; positive integer |
| `learning_rate` | `0.01` | Initial learning rate; finite and positive |
| `early_stop_epsilon` | `0.00001` | Initial latent-MSE stopping threshold; finite and nonnegative |
| `epsilon_increment` | `0.00002` | Threshold increase per timestep; finite and nonnegative |

For step index `i` out of `T`, the learning rate is
`learning_rate * (1 - i / (2 * T))` and the stopping threshold is
`early_stop_epsilon + i * epsilon_increment`. The optimized parameter and Adam
moments use float32, while embeddings are cast to the configured model precision
for UNet calls. Latent MSE is computed in float32.

Null-text inversion requires `sampling.eta: 0.0`, guidance greater than 1, and a
DDIM scheduler with epsilon prediction, leading timestep spacing, and disabled
sample clipping and dynamic thresholding. The inverse schedule must reverse the
editor's timesteps. Inversion supports `cpu_offload: none` and `model`;
sequential offload is rejected because optimization needs the UNet to remain
resident through backward passes. Float16, bfloat16, and float32 use the existing
hardware restrictions. Inversion uses `runtime.inversion_batch_size`; each image
keeps its own float32 Adam parameters, optimizer state, and early stopping
decision. Only active images participate in optimization model calls.
Reconstruction and editing use `runtime.batch_size`.

To compare all four bundled methods over the entire dataset with shared settings:

```powershell
uv run diffuse `
  --method ddim `
  --method null-text `
  --method direct-inversion `
  --method renoise `
  --h-params ALL
```

Select only `--method null-text` for a Null-text run. Refresh an existing
installation with `uv sync` to register the new entry point. The adapter is also
exported as `NullTextInversion` from `image_editing_inversion.inversion`.

Null-text artifacts store the terminal pivot plus `null_text_embeddings` of
shape `[T, 1, token_count, embedding_dim]` in descending denoising order and the
optimization `guidance_scale` repeated in a `[T]` tensor. The replay hook supplies
the saved embedding to both source and target unconditional branches at every
step. Replay rejects incompatible guidance scales and malformed embeddings.
Use `edit_artifacts` with the same sampling guidance and compatible model/scheduler
settings; it requires neither the hyperparameter file nor further optimization.
Sequential offload can be used for replay, which runs without gradients.

### Run Direct Inversion

The bundled `direct-inversion` method implements Ju et al.'s
[Direct Inversion, later published as PnP Inversion](https://github.com/cure-lab/PnPInversion).
It deterministically encodes the source image with the VAE posterior mode and
builds a source-caption DDIM pivot trajectory at guidance 1, using the same
inverse scheduler approach as Null-text Inversion. It then follows the descending
schedule with an empty unconditional caption and the configured sampling guidance.
At each step, it records the latent residual
`previous_pivot - predicted_previous_latent` and advances from the corrected latent.
It performs no optimization and needs no method hyperparameter file.

```powershell
uv run diffuse --method direct-inversion --h-params ALL
```

Refresh an existing installation with `uv sync` to register the entry point.
The adapter is also exported as `DirectInversion` from
`image_editing_inversion.inversion`.

Direct Inversion requires `sampling.eta: 0.0`, finite nonnegative sampling
guidance, and a DDIM scheduler with epsilon prediction, leading timestep spacing,
and disabled sample clipping and dynamic thresholding. The inverse timesteps must
exactly reverse the editor's denoising schedule. Inversion and replay run without
gradients and support `cpu_offload: none`, `model`, and `sequential`, subject to
the existing hardware restrictions. Inversion uses `runtime.inversion_batch_size`;
reconstruction and editing use `runtime.batch_size` and the shared Prompt-to-Prompt
attention policy.

Direct Inversion artifacts store the terminal pivot plus
`direct_inversion_offsets` of shape `[T, 1, 4, height/8, width/8]` in descending
denoising order. They also store the sampling `guidance_scale` repeated in a
float64 `[T]` tensor. The replay hook adds each saved offset at full strength to
the source latent after the DDIM update; the target branch receives no direct
latent correction. Shared Prompt-to-Prompt attention transfer still affects
the target branch. Replay rejects malformed or non-finite state and incompatible
guidance, model, dataset, scheduler, or timestep settings. Use `edit_artifacts`
with the same sampling guidance and compatible configuration; it performs no
new inversion. Reconstruction can differ from the original image because of
VAE encoding/decoding and numerical precision.

### Run ReNoise Inversion

The bundled `renoise` method implements the core iterative noising and averaging
from [ReNoise: Real Image Inversion Through Iterative Noising](https://github.com/garibida/ReNoise-Inversion).
It deterministically encodes the source image with the VAE posterior mode and
traverses the shared DDIM timesteps in ascending order. At each timestep, the
previous latent remains fixed while successive UNet predictions refine the noisy
latent estimate. Every prediction uses the current noisy timestep, the source
caption, an empty unconditional caption, and `sampling.guidance_scale`.
The inverse update uses the editor's actual DDIM alpha products, timestep stride,
and final-step boundary. The selected predictions are averaged in float32 before
one final inverse update; UNet inputs use the configured runtime precision.

Run from the project root so the adapter can find
[`method_h_params/ReNoise.yaml`](method_h_params/ReNoise.yaml). The file is loaded
on the adapter's first inversion preflight, direct inversion, or cache lookup; its validated settings are cached for that
instance. Missing or malformed files fail clearly, unknown keys are rejected,
and omitted keys use these defaults:

| Hyperparameter | Default | Meaning |
| --- | --- | --- |
| `num_renoise_steps` | `9` | Refinements after the initial prediction; nonnegative integer |
| `max_num_renoise_steps_first_step` | `5` | Refinement cap for every timestep below 250; nonnegative integer |
| `average_latent_estimations` | `true` | Use the average of predictions in the selected window; boolean |
| `average_first_step_range` | `[0, 5]` | Prediction averaging window for timesteps below 250 |
| `average_step_range` | `[8, 10]` | Prediction averaging window for timesteps at or above 250 |

Averaging windows use zero-based prediction indices, including the initial
prediction at index 0, with an exclusive end. Each window must satisfy
`0 <= start < end <= effective_refinements + 1`, even when averaging is disabled.
Reduce the windows as well when reducing the iteration counts. With the default
settings, each timestep below 250 makes 6 UNet calls and each later timestep
makes 10; averaging adds no UNet call. Disabling averaging uses the last estimate.
These settings are independent of the shared experiment YAML.

```powershell
uv run diffuse --method renoise --h-params ALL
```

Refresh an existing installation with `uv sync` to register the entry point.
The adapter is also exported as `ReNoiseInversion` from
`image_editing_inversion.inversion`.

ReNoise requires `sampling.eta: 0.0`, finite nonnegative sampling guidance, and
a DDIM scheduler with epsilon prediction, leading timestep spacing, and disabled
sample clipping and dynamic thresholding. Inversion and replay run without
gradients and support `cpu_offload: none`, `model`, and `sequential`, subject to
the existing hardware restrictions. Inversion uses `runtime.inversion_batch_size`;
reconstruction and editing use `runtime.batch_size` and the shared Prompt-to-Prompt
attention policy. This adapter implements deterministic SD 1.5 inversion;
noise regularization and stochastic noise correction are not included.

ReNoise artifacts store the terminal latent plus sampling
`guidance_scale` repeated in a float64 `[T]` tensor. Replay rejects malformed or
non-finite state and incompatible guidance, model, dataset, scheduler, or timestep
settings. The replay hook validates the timestep and latent shape while leaving
the shared denoising state unchanged. Use `edit_artifacts` with the same sampling
guidance and compatible configuration; replay needs neither the hyperparameter
file nor further inversion. Reconstruction is approximate and can differ from
the source because of inversion error, VAE encoding/decoding, and precision.

### Edit from a saved artifact

Schema-v5 collections contain `artifacts/<i>/terminal_latent` tensors with shape
`[1, 4, height/8, width/8]`, plus optional `artifacts/<i>/state/<name>` tensors.
Each state tensor has one entry per timestep and requires the method's registered
denoising hook. The safetensors header has `schema_version: "5"` and `ids`, a
JSON-encoded ordered list of unique nonempty raw dataset UIDs, plus `entries`,
a JSON-encoded list of integer references aligned with those IDs. Each reference
selects a shared production record from the JSON-encoded `contexts` list. This
keeps repeated model/settings records out of large headers. Each record
stores the original model, dataset, scheduler, eta, and timesteps, and optional
`ArtifactProvenance` with model content, dataset content, dtype/tiling, effective
method parameters, sampling guidance, and the resolved model commit for
provenance. Every position
must have exactly one terminal latent. Images, prompts, and masks remain in the
dataset; tensor shapes and dtypes are preserved.

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

repository = DatasetRepository(None)
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
`edit_artifacts` accepts no configuration options and processes every pipeline
parameter file. Results are written under `data/output/<run-id>/` with a
schema-v2 `sweep.json` manifest and `<parameter-file>/<method>/` children containing
`results.jsonl`, `batches.jsonl`, `resolved-config.json`, and images per sample under `images/`.
Run
records contain UIDs, combined artifact paths, and artifact indices rather than
prompts, source images, or masks.

### Add an inversion method

Import adapter contracts from the inversion subpackage:

```python
from image_editing_inversion.inversion import (
    DenoisingHook,
    DenoisingStepState,
    InversionContext,
    InversionMethod,
)
```

Implement `InversionMethod.invert(sample, context)` and return an
`InversionArtifact`. `InversionContext` provides the shared editor and resolved config,
plus dataset reference and fingerprint. A method that needs per-step behavior
also implements `create_denoising_hook(artifact)` to return a `DenoisingHook`.
Optional `validate_inversion_config(config)` checks settings without model weights;
optional `validate_inversion_context(context)` checks the actual scheduler before
samples are processed. Both default to no-ops for existing external adapters.
Implement them to participate fully in whole-sweep preflight, and reuse the same
checks in direct inversion calls. All bundled adapters do so. Replay invokes
neither method, so inversion-only restrictions and hyperparameter files do not
affect `edit_artifacts`.
Override the optional `validate_replay(artifact, context)` to check method-specific
state or configuration before the runner creates that hook; the default does
nothing. Direct editor callers should invoke this validation themselves before
creating a method's hook.
Adapters can override `invert_batch(samples, context)` to perform batched model
calls and must return one artifact per input, in input order. The runner supplies
up to `runtime.inversion_batch_size` uncached samples. The default implementation
calls `invert` sequentially, so existing external adapters continue to work.
Bundled methods retain `invert` as a singleton wrapper. Shared
`context.editor.encode_images(images)` encodes a source batch, and
`context.editor.to_device(tensor, dtype=...)` applies the configured transfers;
omitting `dtype` preserves a tensor's existing dtype.
Raise `ArtifactCompatibilityError` from `image_editing_inversion.artifacts`
for configuration mismatches that replay sweeps can skip; keep malformed-state
errors as ordinary exceptions.

Automatic artifact reuse is opt-in for external adapters. Override
`inversion_cache_parameters(context)` and return a JSON-compatible mapping of
effective inversion settings, including any additional inputs that can change
inversion (such as a stochastic method's seed or versions of helper modules).
The default returns `None`, so artifacts are saved centrally without automatic
reuse. All four bundled adapters opt in; Null-text and ReNoise expose the same
resolved settings used by inversion. Explicit replay does not call this method.
Hooks receive source/target latents and text embeddings before and after each
denoising step. Install the method through the
`image_editing_inversion.methods` Python entry-point group. This discovery group
remains unchanged; Python imports now use `image_editing_inversion.inversion`,
with bundled implementations under `image_editing_inversion.inversion.methods`.
The former methods package has been removed without compatibility wrappers.
Published inversion implementations can then be adapted without changing the
common editor.

To customize attention behavior, subclass `PromptToPrompt` from
`image_editing_inversion.generation` and return that class from the adapter's
`prompt_to_prompt_class` property. The base class exposes `artifacts`,
`settings`, `step_index`, and `pair_maps`. Override `build_pair_map` for token
alignment, `active` for the replacement schedule, or
`replace_cross_attention` / `replace_self_attention` for attention transfer.
`active` receives the attention type, query token count, and UNet layer name.
Each replacement method receives source and target attention probabilities,
the pair index, and the UNet layer name; it must return a tensor with the target
probabilities' shape, device, and dtype. Model loading, four-branch UNet layout,
DDIM denoising, and denoising hooks remain shared.

For example, a subclass can blend mapped source attention with the target's
original attention:

```python
from image_editing_inversion.generation import PromptToPrompt

class BlendedPromptToPrompt(PromptToPrompt):
    @classmethod
    def validate_method_settings(cls, settings):
        if set(settings) != {"target_weight"}:
            raise ValueError("target_weight is required")
        weight = settings["target_weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
            raise ValueError("target_weight must be between 0 and 1")
        return {"target_weight": float(weight)}

    def replace_cross_attention(self, source_prob, target_prob, pair_index, layer_name):
        mapped = super().replace_cross_attention(
            source_prob, target_prob, pair_index, layer_name
        )
        weight = self.settings["target_weight"]
        return (1 - weight) * mapped + weight * target_prob
```

Select `BlendedPromptToPrompt` in `YourMethod`:

```python
@property
def prompt_to_prompt_class(self):
    return BlendedPromptToPrompt
```

Then add these settings to the experiment YAML:

```yaml
methods:
  your-method-id:
    prompt_to_prompt:
      target_weight: 0.25
```

For example, an adapter package can declare:

```toml
[project.entry-points."image_editing_inversion.methods"]
your-method-id = "your_package.method:YourMethod"
```

`YourMethod` must be an `InversionMethod` instance or a zero-argument subclass,
and its `method_id` must equal the entry-point name. An adapter whose editing
hook cannot share the configured GPU batch can set `max_edit_batch_size`;
the runner rejects a larger configured value.
The runner also rejects a GPU batch that mixes P2P subclasses or different
method-specific P2P settings. Replaying saved artifacts requires a registered
adapter for every artifact's method ID.

```powershell
uv run diffuse --method your-method-id --h-params ALL
```

Repeat `--method` for a comparison over the entire dataset, or use `--method ALL`
to select every discovered adapter once. The bundled `ddim`,
`null-text`, `direct-inversion`, and `renoise` methods are available after installing the project;
other method IDs require registered adapters. All methods for a given parameter
file use the same model, sampling, shared Prompt-to-Prompt, and runtime configuration. Both workflows
read `config.yaml` from the working directory. For `diffuse`, `--h-params` is
required: provide a pipeline parameter filename or the exact value `ALL` to
sweep every parameter file.
`edit_artifacts(artifact_id=None)` always sweeps all parameter files.
