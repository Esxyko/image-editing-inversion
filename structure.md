# Project Structure

```text
.
|-- .env.example                # HF_TOKEN template; real .env is ignored
|-- README.md
|-- structure.md
|-- pyproject.toml
|-- config.yaml                 # Shared settings and optional per-method P2P settings
|-- pipeline_h_params/
|   `-- default.yaml            # Step count, guidance, and attention fractions; cwd-relative
|-- method_h_params/
|   |-- Null_text.yaml          # Null-text optimization settings; loaded from cwd
|   `-- ReNoise.yaml            # ReNoise refinement and averaging settings; loaded from cwd
|-- src/
|   `-- image_editing_inversion/
|       |-- __init__.py          # diffuse console entry point
|       |-- cli.py               # Command-line parsing
|       |-- config/
|       |   |-- __init__.py      # Public configuration API
|       |   |-- models.py        # Settings dataclasses and hardware validation
|       |   `-- loader.py        # YAML parsing and validation
|       |-- data/
|       |   |-- __init__.py      # Public dataset API
|       |   |-- schema.py        # Shared dataset references and expected columns
|       |   |-- loader.py        # Fetch and load the project dataset
|       |   |-- identity.py      # Streaming dataset-file and external-image content identity
|       |   `-- repository.py    # Dataset selection, identity, and UID lookup
|       |-- artifacts/
|       |   |-- __init__.py      # Public artifact API
|       |   |-- catalog.py       # Batch staging, atomic group publication, and cache catalog
|       |   |-- collection.py    # Indexed references, schema-v5 validation, and streamed serialization
|       |   |-- layout.py        # Shared method and readable pipeline-group naming
|       |   |-- repository.py    # Raw UID resolution and published collection discovery
|       |   |-- intermediates.py # Validated, atomic per-image component cache
|       |   `-- inversion.py     # Per-image inversion state and compatibility validation
|       |-- inversion/
|       |   |-- __init__.py      # Public methods, adapter contracts, and registry API
|       |   |-- base.py          # InversionMethod adapter interface
|       |   |-- context.py       # InversionContext passed to adapters
|       |   |-- hooks.py         # Denoising state and hook contracts
|       |   |-- registry.py      # Registration and entry-point discovery
|       |   |-- validation.py    # Shared inversion configuration and schedule checks
|       |   `-- methods/
|       |       |-- __init__.py  # Bundled inversion method classes
|       |       |-- common.py    # Shared sources, pivots, validation, artifact building, and reuse
|       |       |-- ddim.py      # Deterministic DDIM inversion baseline
|       |       |-- direct.py    # Conditional pivots, latent residuals, and source-only replay hook
|       |       |-- null_text.py # Pivotal inversion, null-text optimization, and replay hook
|       |       `-- renoise.py   # Iterative DDIM noising, prediction averaging, and replay validation
|       |-- generation/
|       |   |-- __init__.py      # Public generation API; imports Editor on demand
|       |   |-- runtime.py       # ModelRuntime pipeline/device/offload setup
|       |   |-- identity.py      # Selected model weights and effective processing identity
|       |   |-- alignment.py     # Prompt token mapping and PairAttentionMap
|       |   |-- prompt_to_prompt.py # Subclassable attention policy
|       |   |-- processor.py     # Diffusers attention integration
|       |   |-- editor.py        # Shared reconstruction and DDIM editing loop
|       |   `-- result.py        # EditResult image pair
|       `-- workflows/
|           |-- __init__.py      # Public workflow API
|           |-- runner.py        # Sweep preflight, scheduling, configuration, and output routing
|           |-- inversion.py     # Internal InversionCoordinator cache, validation, and publication
|           |-- replay.py        # Internal ReplayPreparer loading, policies, and hooks
|           |-- models.py        # Internal inversion, loading, and prepared-edit records
|           |-- timing.py        # Stage clocks and model-boundary device synchronization
|           `-- output.py        # SweepOutput manifest and per-parameter/method RunOutput
`-- data/                       # Local artifacts; ignored by Git
    |-- state.json              # Saved Hub dataset metadata after first load
    |-- dataset_info.json       # Created after first load
    |-- data-*.arrow            # Saved Hub dataset records after first load
    |-- cache/
    |   `-- inversion/          # Persistent intermediates, independent of replay artifacts
    |       |-- sources/<digest>.safetensors
    |       `-- conditional_pivots/<digest>.safetensors
    |-- artifacts/             # Generated inversions, independent of run output
    |   |-- catalog.json       # Versioned index of reusable inversions
    |   `-- <method>/
    |       `-- steps-50_guidance-7.5/ # Resolved steps and guidance values
    |           |-- h-params.json # Resolved num_inference_steps and guidance_scale
    |           `-- artifacts.safetensors # Numbered tensor entries and embedded ordered IDs
    `-- output/                # Fixed workflow output root
        `-- <run-id>/
            |-- sweep.json     # Schema-v2 parameter/method statuses and aggregate counts
            `-- <parameter-file>/
                `-- <method>/  # resolved-config.json, results.jsonl, batches.jsonl, and images/<uid-with-colons-replaced-by-underscores>/
```

## Responsibilities

- `README.md` describes dataset loading, artifact editing, method adapters,
  and how the combined dataset was constructed.
- `pyproject.toml` defines the project package, the `diffuse` console entry point, bundled
  DDIM, Null-text, Direct Inversion, and ReNoise method entry points, and shared Python
  dependencies.
- `config.yaml` supplies model settings, eta, seed, editing mode, and runtime
  values, including independent inversion_batch_size and editing batch_size;
  it is loaded from the working directory without a filename argument.
  Shared configuration, method hyperparameters, and the dataset stay fixed across
  experiment runs; only `pipeline_h_params/` varies.
- `pipeline_h_params/` supplies grouped YAML files containing sampling step count,
  guidance scale, and the two attention replacement fractions. `diffuse` selects
  a filename with required `--h-params` or processes all top-level YAML files
  sequentially when its value is exactly `ALL`;
  the `edit_artifacts` Python API always processes all files.
- `method_h_params/Null_text.yaml` supplies Null-text optimization settings,
  loaded relative to the working directory and cached by the adapter on its
  first inversion preflight, direct inversion, or cache lookup. Discovery and
  artifact replay do not read this file.
- `method_h_params/ReNoise.yaml` supplies ReNoise refinement counts and prediction
  averaging windows, loaded relative to the working directory and cached by the
  adapter on its first inversion preflight, direct inversion, or cache lookup.
  Discovery and artifact replay do not read it.
- `image_editing_inversion.__init__` exports `diffuse`.
- `image_editing_inversion.cli` provides the parser and entry function for
  `diffuse`, which requires repeatable `--method METHOD|ALL` and `--h-params FILENAME|ALL`.
- `image_editing_inversion.config` loads and validates shared settings and
  optional per-method Prompt-to-Prompt settings from `config.yaml`, discovers
  pipeline parameter files, and resolves one combined `ExperimentConfig` per file.
  Runtime inversion_batch_size is optional and defaults to 1; unindexed CUDA
  hardware validation uses the current device, matching model initialization.
- `image_editing_inversion.data` downloads the private Hub train split on first
  use and loads the saved copy from `data/` afterward. `DatasetRepository`
  selects a Hub or local dataset, validates its schema and identity, and resolves
  samples by UID and exposes ordered UIDs without materializing images.
  Workflows always use the project dataset. Dataset references and expected
  columns live in `schema.py`. `data.identity.DatasetContentIdentity` hashes only
  actual saved Arrow dependencies and raw external source-image files, without
  image decoding. Effective features/format enter identity; descriptions,
  citations, JSON formatting, and external mask contents do not. `DatasetRepository.content_digest` computes this once per invocation.
  First downloads are reloaded from the saved shards before returning.
- `image_editing_inversion.artifacts` owns per-image inversion validation and
  schema-v5 combined safetensors collections. `InversionArtifact` represents one
  image's tensors and compatibility settings; it has no directory serialization.
  `validate_terminal_latent` checks shape, positive spatial dimensions, floating
  dtype, and finite values during construction, serialization, and editing.
  These checks load only one entry or the current batch. Invalid cached tensors
  cause a cache miss; explicit replay treats them as malformed artifacts.
  `ArtifactCollection` stores `artifacts/<i>/terminal_latent` and optional
  `artifacts/<i>/state/<name>` tensors, with metadata containing the schema version
  and JSON-encoded aligned `ids` and `entries` lists. `entries` preserves each
  artifact's original model/dataset/scheduler/eta/timesteps through integer
  references into a deduplicated JSON-encoded `contexts` list, plus optional
  `ArtifactProvenance` containing actual content identities, precision/tiling,
  method parameters, guidance, and a provenance-only resolved model commit.
  Raw UID `ids[i]` identifies position `i`, including
  MagicBrush IDs containing colons. Collections validate unique IDs, complete
  positions, tensor shapes, and timestep counts without materializing all tensors.
  `ArtifactReference(path, index, sample_uid)` selects one entry; loading verifies
  that the UID matches its position and copies only that entry's tensors.
  `ArtifactSettings` supplies current expected model, dataset, scheduler, eta,
  timesteps, and optional content/numerics inputs. Loading restores production
  identity from the file, then compares independent expectations. Directory
  method/step/guidance values must agree with saved production metadata.
  Merges preserve each entry's production fields rather than replacing them
  with current settings. Old schema-v4 files are cache misses and are replaced
  only after successful regeneration; no provenance is inferred.
  `ArtifactRepository` resolves raw IDs across groups and discovers published
  collections in relative-path order followed by positional order. Hidden staging,
  symlinks, legacy sample directories, and hash-named groups are excluded.
  `ArtifactCatalog` owns schema-v2 per-ID cache inputs and references, with one
  checksum per group file. It checks integrity once per unchanged file.
  `store_group` accepts ordered artifacts, aligned optional cache inputs, and
  resolved steps/guidance. Its `group` transaction stages batches in hidden
  temporary files, preserves existing positions, appends new IDs in traversal
  order, and streams the combined file with bounded tensor memory. A completed
  group is atomically published before the catalog is updated, assuming one writer.
  Failed inversion passes retain the previously published group; only published
  inversions can be reused after interruption. Temporary files are cleaned on
  transaction exit, and never discovered or resumed. Older artifact schemas and
  catalog-v1 entries are ignored without deleting their files; regeneration is
  required, and the catalog is replaced only upon successful new publication.
  Compatibility mismatches use `ArtifactCompatibilityError`; malformed state
  remains an ordinary error.
  `IntermediateCache` in `intermediates.py` independently stores per-image tensor
  components under `data/cache/inversion/`, with schema-v1 identity and producer
  batch metadata plus payload checksums. It validates tensor names, shape, dtype,
  and finiteness; malformed or unreadable entries are misses. Writes publish one
  complete entry atomically, assuming one writer. No catalog, replay discovery,
  or automatic eviction applies to these entries. Storage depends on tensor
  contracts only, without inversion or generation imports.
- `image_editing_inversion.inversion` owns the inversion adapter interface,
  `InversionContext`, denoising state and hooks, and method registration and
  entry-point discovery. Optional `validate_inversion_config` and
  `validate_inversion_context` default to no-ops for external adapters. Bundled
  methods share these checks with their actual inversion entry points;
  `validation.py` supplies configuration and inverse-schedule helpers.
  Replay does not call either preflight interface or read inversion hyperparameters.
  Its `methods.ddim` submodule implements the first bundled
  inversion algorithm using the shared pipeline and sampling guidance, with
  a separate inverse scheduler. Its `methods.null_text` submodule builds conditional-only
  DDIM pivots and optimizes per-step unconditional embeddings, then supplies
  their replay hook. Its `methods.direct` submodule builds conditional-only DDIM pivots,
  records per-step latent residuals at the sampling guidance, and supplies a
  source-only after-step replay hook without optimization or extra settings.
  Its `methods.renoise` submodule iteratively reverses the editor's deterministic DDIM
  updates using fixed previous latents and prediction averaging. It records
  sampling guidance and supplies a validating hook that leaves denoising state
  unchanged. It does not implement regularization or stochastic noise correction.
  Adapters can validate method-specific replay compatibility, select a
  Prompt-to-Prompt subclass, and opt into reuse through
  `inversion_cache_parameters`. External adapters disable caching by default.
  InversionMethod.invert_batch defaults to sequential invert calls for external
  adapters. All bundled methods batch source encoding and model calls, retain
  invert as a singleton wrapper, and split results into existing per-image
  artifacts. Null-text uses separate float32 Adam parameters and stopping
  decisions for each image, batching only active images during optimization.
  `methods.common` owns finite-value and strict DDIM scheduler checks, guidance
  state and compatibility checks, and construction of independent CPU artifacts.
  Its `SharedInversionComponents` provider reuses source VAE latents and caption
  embeddings across all bundled methods and conditional trajectories across
  Direct and Null-text. Only missing per-image components enter model calls;
  results are assembled in input order with memory bounded by a batch.
  Source identity version 2 includes dataset/UID/image/prompt, data and model
  contents, effective image/text processing, dtype, and VAE tiling,
  excluding source-code hashes, hardware identity, PyTorch builds,
  CPU thread counts, dependency versions, and small numerical switches. Pivot
  identity adds UNet content and scheduler/timesteps to semantic source identity,
  while source encoding matches only VAE/text encoder/tokenizer contents. Both
  exclude exact input-tensor checksums, guidance, and downstream method settings. Payload
  checksums still validate stored tensors.
  Intermediate identity excludes batch size and composition, allowing small
  numerical differences. Shared computation and cache loading disable gradients
  and inference mode so cached tensors remain suitable for Null-text autograd.
  `InversionContext.source_identity()` supplies the same inputs to final cache
  identity version 2 and intermediate component identity version 2. Its optional
  `dataset_content` is filled by workflows.
  `InversionContext.components` defaults to `None`; direct calls then compute
  locally. Method-specific algorithms and hooks remain in their method modules.
- `image_editing_inversion.generation` owns model loading, the subclassable
  Prompt-to-Prompt attention policy, and shared reconstruction/editing.
  `ModelRuntime` owns pipeline setup, numerical cache settings, and scheduler
  reconfiguration without reloading weights; `Editor` owns shared prompt encoding,
  denoising, hook execution, and synchronized configuration updates.
  Editor supplies shared batch image encoding and device transfers, including
  pinned CPU-to-CUDA inputs and pivot retrieval. `generation.identity.ModelContentIdentity`
  identifies selected weight files and effective model/image/tokenizer settings
  from one resolved snapshot. The actual Hub commit is recorded as provenance
  rather than an independent content match condition. Runtime cache
  settings retain dtype and VAE tiling sizes/overlap, excluding slicing, offload,
  TF32, deterministic/SDPA switches, batch size, device/hardware identity, builds,
  CPU thread counts, and dependency versions. Scheduler normalization removes
  version/default-value/class-name metadata and inactive clipping/thresholding
  settings or unused beta defaults, retaining effective settings. Attention chunking applies only to editing.
  Attention alignment, policy, and Diffusers integration have separate files.
  Editor encloses validation, encoding, attention installation, denoising,
  decoding, and postprocessing in an offload cleanup boundary. Installed attention
  processors are restored on every exit; cleanup errors preserve execution errors.
- `image_editing_inversion.workflows` composes the other modules.
  `WorkflowRunner` owns parameter selection, configuration switching, batch
  scheduling, output routing, and sweep state. Diffusion preflights every selected
  parameter-file/method combination before any sample inversion: configuration,
  hyperparameter files, editing limits, and P2P settings before model loading,
  then actual scheduler constraints using one shared model. Invalid combinations
  are collected and recorded in the sweep, with no per-sample results.
  `InversionCoordinator` owns configuration-based cache identities, cache lookup,
  missing-sample reads and inversion, external artifact validation, staging, and
  group publication. The coordinator adds missing production provenance to
  new external adapter results and validates saved method parameters against
  cache inputs. Null-text matches effective stopping thresholds, and ReNoise
  matches selected prediction intervals for the actual schedule, omitting
  inactive windows/caps and post-average predictions. `ReplayPreparer` owns collection loading, dataset identity,
  method lookup, P2P validation, and hook preparation for both workflows.
  Collaborators receive explicit dependencies and do not hold the runner.
  `models.py` defines their internal dataclass records; these are not public APIs.
  The runner shares the dataset/model across sequential parameter files and
  preserves sample order, method order, and batch boundaries. Diffusion processes
  every dataset UID for each requested
  method, checking inversion-sized UID groups before reading samples, reusing
  individual cache hits, reading and batching only misses, and staging new tensors
  until the entire inversion group succeeds.
  It publishes the combined file before loading independently sized edit batches.
  Only small per-image timing records persist between passes. Partial edit batches
  flush at method and parameter-file boundaries. Results include file and index,
  inversion/edit batch IDs, inversion batch duration, and actual uncached count, with amortized
  per-image inversion time; failed inversion batches record affected UIDs and abort.
  Replay skips incompatible artifact/parameter pairs and flushes batches when
  the method changes. It performs no inversion preflight. Public entry functions
  are `run_methods(method_ids, *, pipeline_h_params_file=None)` and
  `edit_artifacts(artifact_id=None)`; both write to `data/output/`.
  Method selectors expand exact uppercase `ALL` to all discovered adapter IDs in
  sorted order, including installed and programmatically registered adapters.
  Duplicates run once in first-appearance order. The public `run_methods` entry
  validates nonempty selections, registered IDs, and portable method directory
  names before constructing the runner.
  `SweepOutput` owns the parent directory and schema-v2 manifest, with method
  statuses/counts and parameter-level aggregates. `RunOutput` owns each
  `<parameter-file>/<method>` child, resolved snapshots including method ID,
  images, and records. Image directories use UIDs with colons replaced by
  underscores; records and artifacts retain raw UIDs. Both workflows use this
  layout even for one method. Replay reuses outputs across artifact groups and routes loading
  errors using the method directory from the artifact reference.
  `RunOutput` writes one schema-v1 `batches.jsonl` record per attempted inversion
  batch, editing method segment, and publication, including failures. Stage clocks
  distinguish reads, cache lookup, adapter execution, external validation, staging,
  edit preparation, editor execution, image writes, and publication. Publication
  assembly/checksums/catalog time is recorded once for the group. CUDA/MPS model
  stage boundaries synchronize; CPU uses wall time. Legacy successful sample
  timing fields retain their meaning; aggregate timing uses batch records.
  Workflow inversion injects the shared-component provider without changing
  method order or publication boundaries. Final-artifact hits bypass intermediate
  computation. Batch diagnostics count source/pivot hits and misses and record
  nonfatal cache-write errors; component work remains in the method stage.
  Final and intermediate cache identities exclude source-code hashes and method
  class paths; code changes do not automatically invalidate reusable tensors.
- `data/` is not tracked by Git. Its root holds the saved Hub dataset after the
  first load; `cache/inversion/` holds reusable per-image intermediates,
  `artifacts/` holds generated inversions and their catalog, while
  `output/` is the fixed root for experiment outputs.

## Dependency boundaries

- `config`, `artifacts`, and `data` do not depend on inversion, generation, or
  workflow orchestration.
- `inversion` depends on configuration and artifacts. Editor and attention policy
  types are imported only under `TYPE_CHECKING`; discovery does not import
  workflows or the model runtime. Concrete methods use the editor supplied
  by `InversionContext`; all bundled methods import Diffusers only when
  inversion or scheduler preflight executes.
- `generation` depends on configuration, artifacts, and denoising hook contracts.
  Importing its attention policy does not import Diffusers; `Editor` is exported
  lazily and model weights are loaded only when an editor is constructed.
- `workflows` coordinates the lower-level modules. Its output component uses
  configuration and image contracts without importing the editor or runner.
  Internal coordination, replay, records, and timing modules never import the
  runner; lower-level modules do not depend on these collaborators.
- `cli.py` imports workflows only when executing a parsed command.

Each subpackage exports its public API through `__init__.py`. Implementation
helpers remain private. The old flat `artifact.py`, `dataset.py`, and `runner.py`
paths have no compatibility wrappers. Use `image_editing_inversion.inversion`
for inversion contracts and discovery, and `image_editing_inversion.inversion.methods`
for bundled algorithms. The renamed package has no compatibility wrappers.
The console entry point is `diffuse`, without compatibility wrappers. Artifact
replay is available through the `edit_artifacts` Python API. The plugin entry-point
group (`image_editing_inversion.methods`) remains unchanged. New artifacts use
schema v5 collections; older schemas and digest groups require regeneration. Pipeline
parameters now live in separate YAML files; run outputs use filename/method children
and a schema-v2 parent sweep manifest. Existing run folders are not migrated.
`load_config` requires a pipeline parameter file.
