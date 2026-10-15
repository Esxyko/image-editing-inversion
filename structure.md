# Project structure

The package has seven subpackages. Public APIs are exported by their `__init__.py`
files; execution collaborators and implementation helpers remain internal.
Configuration and storage locations are fixed relative to `src/`.

## Source layout

```text
.
|-- README.md                       # Setup and common workflows
|-- structure.md                    # Architecture and ownership
|-- docs/
|   |-- dataset.md                  # Schema, provenance, loading, identity
|   |-- methods.md                  # Algorithms, restrictions, hyperparameters
|   |-- artifacts.md                # Storage, caches, replay, timing
|   |-- evaluation.md               # Archived metrics, source groups, core times, JSON
|   |-- direct-reconstruction-tests.md # Direct GPU diagnostic controls and comparisons
|   `-- adapters.md                 # Extension contracts and API migration
|-- tests/test_direct_reconstruction.py # Opt-in Direct GPU integration tests/probes
|-- tests/test_direct_source_state.py # CPU original-embedding, native residual and collection compatibility regressions
|-- pyproject.toml                  # Dependencies, CLI and method entry points
|-- config.yaml                     # Shared experiment configuration
|-- scripts/                        # Sequential full-dataset diffuse CLI wrappers
|   |-- 1.py                        # default, high_steps, low_steps
|   |-- 2.py                        # high/low_editing_guidance, high_cross_replace
|   `-- 3.py                        # low_cross_replace, high/low_self_replace
|-- pipeline_h_params/              # Sampling, editing CFG and attention sweep parameters
|   |-- default.yaml
|   |-- high_cross_replace.yaml
|   |-- high_editing_guidance.yaml  # Editing CFG 10.0; other settings match default
|   |-- high_self_replace.yaml
|   |-- high_steps.yaml
|   |-- low_cross_replace.yaml
|   |-- low_editing_guidance.yaml   # Editing CFG 5.0; other settings match default
|   |-- low_self_replace.yaml
|   `-- low_steps.yaml
|-- method_h_params/
|   |-- DDIM.yaml
|   |-- Direct.yaml
|   |-- Null_text.yaml
|   `-- ReNoise.yaml
|-- .env.example                    # HF_TOKEN template; .env is ignored
`-- src/image_editing_inversion/
    |-- __init__.py, cli.py          # diffuse/experiment/evaluate exports and command parsing
    |-- diagnostic.py              # Temporary experiment-diagnostic CLI (Null-text only)
    |-- _project.py                 # Internal source-relative fixed paths
    |-- config/
    |   |-- models.py               # Validated settings and hardware checks
    |   |-- validation.py           # Shared semantic checks and ConfigError
    |   |-- method_parameters.py    # Method inversion CFG defaults and loading
    |   `-- loader.py               # Strict YAML schemas and invocation snapshot
    |-- data/
    |   |-- schema.py, loader.py     # Project eval split contract and loading
    |   `-- identity.py, repository.py # Content digest and ordered UID access
    |-- artifacts/
    |   |-- inversion.py            # Per-image state and compatibility
    |   |-- layout.py               # Canonical published group parsing
    |   |-- collection.py           # Indexed schema-v1/v2 safetensors and streaming
    |   |-- integrity.py            # Metadata/payload checksums and legacy catalog verification
    |   |-- catalog.py              # Cache catalog and group publication
    |   `-- repository.py, intermediates.py # Discovery and component storage
    |-- inversion/
    |   |-- base.py, context.py     # Adapter interface and invocation context
    |   |-- hooks.py, registry.py   # Denoising contracts and adapter factories
    |   |-- validation.py           # Sampling, inverse schedule, finite-value checks
    |   |-- components.py           # Source encodings, conditional pivots, reuse
    |   `-- methods/
    |       |-- common.py           # Guidance, DDIM checks, artifact building, cleanup
    |       |-- source_embeddings.py # Captured source encodings, validation and reuse
    |       `-- ddim.py, direct.py, null_text.py, renoise.py
    |-- generation/
    |   |-- runtime.py, identity.py # Pipeline setup and content identity
    |   |-- denoising.py            # Internal DDIM executor and hook/processor lifecycle
    |   |-- alignment.py, prompt_to_prompt.py # Token mapping and attention policy
    |   `-- processor.py, editor.py, result.py # Diffusers integration and editing
    |-- evaluation/
    |   |-- __init__.py, runner.py   # evaluate_archives facade and shared invocation resources
    |   |-- models.py, archive.py   # Settings, validated manifests and streamed RGB ZIP batches
    |   |-- metrics.py              # Pinned CLIP and masked background metric models
    |   `-- timing.py, output.py    # Core batch allocation, source statistics, atomic JSON
    `-- workflows/
        |-- runner.py               # Reusable public facade and selector expansion
        |-- session.py              # One invocation's resources, scheduling, lifecycle
        |-- preflight.py            # Model-free and scheduler sweep validation
        |-- inversion.py            # Reuse, missing-image inversion, publication
        |-- replay.py, editing.py   # Artifact preparation and editing coordination
        |-- models.py, timing.py, output.py # Records, clocks, manifests, RGB tensor batches
        |-- archive.py              # Verified run ZIP publication and source removal
        `-- null_text_diagnostic.py # Temporary observing adapter/session and diagnostic JSONL
```

## Ownership and boundaries

| Package | Owns | Dependencies within the project |
| --- | --- | --- |
| `config` | Configuration models, schemas, shared YAML snapshot, hardware validation | Internal project paths |
| `data` | Fixed project eval dataset, schema, UID index, content identity | Internal project paths |
| `artifacts` | Tensor contracts, compatibility, collection serialization, catalogs, publication | Internal project paths |
| `inversion` | Adapter contracts, factories, algorithms, shared component computation | Configuration and artifacts |
| `generation` | Model runtime, attention control, reconstruction/editing | Configuration, artifacts, denoising contracts |
| `workflows` | Invocation resources, preflight, scheduling, replay, outputs | Configuration, data, artifacts, inversion, generation |
| `evaluation` | Archived RGB evaluation, source groups, core timing, JSON reports | Dataset repository and internal project paths |

Root-level `scripts/1.py`, `2.py`, and `3.py` only orchestrate existing `diffuse`
CLI invocations. Each processes its fixed parameter group sequentially with every
registered method and stops on failure or interruption. Script progress logs append
to `data/output/script-logs/<script-number>.log`; workflow outputs keep their
existing per-run layout.

Lower-level packages do not import workflows. Inversion's editor/policy types are
imported only for type checking; algorithms use the editor supplied through
`InversionContext`. Shared components are independent of concrete method modules.
Concrete methods import Diffusers when inversion or scheduler preflight executes.
Generation exports `Editor` lazily; defining attention policies does not import
Diffusers. Workflows import the editor when the model phase begins. CLI parsing
imports workflows only when executing the parsed command.
Evaluation is independent of workflows/generation and does not load the diffusion
model. Its public facade and CLI parsing do not import optional metric models;
execution loads the `evaluation` extra only when requested. It shares a dataset,
pinned CLIP/LPIPS models and target-text features across selected ZIPs, and reads
one saved tensor batch at a time without extracting archives. Core timings are
allocated through saved batch IDs before source-specific aggregation.

`ProjectPaths` derives the root from the package's fixed position under `src/`.
It accepts no root argument and does not search the working directory. The root
must contain `config.yaml`; loaders validate YAML and parameter directories.
Relative local-model paths resolve from that root. Repository/cache constructors
accept no directory overrides. `ExperimentConfig.project` carries these locations
but is excluded from serialization and cache matching.

## Invocation lifecycle

1. Expand method selectors once and load a shared configuration snapshot with
   resolved configurations for the selected pipeline files.
2. Create the sweep manifest and one fresh adapter per selected method. For
   inversion, validate all configuration/method combinations before loading data.
3. Load the fixed project dataset. For inversion, filter UIDs by source, then apply
   the optional sample limit in dataset order. Load one model and validate each
   resolved scheduler. Replay uses saved state without inversion preflight, but loads
   current method parameter files for strict snapshot matching.
4. Process parameter files sequentially. For each inversion method, resolve cache
   hits before reading missing samples, stage batches, and publish the complete group.
5. Read published artifacts in independently sized editing batches; validate, prepare
   hooks/policies, edit, save RGB tensor batches, and record results and timings.
6. Finalize method/file/sweep statuses. Exceptions preserve their original cause;
   cleanup/reporting failures are attached as notes where an error is already active.

`WorkflowRunner` stores only its parameter-file selection. Each call creates a new
`WorkflowSession`; adapters, settings, dataset/model ownership, and mutable run
state are not shared across calls. Session collaborators receive explicit inputs
and never import or retain the runner. `ReplayPreparer` uses the session's adapters
for both fresh inversions and saved artifacts. Stable cache inputs are resolved
once per method/configuration context and combined with each UID.

Pipeline configuration contains steps, required `sampling.editing_guidance_scale`,
and attention fractions. Method configuration owns inversion CFG and algorithm settings.
The session binds each pipeline/method combination to a normalized
`ExperimentConfig.method_h_params` snapshot before preflight or execution and
reconfigures the shared editor when switching methods. Source denoising uses
editing CFG for every bundled method. Null-text hooks request their saved
optimization CFG, which is validated against editing CFG. Null-text optimization
and target denoising also use editing CFG. Null-text pivots use inversion CFG;
Direct conditional pivots retain CFG 1.
Cache identity version 4 includes all method parameters and separate pipeline editing CFG.
Saved provenance carries the complete method snapshot and `editing_guidance_scale`;
artifacts without either or with different current settings cannot be reused or replayed.
Legacy provenance remains readable without automatically migrating editing CFG.
Group directory guidance and per-step `guidance_scale` continue to mean inversion CFG.
Null-text separately saves `optimization_guidance_scale`, checks it against
editing CFG on replay, and versions the optimization rule in cache identity.
It also saves `pivot_guidance_scale`, validates it against inversion CFG, and
versions the pivot-guidance rule. Older artifacts lacking this state require regeneration.

The session owns one reusable read thread pool and closes it and its model runtime
on success or failure. Runtime schedule lists and ordinary detached empty-caption
embeddings are cached within that invocation. `Editor` validates and organizes
inputs/results; its internal denoising executor owns the DDIM loop, hook dispatch,
and attention processor installation/restoration. Built-in stateful methods use
optional batch hooks in all-source/all-target order; external pair hooks retain
their original contract. Model input views decode four columns eagerly and masks
only when requested, while public `sample()` still returns all five columns.
Collection readers are scoped to one lookup/edit batch and clone tensors before
closing their mappings, so Windows group replacement remains possible.

`experiment` requires one pipeline parameter file and a positive `--nsamples`,
rejects `--h-params ALL`, and defaults to every discovered method. Its optional,
repeatable `--method` selectors use the same expansion as `diffuse`.
`run_methods` and `WorkflowRunner.run_methods` accept optional `dataset`
and `nsamples` keywords, validated before session creation. Each method processes
the same selected UID prefix; selection retains dataset/cache identities and
publication preserves artifacts outside that prefix. Limited invocations record
their requested `nsamples` in the schema-v2 sweep manifest; unlimited runs omit it.

`workflows.compress_run(run_id)` is a separate Python API backed by the internal
`RunArchiver`. It uses only standard-library archive/file operations and fixed
project paths, without creating a workflow session or loading dataset/model
resources. It accepts one inactive run with manifest status `ok` or `error`,
verifies a streaming ZIP and an unchanged source snapshot, atomically replaces
the sibling ZIP, then revalidates and deletes the source. Publication failures
preserve the source and old ZIP; deletion failures retain the new ZIP and report
its path. Callers must exclude concurrent writers.

## Fixed storage layout

```text
data/                               # Generated files ignored except metrics/ reports
|-- state.json, dataset_info.json, data-*.arrow
|-- cache/inversion/
|   |-- sources/<digest>.safetensors
|   `-- conditional_pivots/<digest>.safetensors
|-- cache/evaluation/               # Reusable pinned CLIP and SqueezeNet downloads
|-- artifacts/
|   |-- catalog.json                # Schema v2
|   `-- <method>/steps-N_guidance-G/
|       |-- h-params.json
|       `-- artifacts.safetensors   # Schema v1/v2, ordered IDs and production metadata
|-- metrics/                        # Version-controlled evaluation reports
|   |-- <UTC-time>-<unique-id>/
|   |   `-- <ZIP-stem>.json         # One report: parameter -> method -> groups, scores, timing
|   `-- <evaluation-date>/          # Historical one-off reports retained unchanged
|       |-- clip_masked_metrics.xlsx # Parameter rows, metric columns, one sheet per method
|       |-- summary.json            # Shared metric protocol, provenance and evaluation timing
|       `-- <parameter-file>/
|           `-- metrics.json        # All methods: statistics, per-image scores, evaluation timing
`-- output/
    |-- <run-id>.zip                # Explicitly archived run; original directory removed
    `-- <run-id>/
        |-- sweep.json              # Schema v2
        `-- <parameter-file>/<method>/
            |-- resolved-config.json
            |-- results.jsonl
            |-- batches.jsonl       # Schema v2 batch records
            `-- tensors/batch-<edit_batch_id>.safetensors # RGB output schema v1
```

New evaluation reports consolidate all parameter files and methods of each ZIP in
one JSON. Each method has all/PIE-Bench/MagicBrush statistics, per-image scores,
complete archived settings and two-stage core times from `batches.jsonl`.
Evaluation elapsed time is separate. Historical reports retain their original
layout and metric-evaluation-only timing. See [evaluation](docs/evaluation.md).

Group publication assumes one writer and keeps tensor memory bounded by batches.
New collections embed metadata and per-tensor checksums. Replay verifies loaded
tensors; publication verifies the entire existing group before merging, including
unselected records. Valid embedded checksums permit recovery when catalog publication
lagged behind the file. Legacy collections require a matching catalog checksum;
unverifiable existing groups are preserved and block replacement.
Final artifacts and intermediate components have independent caches. Collection
schema v2 adds per-image static state while retaining v1 readers and writes for
methods without static tensors. Filenames, UID ordering and RGB output layouts
remain unchanged. All bundled methods bind original source embeddings to their
artifacts and reuse them while retaining native inversion and denoising behavior.
Direct retains its source-only residual calibration/replay. DDIM, Null-text and
ReNoise receive no additional latent correction. The forced source-trajectory
overwrite variant has been withdrawn; its artifacts require regeneration.
Pivot cache identity includes its CFG and checksums of its actual source
latent/embedding inputs, including unconditional embeddings for guided pivots.
New editing outputs store uncompressed CPU uint8 RGB batches; sample records identify
their relative file and index. Historical PNG outputs remain untouched. See
[storage contracts](docs/artifacts.md) for compatibility, interruption,
and timing rules, and [migration guidance](docs/adapters.md#migration) for API changes.
