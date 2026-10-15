# Project structure

This reference defines module responsibilities and dependency boundaries. Start
with the [README](README.md) for setup and commands, or use the
[configuration](docs/configuration.md), [workflow](docs/workflows.md), and
[storage](docs/artifacts.md) guides for operational details.

The package has seven subpackages. Their `__init__.py` files export public APIs;
execution collaborators and implementation helpers remain internal. Configuration
and storage locations are fixed relative to `src/`.

- [Source layout](#source-layout)
- [Ownership and boundaries](#ownership-and-boundaries)
- [Invocation lifecycle](#invocation-lifecycle)
- [Fixed storage layout](#fixed-storage-layout)

## Source layout

```text
.
|-- README.md                       # Setup and common workflows
|-- structure.md                    # Architecture and ownership
|-- docs/
|   |-- configuration.md            # Configuration ownership, CFG, runtime, paths
|   |-- workflows.md                # CLI, Python API, selection, replay, archiving
|   |-- dataset.md                  # Schema, provenance, loading, identity
|   |-- methods.md                  # Algorithms, restrictions, hyperparameters
|   |-- artifacts.md                # Inversion storage, caches, publication, replay
|   |-- run-outputs.md               # RGB batches, result records, timing, ZIP format
|   |-- null-text-diagnostics.md     # Optimization observations and replay comparisons
|   |-- evaluation.md               # Archived metrics, core timing, JSON and summaries
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
    |   |-- summary.py              # Standard-library extraction of report statistics to Markdown/CSV
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
| `evaluation` | Archived RGB metrics, source grouping, core timing, JSON reports, and Markdown/CSV summaries | Dataset repository and internal project paths |

### Dependency and import rules

Lower-level packages do not import workflows. Inversion imports editor and policy
types only for type checking; algorithms use the editor supplied through
`InversionContext`. Shared inversion components do not depend on concrete methods.

Heavy dependencies load when needed. Concrete methods import Diffusers during
inversion or scheduler preflight. Generation exports `Editor` lazily, so defining
attention policies does not import Diffusers. Workflows import the editor when the
model phase begins; CLI parsing imports workflows only when executing a command.

Evaluation does not depend on workflows or generation and never loads the diffusion
model. Its public facade and CLI parser do not import optional metric models.
Execution shares the dataset, pinned metric models, and target-text features across
ZIP inputs, reading one saved tensor batch at a time. Core timings are allocated
by batch ID before aggregation by source.

`evaluation.summary` uses only the standard library to read existing JSON group
statistics. It keeps each report, parameter file, method, and source group separate.
Markdown and CSV are published together under `data/metrics/summaries/`, which is
excluded from automatic report discovery.

### Paths and entry points

`ProjectPaths` derives the project root from the package's fixed position under
`src/`. It accepts no root argument and does not search the working directory.
The root must contain `config.yaml`; loaders validate YAML and parameter
directories. Local model paths resolve from that root. Repository and cache
constructors accept no directory overrides. `ExperimentConfig.project` exposes
these paths but is excluded from serialization and cache matching.

The root-level `scripts/1.py`, `2.py`, and `3.py` orchestrate sequential `diffuse`
invocations only. Each stops on failure or interruption and appends progress to
`data/output/script-logs/<script-number>.log`. See
[sequential sweep scripts](docs/workflows.md#sequential-sweep-scripts).

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

### Resource ownership

`WorkflowRunner` stores only the parameter-file selection. Every call creates a
fresh `WorkflowSession` with its own adapters, configuration, dataset, model, and
mutable state. Collaborators receive explicit inputs and do not import or retain
the runner. `ReplayPreparer` uses the session's adapters for fresh and saved
artifacts. Stable cache inputs are resolved once per method/configuration and
combined with each UID.

The session owns and closes its model runtime and reusable read thread pool on
success or failure. Runtime schedules and detached empty-caption embeddings are
cached within that invocation. `Editor` validates inputs and results; its internal
denoising executor owns the DDIM loop, hook dispatch, and attention processor
installation and restoration.

Built-in stateful methods use batch hooks in all-source/all-target order. External
pair hooks keep their existing contract. Model input views eagerly decode UID,
source image, and both prompts; masks load only when requested. Public `sample()`
still returns all five columns. Collection readers are scoped to one lookup or
editing batch and clone tensors before closing memory mappings, allowing group
replacement on Windows.

### Configuration and compatibility

Pipeline configuration owns sampling steps, editing CFG, and attention fractions.
Method configuration owns inversion CFG and algorithm settings. The session binds
each combination to a normalized `ExperimentConfig.method_h_params` snapshot
before preflight and reconfigures the shared editor when changing methods.

Cache identity version 4 and saved provenance include the complete method snapshot
and separate editing CFG. Group names and historical `guidance_scale` fields use
inversion CFG. Null-text also saves optimization and pivot CFG, validating each
against the corresponding current setting. See [configuration](docs/configuration.md)
and [method requirements](docs/methods.md) for guidance behavior and regeneration
rules.

`experiment` and the Python APIs select the same UID prefix for every method,
filtering by source before applying a sample limit. Selection retains the combined
dataset's identity, and publication preserves artifacts outside that prefix.
See [workflow selection](docs/workflows.md#small-comparisons).

### Independent archiving

`workflows.compress_run(run_id)` delegates to the internal `RunArchiver`. It uses
standard-library file and archive operations with fixed project paths; it does not
create a workflow session or load data or models. It verifies and publishes a ZIP,
then removes the source run. Callers must exclude concurrent writers. Failure
behavior is defined in the [archive contract](docs/run-outputs.md#run-archives).

## Fixed storage layout

```text
data/
|-- state.json, dataset_info.json, data-*.arrow  # Saved eval dataset
|-- cache/
|   |-- inversion/
|   |   |-- sources/<digest>.safetensors
|   |   `-- conditional_pivots/<digest>.safetensors
|   `-- evaluation/                           # Pinned metric model downloads
|-- artifacts/
|   |-- catalog.json                          # Catalog schema v2
|   `-- <method>/steps-N_guidance-G/
|       |-- h-params.json
|       `-- artifacts.safetensors             # Collection schema v1/v2
|-- metrics/                                  # Version-controlled reports
|   |-- <UTC-time>-<unique-id>/<ZIP-stem>.json
|   `-- summaries/<UTC-time>-<unique-id>/
|       |-- report.md
|       `-- summary.csv
`-- output/
    |-- script-logs/<script-number>.log
    |-- <run-id>.zip                           # Archived run; source removed
    `-- <run-id>/
        |-- sweep.json                        # Manifest schema v2
        `-- <parameter-file>/<method>/
            |-- resolved-config.json
            |-- results.jsonl
            |-- batches.jsonl                 # Timing schema v2
            `-- tensors/batch-<edit_batch_id>.safetensors  # RGB schema v1
```

Generated data is ignored by Git except for `data/metrics/`. Previously tracked
files and historical reports remain in place. Historical evaluation directories
may contain `clip_masked_metrics.xlsx`, `summary.json`, and per-parameter
`metrics.json`; their metric-evaluation timing differs from current core timing.

Group publication assumes one writer and keeps tensor memory bounded by batches.
Final inversion artifacts and intermediate components have independent caches.
Collections with static state use schema v2; those without it retain v1. All
bundled methods save original source embeddings and retain their native latent
updates. The withdrawn source-trajectory overwrite variant requires regeneration.

See [artifact contracts](docs/artifacts.md) for checksums, cache identity,
interruption recovery, and replay; [run outputs](docs/run-outputs.md) for RGB and
timing formats; and [evaluation](docs/evaluation.md) for archived reports and
summary publication.
