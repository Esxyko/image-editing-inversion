# Project structure

The package has six subpackages. Public APIs are exported by their `__init__.py`
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
|   `-- adapters.md                 # Extension contracts and API migration
|-- pyproject.toml                  # Dependencies, CLI and method entry points
|-- config.yaml                     # Shared experiment configuration
|-- pipeline_h_params/default.yaml  # Sampling and attention sweep parameters
|-- method_h_params/
|   |-- Null_text.yaml
|   `-- ReNoise.yaml
|-- .env.example                    # HF_TOKEN template; .env is ignored
`-- src/image_editing_inversion/
    |-- __init__.py, cli.py          # diffuse/experiment exports and command parsing
    |-- _project.py                 # Internal source-relative fixed paths
    |-- config/
    |   |-- models.py               # Validated settings and hardware checks
    |   |-- validation.py           # Shared semantic checks and ConfigError
    |   `-- loader.py               # Strict YAML schemas and invocation snapshot
    |-- data/
    |   |-- schema.py, loader.py     # Project eval split contract and loading
    |   `-- identity.py, repository.py # Content digest and ordered UID access
    |-- artifacts/
    |   |-- inversion.py            # Per-image state and compatibility
    |   |-- layout.py               # Canonical published group parsing
    |   |-- collection.py           # Indexed schema-v1 safetensors and streaming
    |   |-- catalog.py              # Cache catalog and group publication
    |   `-- repository.py, intermediates.py # Discovery and component storage
    |-- inversion/
    |   |-- base.py, context.py     # Adapter interface and invocation context
    |   |-- hooks.py, registry.py   # Denoising contracts and adapter factories
    |   |-- validation.py           # Sampling, inverse schedule, finite-value checks
    |   |-- components.py           # Source encodings, conditional pivots, reuse
    |   `-- methods/
    |       |-- common.py           # Guidance, DDIM checks, artifact building, cleanup
    |       `-- ddim.py, direct.py, null_text.py, renoise.py
    |-- generation/
    |   |-- runtime.py, identity.py # Pipeline setup and content identity
    |   |-- denoising.py            # Internal DDIM executor and hook/processor lifecycle
    |   |-- alignment.py, prompt_to_prompt.py # Token mapping and attention policy
    |   `-- processor.py, editor.py, result.py # Diffusers integration and editing
    `-- workflows/
        |-- runner.py               # Reusable public facade and selector expansion
        |-- session.py              # One invocation's resources, scheduling, lifecycle
        |-- preflight.py            # Model-free and scheduler sweep validation
        |-- inversion.py            # Reuse, missing-image inversion, publication
        |-- replay.py, editing.py   # Artifact preparation and editing coordination
        `-- models.py, timing.py, output.py # Records, clocks, manifests, RGB tensor batches
```

## Ownership and boundaries

| Package | Owns | Dependencies within the project |
| --- | --- | --- |
| `config` | Configuration models, schemas, shared YAML snapshot, hardware validation | Internal project paths |
| `data` | Fixed project eval dataset, schema, UID index, content identity | Internal project paths |
| `artifacts` | Tensor contracts, compatibility, collection serialization, catalogs, publication | Internal project paths |
| `inversion` | Adapter contracts, factories, algorithms, shared component computation | Configuration and artifacts |
| `generation` | Model runtime, attention control, reconstruction/editing | Configuration, artifacts, denoising contracts |
| `workflows` | Invocation resources, preflight, scheduling, replay, outputs | All lower-level packages |

Lower-level packages do not import workflows. Inversion's editor/policy types are
imported only for type checking; algorithms use the editor supplied through
`InversionContext`. Shared components are independent of concrete method modules.
Concrete methods import Diffusers when inversion or scheduler preflight executes.
Generation exports `Editor` lazily; defining attention policies does not import
Diffusers. Workflows import the editor when the model phase begins. CLI parsing
imports workflows only when executing the parsed command.

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
   resolved scheduler. Replay uses saved state without inversion preflight or method parameter files.
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

## Fixed storage layout

```text
data/                               # New generated files ignored; tracked history retained
|-- state.json, dataset_info.json, data-*.arrow
|-- cache/inversion/
|   |-- sources/<digest>.safetensors
|   `-- conditional_pivots/<digest>.safetensors
|-- artifacts/
|   |-- catalog.json                # Schema v2
|   `-- <method>/steps-N_guidance-G/
|       |-- h-params.json
|       `-- artifacts.safetensors   # Schema v1, ordered IDs and production metadata
`-- output/<run-id>/
    |-- sweep.json                  # Schema v2
    `-- <parameter-file>/<method>/
        |-- resolved-config.json
        |-- results.jsonl
        |-- batches.jsonl           # Schema v2 batch records
        `-- tensors/batch-<edit_batch_id>.safetensors # RGB output schema v1
```

Group publication assumes one writer and keeps tensor memory bounded by batches.
Final artifacts and intermediate components have independent caches. Their schemas,
cache identities, filenames, UID ordering, and tensor layouts remain unchanged.
New editing outputs store uncompressed CPU uint8 RGB batches; sample records identify
their relative file and index. Historical PNG outputs remain untouched. See
[storage contracts](docs/artifacts.md) for compatibility, interruption,
and timing rules, and [migration guidance](docs/adapters.md#migration) for API changes.
