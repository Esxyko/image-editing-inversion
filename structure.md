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
|       |-- __init__.py          # Console entry point
|       |-- cli.py               # Command-line parsing
|       |-- config/
|       |   |-- __init__.py      # Public configuration API
|       |   |-- models.py        # Settings dataclasses and hardware validation
|       |   `-- loader.py        # YAML parsing and validation
|       |-- data/
|       |   |-- __init__.py      # Public dataset API
|       |   |-- schema.py        # Shared dataset references and expected columns
|       |   |-- loader.py        # Fetch and load the project dataset
|       |   `-- repository.py    # Dataset selection, identity, and UID lookup
|       |-- artifacts/
|       |   |-- __init__.py      # Public artifact API
|       |   |-- catalog.py       # Shared artifact storage, lookup, integrity, and atomic catalog
|       |   `-- inversion.py     # Artifact validation and serialization
|       |-- methods/
|       |   |-- __init__.py      # Public methods, adapter contracts, and registry API
|       |   |-- base.py          # InversionMethod adapter interface
|       |   |-- context.py       # InversionContext passed to adapters
|       |   |-- ddim.py          # Deterministic DDIM inversion baseline
|       |   |-- direct.py        # Conditional pivots, latent residuals, and source-only replay hook
|       |   |-- hooks.py         # Denoising state and hook contracts
|       |   |-- null_text.py     # Pivotal inversion, null-text optimization, and replay hook
|       |   |-- renoise.py       # Iterative DDIM noising, prediction averaging, and replay validation
|       |   `-- registry.py      # Registration and entry-point discovery
|       |-- editing/
|       |   |-- __init__.py      # Public editing API; imports Editor on demand
|       |   |-- runtime.py       # ModelRuntime pipeline/device/offload setup
|       |   |-- alignment.py     # Prompt token mapping and PairAttentionMap
|       |   |-- prompt_to_prompt.py # Subclassable attention policy
|       |   |-- processor.py     # Diffusers attention integration
|       |   |-- editor.py        # Shared reconstruction and DDIM editing loop
|       |   `-- result.py        # EditResult image pair
|       `-- experiments/
|           |-- __init__.py      # Public experiment API
|           |-- runner.py        # ExperimentRunner batching and orchestration
|           `-- output.py        # SweepOutput manifest and RunOutput snapshots/images/records
`-- data/                       # Local artifacts; ignored by Git
    |-- state.json              # Saved Hub dataset metadata after first load
    |-- dataset_info.json       # Created after first load
    |-- data-*.arrow            # Saved Hub dataset records after first load
    |-- artifacts/             # Generated inversions, independent of run output
    |   |-- catalog.json       # Versioned index of reusable inversions
    |   `-- <unique-id>/        # artifact.json and tensors.safetensors
    `-- runs/                  # Example output root; configurable with --output
        `-- <run-id>/           # sweep.json and one child per pipeline parameter filename
```

## Responsibilities

- `README.md` describes dataset loading, artifact editing, method adapters,
  and how the combined dataset was constructed.
- `pyproject.toml` defines the project package, console entry point, bundled
  DDIM, Null-text, Direct Inversion, and ReNoise method entry points, and shared Python
  dependencies.
- `config.yaml` supplies model settings, eta, seed, editing mode, and runtime
  values; it is loaded from the working directory without a filename argument.
- `pipeline_h_params/` supplies grouped YAML files containing sampling step count,
  guidance scale, and the two attention replacement fractions. Both commands
  select a filename or process all top-level YAML files sequentially.
- `method_h_params/Null_text.yaml` supplies Null-text optimization settings,
  loaded relative to the working directory and cached by the adapter on its
  first inversion or cache lookup. Discovery and artifact replay do not read this file.
- `method_h_params/ReNoise.yaml` supplies ReNoise refinement counts and prediction
  averaging windows, loaded relative to the working directory and cached by the
  adapter on its first inversion or cache lookup. Discovery and artifact replay do not read it.
- `image_editing_inversion.__init__` exports the console entry point.
- `image_editing_inversion.cli` parses `edit-artifact` and `run` commands.
- `image_editing_inversion.config` loads and validates shared settings and
  optional per-method Prompt-to-Prompt settings from `config.yaml`, discovers
  pipeline parameter files, and resolves one combined `ExperimentConfig` per file.
- `image_editing_inversion.data` downloads the private Hub train split on first
  use and loads the saved copy from `data/` afterward. `DatasetRepository`
  selects a Hub or local dataset, validates its schema and identity, and resolves
  samples by UID. Dataset references and expected columns live in `schema.py`.
- `image_editing_inversion.artifacts` owns the portable inversion artifact
  format, compatibility checks, and JSON/safetensors serialization.
  `ArtifactCatalog` owns central storage and a versioned JSON index with input
  digests and file checksums. It publishes complete artifacts before atomically
  updating the catalog and assumes one active writer. Compatibility mismatches
  use `ArtifactCompatibilityError`; malformed state remains an ordinary error.
- `image_editing_inversion.methods` owns the inversion adapter interface,
  `InversionContext`, denoising state and hooks, and method registration and
  entry-point discovery. Its `ddim` submodule implements the first bundled
  inversion algorithm using the shared pipeline and sampling guidance, with
  a separate inverse scheduler. Its `null_text` submodule builds conditional-only
  DDIM pivots and optimizes per-step unconditional embeddings, then supplies
  their replay hook. Its `direct` submodule builds conditional-only DDIM pivots,
  records per-step latent residuals at the sampling guidance, and supplies a
  source-only after-step replay hook without optimization or extra settings.
  Its `renoise` submodule iteratively reverses the editor's deterministic DDIM
  updates using fixed previous latents and prediction averaging. It records
  sampling guidance and supplies a validating hook that leaves denoising state
  unchanged. It does not implement regularization or stochastic noise correction.
  Adapters can validate method-specific replay compatibility, select a
  Prompt-to-Prompt subclass, and opt into reuse through
  `inversion_cache_parameters`. External adapters disable caching by default.
- `image_editing_inversion.editing` owns model loading, the subclassable
  Prompt-to-Prompt attention policy, and shared reconstruction/editing.
  `ModelRuntime` owns pipeline setup, numerical cache settings, and scheduler
  reconfiguration without reloading weights; `Editor` owns shared prompt encoding,
  denoising, hook execution, and synchronized configuration updates.
  Attention alignment, policy, and Diffusers integration have separate files.
- `image_editing_inversion.experiments` composes the other modules.
  `ExperimentRunner` supplies configuration and samples to adapters and the
  editor, builds inversion cache identities, and validates replay state before
  creating hooks. It shares the dataset/model across sequential parameter files,
  reuses catalog matches, skips incompatible explicit replay pairs, and rejects
  incompatible batches. `SweepOutput` owns the parent directory and manifest;
  `RunOutput` owns each filename child, resolved snapshots, images, and records.
- `data/` is not tracked by Git. Its root holds the saved Hub dataset after the
  first load; `artifacts/` holds generated inversions and their catalog, while
  `runs/` is the example root for experiment outputs.

## Dependency boundaries

- `config`, `artifacts`, and `data` do not depend on methods, editing, or
  experiment orchestration.
- `methods` depends on configuration and artifacts. Editor and attention policy
  types are imported only under `TYPE_CHECKING`; discovery does not import
  experiments or the model runtime. Concrete methods use the editor supplied
  by `InversionContext`; all bundled methods import Diffusers only when
  inversion executes.
- `editing` depends on configuration, artifacts, and denoising hook contracts.
  Importing its attention policy does not import Diffusers; `Editor` is exported
  lazily and model weights are loaded only when an editor is constructed.
- `experiments` coordinates the lower-level modules. Its output component uses
  configuration and image contracts without importing the editor or runner.
- `cli.py` imports experiments only when executing a parsed command.

Each subpackage exports its public API through `__init__.py`. Implementation
helpers remain private. The old flat `artifact.py`, `dataset.py`, and `runner.py`
paths have no compatibility wrappers. The console entry point, plugin
entry-point group and portable artifact schema remain unchanged. Pipeline
parameters now live in separate YAML files; run outputs use filename children
and a parent sweep manifest. `load_config` requires a pipeline parameter file.
