# Project Structure

```text
.
|-- .env.example                # HF_TOKEN template; real .env is ignored
|-- README.md
|-- structure.md
|-- pyproject.toml
|-- config/
|   `-- experiment.yaml         # Shared settings and optional per-method P2P settings
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
|           `-- output.py        # RunOutput paths, snapshots, images, and records
`-- data/                       # Local artifacts; ignored by Git
    |-- state.json              # Saved Hub dataset metadata after first load
    |-- dataset_info.json       # Created after first load
    `-- data-*.arrow            # Saved Hub dataset records after first load
```

## Responsibilities

- `README.md` describes dataset loading, artifact editing, method adapters,
  and how the combined dataset was constructed.
- `pyproject.toml` defines the project package, console entry point, bundled
  DDIM, Null-text, Direct Inversion, and ReNoise method entry points, and shared Python
  dependencies.
- `config/experiment.yaml` is the checked-in example of common pipeline
  settings and hardware-limited runtime values.
- `method_h_params/Null_text.yaml` supplies Null-text optimization settings,
  loaded relative to the working directory and cached by the adapter on its
  first inversion. Discovery and artifact replay do not read this file.
- `method_h_params/ReNoise.yaml` supplies ReNoise refinement counts and prediction
  averaging windows, loaded relative to the working directory and cached by the
  adapter on its first inversion. Discovery and artifact replay do not read it.
- `image_editing_inversion.__init__` exports the console entry point.
- `image_editing_inversion.cli` parses `edit-artifact` and `run` commands.
- `image_editing_inversion.config` loads and validates shared settings and
  optional per-method Prompt-to-Prompt settings from one YAML configuration.
- `image_editing_inversion.data` downloads the private Hub train split on first
  use and loads the saved copy from `data/` afterward. `DatasetRepository`
  selects a Hub or local dataset, validates its schema and identity, and resolves
  samples by UID. Dataset references and expected columns live in `schema.py`.
- `image_editing_inversion.artifacts` owns the portable inversion artifact
  format, compatibility checks, and JSON/safetensors serialization.
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
  Adapters can validate method-specific replay compatibility
  and select a Prompt-to-Prompt subclass.
- `image_editing_inversion.editing` owns model loading, the subclassable
  Prompt-to-Prompt attention policy, and shared reconstruction/editing.
  `ModelRuntime` owns pipeline setup; `Editor` owns shared prompt encoding,
  denoising, and hook execution.
  Attention alignment, policy, and Diffusers integration have separate files.
- `image_editing_inversion.experiments` composes the other modules.
  `ExperimentRunner` supplies configuration and samples to adapters and the
  editor and validates method-specific replay state before creating hooks.
  It rejects incompatible batches. `RunOutput` owns run directories,
  configuration snapshots, artifact/image paths, image writes, and run records.
- `data/` is not tracked by Git. Its root holds the saved Hub dataset after the
  first load.

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
entry-point group, YAML format, artifact schema, and run formats are unchanged.
