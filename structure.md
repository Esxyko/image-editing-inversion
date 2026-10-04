# Project Structure

```text
.
|-- .env.example                # HF_TOKEN template; real .env is ignored
|-- README.md
|-- structure.md
|-- pyproject.toml
|-- config/
|   `-- experiment.yaml         # Shared settings and optional per-method P2P settings
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
|       |   |-- __init__.py      # Public adapter contracts and registry API
|       |   |-- base.py          # InversionMethod adapter interface
|       |   |-- context.py       # InversionContext passed to adapters
|       |   |-- hooks.py         # Denoising state and hook contracts
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
- `pyproject.toml` defines the project package, console entry point, and shared
  Python dependencies.
- `config/experiment.yaml` is the checked-in example of common pipeline
  settings and hardware-limited runtime values.
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
  entry-point discovery. Adapters can select a Prompt-to-Prompt subclass.
- `image_editing_inversion.editing` owns model loading, the subclassable
  Prompt-to-Prompt attention policy, and shared reconstruction/editing.
  `ModelRuntime` owns pipeline setup; `Editor` owns denoising and hook execution.
  Attention alignment, policy, and Diffusers integration have separate files.
- `image_editing_inversion.experiments` composes the other modules.
  `ExperimentRunner` supplies configuration and samples to adapters and the
  editor and rejects incompatible batches. `RunOutput` owns run directories,
  configuration snapshots, artifact/image paths, image writes, and run records.
- `data/` is not tracked by Git. Its root holds the saved Hub dataset after the
  first load.

## Dependency boundaries

- `config`, `artifacts`, and `data` do not depend on methods, editing, or
  experiment orchestration.
- `methods` depends on configuration and artifacts. Editor and attention policy
  types are imported only under `TYPE_CHECKING`; discovery does not import
  experiments or the model runtime.
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
