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
|       |-- artifact.py          # Inversion artifacts, hooks, and method registry
|       |-- cli.py               # Command-line parsing
|       |-- config.py            # YAML configuration loading and validation
|       |-- dataset.py           # Fetch and load the project dataset
|       |-- editing.py           # SD 1.5 Prompt-to-Prompt editing backbone
|       `-- runner.py            # Dataset selection, batching, and run outputs
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
- `image_editing_inversion.dataset` downloads the private Hub train split on
  first use and loads the saved copy from `data/` afterward.
- `image_editing_inversion.artifact` owns the portable inversion artifact format,
  compatibility checks, per-step hook interface, method registry, and method
  selection of a Prompt-to-Prompt subclass.
- `image_editing_inversion.editing` owns model loading, the subclassable
  Prompt-to-Prompt attention policy, and shared reconstruction/editing.
- `image_editing_inversion.runner` selects dataset records, supplies the shared
  configuration to adapters and the editor, rejects incompatible editing
  batches, and writes images and compact run records.
- `data/` is not tracked by Git. Its root holds the saved Hub dataset after the
  first load.
