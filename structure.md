# Project Structure

```text
.
|-- .env.example                # HF_TOKEN template; real .env is ignored
|-- README.md
|-- structure.md
|-- pyproject.toml
|-- src/
|   `-- image_editing_inversion/
|       |-- __init__.py
|       `-- dataset.py           # Fetch and load the project dataset
`-- data/                       # Local artifacts; ignored by Git
    |-- state.json              # Saved Hub dataset metadata after first load
    |-- dataset_info.json
    |-- data-*.arrow            # Saved Hub dataset records
    |-- raw/
    |-- cache/
    |-- processed/
    `-- final/
        `-- image-editing/      # Original locally built artifact
```

## Responsibilities

- `README.md` describes how to load the Hub dataset and how the combined
  dataset was constructed.
- `pyproject.toml` defines the project package, console entry point, and shared
  Python dependencies.
- `image_editing_inversion.__init__` provides the existing console entry point.
- `image_editing_inversion.dataset` downloads the private Hub train split on
  first use and loads the saved copy from `data/` afterward.
- `data/` is not tracked by Git. Its root holds the saved Hub dataset, while
  its existing subfolders hold other project artifacts.
