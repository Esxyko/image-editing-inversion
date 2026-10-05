# Image Editing Inversion

Compare DDIM, Null-text, Direct Inversion, and ReNoise with one Stable Diffusion
1.5 runtime and Prompt-to-Prompt editor. Each dataset record produces a source
reconstruction and a target-caption edit. Inversions are saved and reused across
compatible experiments; dataset masks are reserved for later metrics.

## Setup

Use Python **3.14+** and `uv`. The checked-in configuration targets CUDA with
float16; select a supported device and precision before running.

```powershell
uv sync
```

The project dataset, `beatle-ju1ce/image-editing-inversion`, is private. Request
access, then set `HF_TOKEN` in your environment, fill the root `.env` using
[.env.example](.env.example), or use an existing Hugging Face login. The first
load downloads the train split into `data/`; later loads use that saved copy.
Allow several gigabytes for the download cache and saved dataset.

The directories containing configuration and data have fixed positions beside
`src/`. Package paths resolve from that source layout, independent of the working
directory. Run the `uv` examples from the project root; the environment's installed
`diffuse` command uses the same fixed paths when called elsewhere.

## Configuration

| Location | Settings |
| --- | --- |
| [config.yaml](config.yaml) | Model/revision/resolution, eta/seed, P2P mode, runtime, optional per-method P2P settings |
| [pipeline_h_params/](pipeline_h_params/) | Steps, guidance, cross-attention fraction, self-attention fraction |
| [method_h_params/](method_h_params/) | Inversion-only Null-text and ReNoise parameters |

Hold shared configuration, method parameters, and the dataset fixed for a
comparison. Vary pipeline parameter files to sweep sampling and editing settings.
Each pipeline YAML requires:

```yaml
sampling:
  num_inference_steps: 50
  guidance_scale: 7.5
prompt_to_prompt:
  cross_replace_fraction: 0.4
  self_replace_fraction: 0.6
```

Steps must be positive, guidance finite and nonnegative, and fractions within
`[0, 1]`. Unknown settings and misplaced pipeline settings are rejected.
P2P `auto` selects replacement for equal whitespace-word counts and refinement
otherwise. Method-specific P2P extras do not override shared mode or fractions.

Checked-in batch sizes are **2** source images for inversion and **4** image
pairs for editing. The stages are independent; an editing batch of `N` pairs
contains `N` source and `N` target branches. Omitted `inversion_batch_size`
defaults to `1`. Reduce batches for smaller GPUs; CPU offload and attention
chunking are configurable. Null-text inversion supports `none`/`model` offload;
its replay also supports sequential offload. An out-of-memory error stops the run.

Shared YAML is read once per invocation. Method parameters stay fixed for that
invocation and are reloaded on the next one. Hardware/configuration constraints
fail before model loading; complete method preflight precedes sample inversion.
See [method requirements and defaults](docs/methods.md).

## Run experiments

Invert, reconstruct, and edit every project record with one parameter file:

```powershell
uv run diffuse --method ddim --h-params default.yaml
```

Compare selected methods across every top-level pipeline YAML:

```powershell
uv run diffuse --method ddim --method null-text --method direct-inversion --method renoise --h-params ALL
```

Use `--method ALL` for all discovered adapters. Repeatable selectors expand in
command-line order, duplicates run once, and `ALL` expands IDs in sorted order.
Only uppercase `ALL` is reserved. `--h-params ALL` processes filenames in sorted
order, completing every selected method for one file before the next.

Python callers can use the same workflow:

```python
from image_editing_inversion.workflows import run_methods

run_dir = run_methods(["ddim", "null-text"], pipeline_h_params_file="default.yaml")
print(run_dir)
```

## Replay saved inversions

```python
from image_editing_inversion.workflows import edit_artifacts

run_dir = edit_artifacts()            # All published artifacts, all parameter files
# run_dir = edit_artifacts("435242:2") # Every matching artifact for one raw dataset UID
```

Replay uses registered adapters and saved state without reading inversion
hyperparameters or recomputing inversion. Incompatible artifact/file pairs are
recorded as `skipped`; malformed artifacts and operational failures stop execution.
If every pair is skipped, replay raises an error.

## Outputs and reuse

```text
data/
|-- state.json, dataset_info.json, data-*.arrow
|-- cache/inversion/                 # Per-image source encodings and conditional pivots
|-- artifacts/
|   |-- catalog.json
|   `-- <method>/steps-50_guidance-7.5/
|       |-- h-params.json
|       `-- artifacts.safetensors
`-- output/<run-id>/
    |-- sweep.json
    `-- <parameter-file>/<method>/
        |-- resolved-config.json
        |-- results.jsonl
        |-- batches.jsonl
        `-- images/<uid-with-colons-replaced-by-underscores>/
            |-- reconstructed.png
            `-- edited.png
```

`data/` is ignored by Git. Each invocation returns its parent run directory.
Results retain raw UIDs and artifact positions; image paths are relative to their
method output. Existing image directories cause an error instead of being overwritten.

Changing attention fractions or editing policies reuses compatible inversions.
Changing steps or guidance selects another artifact group. Other effective
inversion changes replace affected entries only after successful publication.
Interrupted inversion leaves the previous published group intact; editing failure
leaves completed inversions reusable. Use `batches.jsonl` for aggregate timing:
per-image batch durations repeat and must not be summed across members.

## References

- [Dataset schema and provenance](docs/dataset.md)
- [Bundled methods and hyperparameters](docs/methods.md)
- [Artifact formats, caching, replay, and timing](docs/artifacts.md)
- [Adapters, attention extensions, and API migration](docs/adapters.md)
- [Architecture and module responsibilities](structure.md)

Import public APIs from `config`, `data`, `artifacts`, `inversion`, `generation`,
and `workflows`. External adapters register zero-argument factories. A reusable
`WorkflowRunner` creates a fresh session per call; configuration, dataset/model
ownership, and execution state remain internal.
