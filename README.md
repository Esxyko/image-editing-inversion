# Image Editing Inversion

Compare **DDIM, Null-text, Direct Inversion, and ReNoise** using one Stable
Diffusion 1.5 runtime and a shared Prompt-to-Prompt editor. Each dataset record
produces a source reconstruction and a target-caption edit, saved as RGB pixel
tensors. Compatible inversions are cached and can be replayed; archived runs can
be evaluated against dataset captions and edit masks.

## Setup

Use **Python 3.14+** and **uv**. Install dependencies from the project root:

```powershell
uv sync
```

The dataset, `beatle-ju1ce/image-editing-inversion`, is private. Obtain access,
then set `HF_TOKEN` in your environment, create a root `.env` from
[.env.example](.env.example), or use an existing Hugging Face login. The first
load saves the Hub `eval` split under `data/`; later runs reuse the local copy.
Allow several gigabytes for the download cache and saved dataset.

The checked-in runtime uses CUDA and float16. Review [config.yaml](config.yaml)
and the [configuration guide](docs/configuration.md) for device, precision, and
batch sizes before running. Configuration and storage paths are fixed relative
to the source layout; run these `uv` examples from the project root.

## Run a comparison

Start with all discovered methods on the first 10 eval records:

```powershell
uv run experiment --h-params default.yaml --nsamples 10
```

Select specific methods and a source dataset when needed:

```powershell
uv run experiment --method ddim --method null-text --h-params default.yaml --nsamples 10 --dataset magic-brush
```

For a full-dataset run or a sweep across all pipeline parameter files:

```powershell
uv run diffuse --method ddim --h-params default.yaml
uv run diffuse --method ALL --h-params ALL
```

Both commands use the combined `eval` split: 1,400 records from PIE-Bench and
MagicBrush. `--dataset pie-bench` or `--dataset magic-brush` filters that split
before any sample limit. `experiment` requires one parameter filename and a
positive sample count; `diffuse` also supports `--h-params ALL`.

See [workflows](docs/workflows.md) for selection rules, sequential sweep scripts,
Python APIs, replay, and archiving. See the [dataset reference](docs/dataset.md)
for schema, provenance, and the separate `dev` split.

## Outputs and reuse

Each invocation writes `data/output/<run-id>/`, with a sweep manifest and one
`<parameter-file>/<method>/` directory per comparison. Method directories contain
configuration snapshots, sample results, batch timings, and uncompressed
`reconstructed` / `edited` uint8 RGB tensors in `[N, 3, height, width]` layout.

Two 512 ? 512 results use 1.5 MiB per sample, or about 8.2 GiB for 1,400 samples
across four methods. See [run outputs](docs/run-outputs.md) for the directory
layout, tensor-loading examples, and timing definitions.

Inversions live separately under `data/artifacts/`. Changing only attention
fractions or editing policies reuses compatible inversions. Changing method
parameters or editing guidance invalidates reuse; changing steps or inversion
guidance selects another artifact group. See
[artifact storage and compatibility](docs/artifacts.md).

## Evaluate archived runs

Archive a finished run with the Python
[`compress_run` API](docs/workflows.md#compress-a-finished-run). It verifies the
ZIP, publishes it under `data/output/`, then deletes the original run directory.
Install metric dependencies and evaluate one archive or all available ZIPs:

```powershell
uv sync --extra evaluation
uv run --extra evaluation evaluate data/output/<run-id>.zip --device cuda --batch-size 16
uv run --extra evaluation evaluate
```

Replace `<run-id>` with an existing archive name. Omitting paths selects all
`data/output/*.zip`; explicit paths resolve from the current working directory.
The evaluator reads archived batches without extracting whole runs. It downloads
pinned metric weights on first use and caches them in `data/cache/evaluation/`.
CPU and MPS are also supported; metrics use float32.

Each ZIP produces one JSON report under `data/metrics/<UTC-time>-<unique-id>/`.
Reports include full-image and edited-region CLIP, masked PSNR/SSIM/LPIPS, and
batch-amortized inversion/editing core time. Results are grouped by parameter
file, method, and source (`all`, `pie-bench`, `magic-brush`). Cached inversion
time is zero for the evaluated invocation; replay-only inversion time is unknown.
See [evaluation definitions and JSON layout](docs/evaluation.md).

### Summarize existing metrics

Create Markdown and CSV summaries without loading datasets or models:

```powershell
uv run python -m image_editing_inversion.evaluation.summary
uv run python -m image_editing_inversion.evaluation.summary data/metrics/<report-directory>
```

Omitted inputs recursively select JSON reports under `data/metrics/`, excluding
`summaries/`. Each invocation writes `report.md` and `summary.csv` under
`data/metrics/summaries/<UTC-time>-<unique-id>/`. Reports retain quality statistics,
core timing, cache counts, experiment settings, and provenance for each run.
The summarizer uses only the standard library and keeps different runs separate.
See [summary format and limitations](docs/evaluation.md#summarizing-existing-reports).

## Documentation

| Guide | Covers |
| --- | --- |
| [Configuration](docs/configuration.md) | Shared settings, pipeline presets, guidance, runtime, and paths |
| [Workflows](docs/workflows.md) | Experiments, full runs, sweeps, Python API, replay, and compression |
| [Dataset](docs/dataset.md) | Record schema, loading, source selection, and split construction |
| [Methods](docs/methods.md) | Algorithms, requirements, and method-specific hyperparameters |
| [Artifacts](docs/artifacts.md) | Inversion collections, cache identity, publication, and replay compatibility |
| [Run outputs](docs/run-outputs.md) | RGB tensors, result records, batch timing, and archive format |
| [Evaluation](docs/evaluation.md) | Metric definitions, archived reports, and Markdown/CSV summaries |
| [Null-text diagnostics](docs/null-text-diagnostics.md) | Optimization observations and replay comparisons |
| [Direct diagnostics](docs/direct-reconstruction-tests.md) | Opt-in reconstruction checks and controlled comparisons |
| [Adapter extensions](docs/adapters.md) | Custom methods, denoising hooks, attention policies, and API migration |
| [Project structure](structure.md) | Module responsibilities, dependency boundaries, and execution lifecycle |
