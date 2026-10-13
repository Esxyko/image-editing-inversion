# Image Editing Inversion

Compare DDIM, Null-text, Direct Inversion, and ReNoise with one Stable Diffusion
1.5 runtime and Prompt-to-Prompt editor. Each dataset record produces a source
reconstruction and a target-caption edit as RGB pixel tensors. Inversions are saved and reused across
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
load downloads the eval split into `data/`; later loads use that saved copy.
Both `diffuse` and `experiment` use `eval` by default. Their `--dataset`
argument filters source datasets within `eval`.
The dev split contains the MagicBrush records excluded by the edit-turn rules
in [the dataset reference](docs/dataset.md#eval-and-dev-splits).
Allow several gigabytes for the download cache and saved dataset.

The directories containing configuration and data have fixed positions beside
`src/`. Package paths resolve from that source layout, independent of the working
directory. Run the `uv` examples from the project root; the environment's installed
`diffuse` and `experiment` commands use the same fixed paths when called elsewhere.

## Configuration

| Location | Settings |
| --- | --- |
| [config.yaml](config.yaml) | Model/revision/resolution, eta/seed, P2P mode, runtime, optional per-method P2P settings |
| [pipeline_h_params/](pipeline_h_params/) | Steps, shared editing CFG, cross-attention fraction, self-attention fraction |
| [method_h_params/](method_h_params/) | Per-method inversion CFG and algorithm parameters |

Hold shared configuration, method parameters, and the dataset fixed for a
comparison. Vary pipeline parameter files to sweep sampling and editing settings.
Each pipeline YAML requires:

```yaml
sampling:
  num_inference_steps: 30
  editing_guidance_scale: 7.5
prompt_to_prompt:
  cross_replace_fraction: 0.4
  self_replace_fraction: 0.6
```

Steps must be positive and fractions within `[0, 1]`.
Unknown settings and misplaced pipeline settings are rejected, including the old
pipeline `guidance_scale` field. Each method YAML includes:

```yaml
inversion_guidance_scale: 7.5
```

Both CFG values must be finite and nonnegative; Null-text requires both to be
at least `1`. Editing CFG is shared through pipeline `sampling.editing_guidance_scale`;
it is rejected in method YAML and method snapshots. Null-text optimizes its
empty-caption embeddings at editing CFG
and uses that same CFG for source reconstruction and target editing. Its
DDIM pivots use inversion CFG. When editing CFG is `1`, Adam updates
are skipped because their gradients are zero. Source reconstruction uses editing
CFG for every bundled method. All target
branches use editing CFG. DDIM and Direct read `DDIM.yaml` and `Direct.yaml`; Null-text and
ReNoise retain their existing method files. Method parameters are loaded for both
inversion and replay. Cache reuse and replay require the complete current method
parameters and pipeline editing CFG to match the saved provenance. Historical
artifacts without the method snapshot or separate editing CFG require regeneration.
Older Null-text artifacts without
inversion-CFG pivot state also require regeneration.
P2P `auto` selects replacement for equal whitespace-word counts and refinement
otherwise. Method-specific P2P extras do not override shared mode or fractions.

The seven existing pipeline presets use editing CFG `7.5`.
[`high_editing_guidance.yaml`](pipeline_h_params/high_editing_guidance.yaml) uses `10.0` and
[`low_editing_guidance.yaml`](pipeline_h_params/low_editing_guidance.yaml) uses `5.0`;
both retain `default.yaml`'s 15 steps and attention fractions `0.4`/`0.4`.
Select either with `--h-params`, or include both through `--h-params ALL`.

Checked-in batch sizes are **8** source images for inversion and **32** RGB tensor
pairs for editing, with **4** threads for dataset/artifact reads. These are
starting settings for a 32 GB VRAM GPU at 512x512; tune batch sizes using peak
memory and images per second, including Null-text backward passes.
The stages are independent; an editing batch of `N` pairs contains `N` source
and `N` target branches, doubled to `4N` UNet branches for classifier-free
guidance. Omitted `inversion_batch_size`
defaults to `1`. Reduce batches for smaller GPUs; CPU offload and attention
chunking are configurable. Null-text inversion supports `none`/`model` offload;
its replay also supports sequential offload. An out-of-memory error stops the run.

Shared YAML is read once per invocation. Method parameters stay fixed for that
invocation and are reloaded on the next one. Hardware/configuration constraints
fail before model loading; complete method preflight precedes sample inversion.
See [method requirements and defaults](docs/methods.md).

## Run experiments

Compare every discovered method on the first 10 records with one parameter file:

```powershell
uv run experiment --h-params default.yaml --nsamples 10
uv run experiment --h-params default.yaml --nsamples 10 --dataset magic-brush
uv run experiment --method ddim --method null-text --h-params default.yaml --nsamples 10
```

`experiment` defaults to all discovered methods. Use the repeatable `--method`
argument to select methods, or `--method ALL` to explicitly select all methods.
Selectors use the same ordering and deduplication rules as `diffuse` below.
Both `--h-params` and `--nsamples` are required; `--h-params ALL` is
rejected and the sample count must be a positive integer. Source filtering, when
specified, occurs before taking the first records in dataset order. Every method
inverts, reconstructs, and edits the same prefix. A count larger than the available
records processes every available record; an empty selection fails. MagicBrush
turns count as individual records. Cached records count toward the limit and reuse
compatible inversions; existing artifacts outside the prefix remain published.
Limited runs record the requested count in the top-level `nsamples` field of
`sweep.json`.

For a temporary Null-text diagnostic run with the same sample selection, model,
hyperparameters, inversion/edit batch sizes, and RGB outputs:

```powershell
uv run experiment-diagnostic --method null-text --h-params high_steps.yaml --nsamples 64
```

This command supports only `null-text`; omitting `--method` selects it. The other
experiment arguments, including `--dataset`, have the same meanings. It always
recomputes inversion, source encodings, and conditional pivots. It publishes the
selected prefix into the ordinary Null-text artifact group, as a fresh experiment
does, but does not register these diagnostic inversions as cache hits for later runs.

Under `data/output/<run-id>/high_steps.yaml/null-text/`, the ordinary outputs are
accompanied by `optimization.jsonl` (one sample/timestep per line),
`reconstruction.jsonl` (final reconstruction diagnostics per sample), and
`diagnostic-config.json` (actual schedules, scheduler configurations, versions,
and source hashes). Losses, gradient/update norms, and the fraction of embedding
elements changed after conversion to the model dtype are collected from the
existing evaluations; no extra UNet calls are made. Recording adds runtime overhead,
so use ordinary experiments to compare execution times. See
[diagnostic field definitions](docs/artifacts.md#temporary-null-text-diagnostics).

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

The three Python scripts in `scripts/` split the nine pipeline configurations
into fixed groups. Each runs every discovered method on the entire combined eval
dataset, completing one configuration before starting the next:

| Script | Configuration order |
| --- | --- |
| `scripts/1.py` | `default.yaml`, `high_steps.yaml`, `low_steps.yaml` |
| `scripts/2.py` | `high_editing_guidance.yaml`, `low_editing_guidance.yaml`, `high_cross_replace.yaml` |
| `scripts/3.py` | `low_cross_replace.yaml`, `high_self_replace.yaml`, `low_self_replace.yaml` |

Run each script separately from the project root, waiting for it to finish before
starting the next:

```powershell
uv run python scripts/1.py
uv run python scripts/2.py
uv run python scripts/3.py
```

Scripts locate the project root relative to their own files and invoke
`uv run diffuse --method ALL --h-params <filename>` there. Diffusion output remains
visible in the terminal. Each completed configuration is logged immediately to
the terminal and `data/output/script-logs/<script-number>.log`, with a timestamp,
filename, progress, and elapsed seconds. Logs use UTF-8 and append on subsequent
runs. A failed subprocess, launch error, or user interruption is logged and stops
the current script with a nonzero exit code (130 for user interruption). Rerunning
a script starts its configuration list again; there is no automatic retry or
resume.

Use `--dataset pie-bench` or `--dataset magic-brush` to invert, reconstruct,
and edit only records from that source across every selected method and parameter
file. Omit it to process the entire combined dataset:

```powershell
uv run diffuse --method ddim --h-params default.yaml --dataset pie-bench
uv run diffuse --method ddim --h-params default.yaml --dataset magic-brush
```

The top-level `dataset` field in `sweep.json` records this selection, or `null`
when omitted. Source filtering retains the combined dataset's cache identity.

Python callers can use the same workflow:

```python
from image_editing_inversion.workflows import run_methods

run_dir = run_methods(["ddim", "null-text"], pipeline_h_params_file="default.yaml")
print(run_dir)
```

Both `run_methods` and `WorkflowRunner.run_methods` accept the optional keyword
`dataset="pie-bench"` or `dataset="magic-brush"`; `None` selects every record.
Unsupported values raise `ValueError` before an execution session is created.

```python
run_dir = run_methods(["ddim"], pipeline_h_params_file="default.yaml", dataset="magic-brush")
```

Both APIs also accept `nsamples=10` to limit processing to the first 10 records
after source filtering, or `None` (the default) to process every selected record.
Booleans, non-integer values, and nonpositive counts raise `ValueError` before an
execution session is created. The Python equivalent of `experiment` is:

```python
run_dir = run_methods(["ALL"], pipeline_h_params_file="default.yaml", nsamples=10)
```

## Replay saved inversions

```python
from image_editing_inversion.workflows import edit_artifacts

run_dir = edit_artifacts()            # All published artifacts, all parameter files
# run_dir = edit_artifacts("435242:2") # Every matching artifact for one raw dataset UID
```

Replay uses registered adapters and saved state, loading current method
hyperparameters for full snapshot matching without recomputing inversion. Incompatible artifact/file pairs are
recorded as `skipped`; malformed artifacts and operational failures stop execution.
If every pair is skipped, replay raises an error.

## Outputs and reuse

```text
data/
|-- state.json, dataset_info.json, data-*.arrow
|-- cache/inversion/                 # Per-image source encodings and conditional pivots
|-- artifacts/
|   |-- catalog.json
|   `-- <method>/steps-30_guidance-7.5/
|       |-- h-params.json
|       `-- artifacts.safetensors
`-- output/<run-id>/
    |-- sweep.json
    `-- <parameter-file>/<method>/
        |-- resolved-config.json
        |-- results.jsonl
        |-- batches.jsonl
        `-- tensors/
            `-- batch-<edit_batch_id>.safetensors
```

New files under `data/` are ignored by Git; previously tracked history remains.
Each invocation returns its parent run directory. Results retain raw UIDs and
artifact positions, plus a relative `tensor_file` and zero-based `tensor_index`.
Each editing batch atomically publishes uncompressed `reconstructed` and `edited`
uint8 tensors in RGB `[N, 3, height, width]` layout. New runs save no PNG files;
VAE decoding and pixel quantization still run. Historical PNG outputs are retained.
At 512x512, two results occupy 1.5 MiB per sample; a full 1,400-sample, four-method
run stores about 8.2 GiB of pixels. See [tensor loading examples](docs/artifacts.md#rgb-tensor-outputs).

Changing attention fractions or editing policies reuses compatible inversions.
Changing steps or inversion CFG selects another artifact group. Changing any
method hyperparameter or pipeline editing CFG invalidates reuse; affected entries
are replaced only after successful publication.
Interrupted inversion leaves the previous published group intact; editing failure
leaves completed inversions reusable. New artifacts embed metadata and tensor checksums,
so valid published files remain recoverable when catalog updates are interrupted.
Unverifiable existing groups are preserved and block replacement; legacy replay
requires a matching catalog checksum. Use `batches.jsonl` for aggregate timing:
per-image batch durations repeat and must not be summed across members.

## References

- [Dataset schema and provenance](docs/dataset.md)
- [Bundled methods and hyperparameters](docs/methods.md)
- [Artifact formats, caching, replay, and timing](docs/artifacts.md)
- [Direct reconstruction diagnostic tests](docs/direct-reconstruction-tests.md)
- [Adapters, attention extensions, and API migration](docs/adapters.md)
- [Architecture and module responsibilities](structure.md)

Import public APIs from `config`, `data`, `artifacts`, `inversion`, `generation`,
and `workflows`. External adapters register zero-argument factories. A reusable
`WorkflowRunner` creates a fresh session per call; configuration, dataset/model
ownership, and execution state remain internal.
