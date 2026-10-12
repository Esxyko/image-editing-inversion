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
| [pipeline_h_params/](pipeline_h_params/) | Steps, guidance, cross-attention fraction, self-attention fraction |
| [method_h_params/](method_h_params/) | Inversion-only Null-text and ReNoise parameters |

Hold shared configuration, method parameters, and the dataset fixed for a
comparison. Vary pipeline parameter files to sweep sampling and editing settings.
Each pipeline YAML requires:

```yaml
sampling:
  num_inference_steps: 30
  guidance_scale: 7.5
prompt_to_prompt:
  cross_replace_fraction: 0.4
  self_replace_fraction: 0.6
```

Steps must be positive, guidance finite and nonnegative, and fractions within
`[0, 1]`. Unknown settings and misplaced pipeline settings are rejected.
P2P `auto` selects replacement for equal whitespace-word counts and refinement
otherwise. Method-specific P2P extras do not override shared mode or fractions.

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
```

`experiment` always selects all registered methods and accepts no `--method`
argument. Both `--h-params` and `--nsamples` are required; `--h-params ALL` is
rejected and the sample count must be a positive integer. Source filtering, when
specified, occurs before taking the first records in dataset order. Every method
inverts, reconstructs, and edits the same prefix. A count larger than the available
records processes every available record; an empty selection fails. MagicBrush
turns count as individual records. Cached records count toward the limit and reuse
compatible inversions; existing artifacts outside the prefix remain published.
Limited runs record the requested count in the top-level `nsamples` field of
`sweep.json`.

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
