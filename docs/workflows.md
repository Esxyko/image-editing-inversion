# Running experiments and replaying inversions

Use `experiment` for a comparison on a dataset prefix, `diffuse` for a full run
or parameter sweep, and the Python API to replay saved inversions or archive a
finished run. All generation workflows reconstruct the source and edit the target
caption for each selected record.

Complete the [setup](../README.md#setup) and review the
[configuration guide](configuration.md) before running these examples from the
project root.

- [Small comparisons](#small-comparisons)
- [Full-dataset runs](#full-dataset-runs)
- [Sequential sweep scripts](#sequential-sweep-scripts)
- [Source selection](#select-a-source-dataset)
- [Python API](#python-api)
- [Replay](#replay-saved-inversions)
- [Archiving](#compress-a-finished-run)

## Small comparisons

Compare every discovered method on the first 10 records with one parameter file:

```powershell
uv run experiment --h-params default.yaml --nsamples 10
uv run experiment --h-params default.yaml --nsamples 10 --dataset magic-brush
uv run experiment --method ddim --method null-text --h-params default.yaml --nsamples 10
```

`experiment` defaults to all discovered methods. Repeat `--method` to select
specific methods, or use `--method ALL` to select all explicitly. Selection follows
the ordering and deduplication rules described under [full-dataset runs](#full-dataset-runs).

Both `--h-params` and `--nsamples` are required. `--h-params ALL` is rejected;
`--nsamples` must be a positive integer. The same record prefix is used for every
method, with these rules:

- Source filtering happens first, followed by selection in dataset order.
- A count larger than the available records selects all of them; an empty selection fails.
- Each MagicBrush turn counts as one record.
- Cached records count toward the limit and reuse compatible inversions.
- Publication preserves existing artifacts outside the selected prefix.

Limited runs record the requested count in the top-level `nsamples` field of
`sweep.json`.

## Full-dataset runs

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

## Sequential sweep scripts

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

## Select a source dataset

Use `--dataset pie-bench` or `--dataset magic-brush` to invert, reconstruct,
and edit only records from that source across every selected method and parameter
file. Omit it to process the entire combined dataset:

```powershell
uv run diffuse --method ddim --h-params default.yaml --dataset pie-bench
uv run diffuse --method ddim --h-params default.yaml --dataset magic-brush
```

The top-level `dataset` field in `sweep.json` records this selection, or `null`
when omitted. Source filtering retains the combined dataset's cache identity.

## Python API

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
hyperparameters to match complete saved snapshots without recomputing inversion.
Incompatible artifact/file pairs are recorded as `skipped`; malformed artifacts
and operational failures stop execution.
If every pair is skipped, replay raises an error.

## Compress a finished run

Use the Python API with one directory name under `data/output/`:

```python
from image_editing_inversion.workflows import compress_run

archive_path = compress_run("20261013T080000Z-01234567")  # Replace with an existing run ID.
# After run_methods() or edit_artifacts() returns, use compress_run(run_dir.name).
print(archive_path)
```

The API writes `data/output/<run-id>.zip` and returns its absolute path. It
preserves the `<run-id>/` top-level directory, all files (including hidden files),
and empty directories. It uses DEFLATE level 6 and ZIP64, streaming one file at a
time. Only runs whose `sweep.json` status is `ok` or `error` are accepted. Pass a
directory name, not a path; links, junctions, and special files are rejected.

After checking member names, sizes, and all CRCs, it atomically replaces any
existing same-name ZIP and deletes the original run directory. Failures before
publication preserve the source and the old ZIP. If deletion fails, the verified
ZIP remains and the exception reports its path; the source directory may have
been partially deleted. Ensure the run is inactive and no other process modifies
the run or ZIP during this call; change detection does not provide a writer lock.
There is no compression CLI or HTTP endpoint. See the
[archive storage contract](run-outputs.md#run-archives).

For output formats, loading examples, and timing fields, see
[run outputs](run-outputs.md). For optimization diagnostics, see
[Null-text diagnostics](null-text-diagnostics.md).
