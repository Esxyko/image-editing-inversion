# Archived-run evaluation

Evaluate successful RGB run archives without generating images or extracting
whole ZIP files. The evaluator writes detailed JSON reports; the independent
summarizer turns existing reports into Markdown and CSV tables.

- [Run evaluation](#run-evaluation)
- [Inputs and identity](#inputs-and-identity)
- [Metric definitions](#metric-definitions)
- [Two-stage core timing](#two-stage-core-timing)
- [JSON layout and publication](#json-layout-and-publication)
- [Summarizing existing reports](#summarizing-existing-reports)

## Run evaluation

```powershell
uv sync --extra evaluation
uv run --extra evaluation evaluate
uv run --extra evaluation evaluate data/output/<run-id>.zip --device cuda --batch-size 16
```

Replace `<run-id>` with an existing archive name. Omitting ZIP paths selects all
archives directly under `data/output/`. CPU and MPS are also supported; reduce
`--batch-size` for smaller devices.

`evaluate_archives(archives=None, *, device="cuda", batch_size=16) -> Path` is
exported by `image_editing_inversion.evaluation`. The `evaluate` CLI has the same
selection and settings. Install with `uv sync --extra evaluation`.

```python
from image_editing_inversion.evaluation import evaluate_archives

report_dir = evaluate_archives(["data/output/<run-id>.zip"], device="cuda", batch_size=16)
print(report_dir)
```

Omitted inputs select the ZIP files directly inside the fixed `data/output`
directory. Explicit inputs are file paths resolved from the caller's working
directory. Duplicate paths run once; distinct archives sharing a basename are
rejected because their report filenames would collide. The returned directory is
`data/metrics/<UTC-time>-<unique-id>/`, independent of the working directory.

## Inputs and identity

The evaluator accepts successful archived runs with `sweep.json`, method-level
`resolved-config.json`, `results.jsonl`, `batches.jsonl`, and schema-v1 RGB
safetensors output. It does not support historical PNG outputs or partial/failed
runs. The eval dataset is loaded through the existing `DatasetRepository`.

Manifest, result, configuration, and timing identities must agree. Every result
must reference a unique known UID with the local dataset fingerprint; methods
under one parameter file must use identical UID sequences. Limited and
source-filtered runs are allowed: the evaluator does not require all 1,400 UIDs.
Tensor metadata must match its method, parameter file, edit batch ID, RGB layout,
dimensions, and UID/index mapping. Duplicate archive members, unsafe member paths,
missing members, and inconsistent batch associations fail explicitly.

Archives are read without extraction. Only the current saved tensor batch and a
bounded evaluation batch are decoded at once. Dataset and model resources live
for the invocation; projected target-text features are reused across methods and
archives. The first invocation downloads models to `data/cache/evaluation`.
Evaluation does not depend on local `archive/` scripts, current experiment YAML
values, or the generation model. It preserves the original ZIPs and existing
metric reports.

## Metric definitions

All CLIP scores are raw cosine similarities, without multiplication by 100 or
CLIPScore scaling. The model is `openai/clip-vit-large-patch14`, revision
`32bd64288804d66eefd0ccbe215aa642df71cc41`, with 224px model preprocessing:
bicubic resize, center crop, RGB normalization. Target text is `target_prompt`,
truncated at 77 tokens; each observation records whether truncation occurred.
Models and metric arithmetic use float32, with TF32 disabled during evaluation.

| JSON metric | Definition | Direction |
| --- | --- | --- |
| `clip_text_similarity` | Full edited image against target text | Higher |
| `edited_region_clip` | Keep pixels where `mask_img==255`, set other pixels to black, then apply standard full-canvas CLIP preprocessing against target text | Higher |
| `masked_psnr_db` | Set edited regions to black in both edited and source RGB images; `-10*log10(mean RGB squared error)` over the full canvas, range `[0,1]` | Higher |
| `masked_ssim` | Same masked images; torchmetrics functional SSIM with Gaussian 11x11 window, sigma 1.5, k1 0.01, k2 0.03, data range 1 | Higher |
| `masked_lpips` | Same masked images mapped from `[0,1]` to `[-1,1]`; official LPIPS v0.1, SqueezeNet 1.1 ImageNet V1, spatial=False | Lower |

The mask must be binary (0 or 255). Masked background scores follow the
PIE-Bench black-out convention, with no background-area normalization. Region
CLIP does not crop the mask's bounding box. Empty backgrounds have null masked
scores; empty edit masks have null region CLIP. Full-image CLIP still applies.
An exact masked pixel match has infinite PSNR, represented as the string `"+inf"`
for strict JSON. A group containing such a value has mean `"+inf"` and null std.

Each group reports sample count and per-metric `n`, `mean`, and sample `std`.
Empty groups/metrics have n=0 and null mean/std. A single value has null std.
Null values are excluded separately per metric. Source membership uses
`DatasetRepository.select_uids`: PIE-Bench UIDs have no colon, MagicBrush UIDs do.
The `all` group aggregates per-image values directly, not the mean of source
means. Valid metric counts can differ within a group.

## Two-stage core timing

Source: unique `batches.jsonl` records, associated through each result's
`inversion_batch_id` and `edit_batch_id`. Legacy repeated `inversion_seconds`,
`inversion_batch_seconds`, and `edit_batch_seconds` are retained as original
result metadata but are not used to aggregate core times.

- **Inversion:** only `durations_seconds.method`. Divide equally among that
  batch's `uncached_count` records with `artifact_reused=false`; cached records
  receive zero. Cache counts and participation must agree with the result rows.
- **Editing:** only `durations_seconds.editor`, divided equally among the
  `edited_count` participating records. This includes the source reconstruction
  and target edit produced by the editor for each input record.
- A mixed-source batch is allocated to its participating images before source
  grouping. Each group's total is the sum of its allocations, and its mean uses
  its images with known phase timing. Cached zeros participate in the mean.
- A null inversion batch ID denotes replay-only inversion timing. The duration
  and cache flag stay null and do not contribute to the known-time denominator.

These are amortized observed compute times, not individually measured image
latencies or cold-cache algorithm benchmarks. They exclude input reads, cache
lookup, workflow preparation/validation/staging, file writes, publication, and
model loading. The adapter call still includes its internal encoding and
transfers. Editing includes work inside the editor, including output RGB
quantization/transfers. Original cache-generation time is not recoverable.

Each source group records `timing.inversion` and `timing.editing`, each with `n`,
`total_core_seconds`, and `mean_core_seconds_per_image`. It also records
`inversion_cache.known_count`, `hit_count`, and `hit_rate`. Unknown timing has n=0
and null total/mean; missing source groups have zero sample count. Evaluation
duration is stored separately and is never used as experiment runtime.

## JSON layout and publication

One `<ZIP-stem>.json` is atomically published per ZIP:

```text
schema_version, status
archive: path, filename, size, modification time, original sweep, member hashes
dataset: reference, fingerprint, content digest, split, source assignment
evaluation: model/protocol/settings/software/hardware and evaluation timing
parameter_files
  <pipeline YAML filename>
    methods
      <method ID>
        configuration: full original resolved-config.json snapshot
        evaluation_elapsed_seconds
        groups: all / pie-bench / magic-brush
        scores: UID, source, target text, mask counts, metrics, core timing,
                and original result_metadata including batch associations
```

The archived configuration is authoritative, including original generation
model/revision, sampling steps/seed/eta, inversion and editing CFG, attention
settings, method settings, and runtime configuration. Evaluation device/hardware
and batch size are separate from those experiment settings; unavailable original
hardware details are not inferred from the evaluation machine.

Archive member SHA-256 hashes cover the manifests, configuration, records, timing,
and tensors read by the evaluator. Archive size/modification checks detect input
changes before publication. A failure leaves no partial final JSON for that
archive; reports for previously completed ZIPs remain usable. Temporary output is
removed on failure. Evaluation itself creates no workbook, preview, or separate
summary files. The independent summarizer below can extract compact reports later.

## Summarizing existing reports

```powershell
uv run python -m image_editing_inversion.evaluation.summary [REPORT_OR_DIR ...]
```

### Input selection

`evaluation/summary.py` implements `MetricsSummarizer(reports=None).run() -> Path`.
It reads existing JSON group statistics using only the standard library; it
does not load datasets, diffusion models or metric dependencies, or recompute
scores. The existing `evaluate_archives` API and evaluator outputs are unchanged.

Omitted inputs recursively select JSON files inside the fixed `data/metrics/`
root, excluding its `summaries/` subtree. Explicit file/directory paths resolve
from the caller's working directory; directories expand recursively and repeated
resolved files are deduplicated. Reports must use `schema_version=1` and
`status=complete`. Historical one-off formats are unsupported. Invalid inputs
fail with their report path and relevant field before publishing any output.

### Output files

Every invocation publishes a fresh directory
`data/metrics/summaries/<UTC-time>-<unique-id>/` containing:

- **`report.md`:** source reports, dataset identity, distinct metric protocol
  summaries, compact experiment settings, and separate quality and timing tables
  for `all`, `pie-bench`, and `magic-brush`. Quality cells show mean, sample std
  and valid n; timing tables include both phase counts, total core seconds,
  mean core seconds per image, cache known/hit counts, and cache hit rate.
- **`summary.csv`:** one row per source report, parameter file, method and group.
  Source columns include `source_id`, absolute `source_report`, archive filename,
  dataset reference/fingerprint/content digest/split/source assignment, and a
  `protocol_id` with compact JSON `evaluation_protocol`. IDs are local to this
  output. Settings include model ID/revision/resolution, sampling steps/seed/editing
  CFG, attention replacement fractions, and JSON `method_h_params`. Statistical
  columns are `sample_count`, `<metric>_n/mean/std`,
  `<phase>_n/total_core_seconds/mean_core_seconds_per_image`, and
  `inversion_cache_known_count/hit_count/hit_rate` (each slash denotes separate
  columns, with the prefix repeated).

CSV uses UTF-8 with BOM for Excel compatibility and preserves original numeric
precision. Null is an empty CSV cell and infinite PSNR is `+inf`. Markdown uses
four decimal places, `—` for null, and `+∞` for infinite PSNR. Cache hit rates
are fractions in `[0, 1]`. Source IDs distinguish repeated parameter/method names
across reports; rows are never averaged across runs, datasets or protocols.
The `all` statistics are copied directly rather than averaged from source means.

### Interpretation and limitations

The summarizer omits per-image scores, full configuration snapshots, member
hashes and evaluation duration. Core times remain observed batch-amortized times,
with cached inversion zeros and unknown timing distinguished. They must not be
read as cold-cache method benchmarks or individually measured image latencies.
The script preserves all inputs and publishes the two outputs together only
after validation and serialization succeed.

## Metric sources

[CLIP model](https://huggingface.co/openai/clip-vit-large-patch14),
[PIE-Bench masking implementation](https://github.com/cure-lab/PnPInversion/blob/main/evaluation/matrics_calculator.py),
[official LPIPS](https://github.com/richzhang/PerceptualSimilarity).
