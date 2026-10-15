# Configuration guide

Configuration is split by responsibility. Keep shared settings, method parameters,
and the dataset fixed when comparing methods; vary pipeline parameter files to
sweep sampling and editing settings.

| Location | Responsibility |
| --- | --- |
| [config.yaml](../config.yaml) | Model, revision, image size, eta, seed, Prompt-to-Prompt mode, runtime, and optional method-specific attention settings |
| [pipeline_h_params/](../pipeline_h_params/) | Sampling steps, shared editing guidance, cross-attention and self-attention fractions |
| [method_h_params/](../method_h_params/) | Inversion guidance and algorithm-specific parameters |

Shared YAML is read once per invocation. Method parameters also stay fixed for
that invocation and are reloaded on the next call. Unknown settings and pipeline
settings placed in method files are rejected. Hardware and configuration checks
run before model loading; whole-sweep method preflight runs before sample inversion.

## Pipeline parameters

Every pipeline YAML requires all four fields:

```yaml
sampling:
  num_inference_steps: 15
  editing_guidance_scale: 7.5
prompt_to_prompt:
  cross_replace_fraction: 0.4
  self_replace_fraction: 0.4
```

Steps must be positive, attention fractions must be in `[0, 1]`, and editing
guidance must be finite and nonnegative. The old pipeline `guidance_scale` field
is no longer accepted.

The nine checked-in presets vary one setting relative to `default.yaml`:

| Preset | Setting changed from default |
| --- | --- |
| `default.yaml` | 15 steps, editing guidance 7.5, attention fractions 0.4 / 0.4 |
| `high_steps.yaml`, `low_steps.yaml` | Sampling steps |
| `high_editing_guidance.yaml`, `low_editing_guidance.yaml` | Editing guidance 10.0 / 5.0 |
| `high_cross_replace.yaml`, `low_cross_replace.yaml` | Cross-attention fraction |
| `high_self_replace.yaml`, `low_self_replace.yaml` | Self-attention fraction |

Select a filename with `--h-params default.yaml`. `diffuse --h-params ALL` sweeps
all top-level pipeline YAML files in filename order. `experiment` requires a
single filename. See [workflow selection rules](workflows.md).

## Inversion and editing guidance

Classifier-free guidance (CFG) has two separate settings:

| Setting | Defined in | Used for |
| --- | --- | --- |
| `inversion_guidance_scale` | Each method YAML | DDIM/ReNoise inversion, Direct residual calibration, and Null-text DDIM pivots |
| `sampling.editing_guidance_scale` | Each pipeline YAML | Source reconstruction and target editing for every bundled method; Null-text embedding optimization |

Every method YAML can specify inversion guidance:

```yaml
inversion_guidance_scale: 1.25
```

The checked-in method files use `1.25`; the implementation default for an omitted
key is `7.5`. DDIM reads `DDIM.yaml`, Direct reads `Direct.yaml`, Null-text reads
`Null_text.yaml`, and ReNoise reads `ReNoise.yaml`.

Both CFG values must be finite and nonnegative. Null-text requires both to be at
least `1`. Its empty-caption embeddings are optimized at editing CFG and replayed
at that same scale. When editing CFG is `1`, gradients for those embeddings are
zero, so Adam updates are skipped. Its DDIM pivots use inversion CFG; Direct's
conditional pivots use CFG `1`.

Editing guidance is shared by all methods in a pipeline run. It is rejected in
method YAML and method snapshots. For algorithm-specific settings and
restrictions, see [bundled methods](methods.md).

## Prompt-to-Prompt attention

`config.yaml` sets the shared mode. In `auto` mode, equal source/target
whitespace-word counts select replacement; different counts select refinement.
Pipeline YAML sets the cross-attention and self-attention fractions.

Optional settings under `methods.<method_id>.prompt_to_prompt` configure a
method's attention policy. They cannot override the shared mode or fractions.
See [attention extensions](adapters.md#attention-policies) for custom policies.

## Runtime and memory

The current checked-in runtime settings are:

| Setting | Value | Meaning |
| --- | --- | --- |
| `device` / `dtype` | `cuda` / `float16` | Shared model device and precision |
| `inversion_batch_size` | `8` | Source images per inversion batch |
| `batch_size` | `4` | Source/target pairs per editing batch |
| `num_workers` | `4` | Threads for dataset and artifact reads |
| `cpu_offload` | `none` | Model offload strategy |
| `attention_query_chunk_size` | `4096` | Attention query chunk limit |

Inversion and editing batch sizes are independent. An editing batch of `N` pairs
contains `N` source and `N` target branches, or `4N` UNet branches when using CFG.
Omitting `inversion_batch_size` defaults it to `1`.

Tune both batch sizes for the available memory and throughput, including
Null-text backward passes. CPU offload and attention chunking are configurable.
Null-text inversion supports `none` and `model` offload; replay also supports
`sequential`. An out-of-memory error stops the run.

## Compatibility and reuse

Cache reuse and replay require the complete current method parameters and pipeline
editing CFG to match the saved provenance. Changing either invalidates reuse, even
if an algorithm does not currently use one of its settings. Changing steps or
inversion CFG selects a different artifact group. Changing only attention fractions
or editing policies reuses compatible inversions.

Artifacts without complete method snapshots or separate editing CFG require
regeneration. Older Null-text artifacts without inversion-CFG pivot state also
require regeneration. See [artifact compatibility](artifacts.md#cache-reuse) and
[API migration](adapters.md#migration) for details.

## Project paths

Configuration and storage have fixed locations beside `src/`. Package paths resolve
from that source layout, independently of the working directory. Local model paths
resolve from the project root. Run the `uv` examples from the project root; installed
commands use the same fixed paths when invoked elsewhere.
