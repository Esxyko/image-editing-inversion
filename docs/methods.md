# Bundled inversion methods

All methods use the shared Stable Diffusion 1.5 runtime and Prompt-to-Prompt
editor. Inversion uses `runtime.inversion_batch_size`; reconstruction/editing
uses `runtime.batch_size`. Both stages share device, dtype, and VAE settings.
All bundled inversions require eta zero and use deterministic VAE posterior mode.
The dataset mask is reserved for metrics and does not constrain editing.

The defaults in the tables below are **implementation defaults for omitted keys**.
The checked-in ReNoise YAML instead uses one refinement, a cap of one, and windows
`[0, 2]` / `[1, 2]`. The checked-in inversion/editing batch sizes are **2/8**.
Refresh entry-point metadata with `uv sync` after changing an installation.

## Baseline DDIM inversion

The bundled `ddim` method converts the source image to RGB, resizes it through
the pipeline image processor to the configured dimensions, and uses the VAE
posterior mode to encode it deterministically. It then follows the ascending
DDIM noise schedule using `source_prompt`, an empty negative caption, and the
shared `sampling.guidance_scale`. The inverse schedule must exactly reverse the
editor's denoising timesteps. This baseline requires `sampling.eta: 0.0` and
rejects dynamic thresholding or incompatible inverse schedules.

```powershell
uv run diffuse --method ddim --h-params ALL
```

The runner saves or reuses the terminal latent and provenance through the
artifact catalog, then produces `reconstructed.png` and `edited.png` with the shared
Prompt-to-Prompt policy. DDIM artifacts have no per-step state and need no
denoising hook. Inversion uses `runtime.inversion_batch_size`; reconstruction and
editing use `runtime.batch_size`. Saved DDIM artifacts can be replayed with
`edit_artifacts` using a compatible configuration. DDIM inversion is approximate,
so reconstruction need not reproduce the source image exactly.

The implementation is also available as `DDIMInversion` from
`image_editing_inversion.inversion`. Adapters can share prompt encoding through
`context.editor.encode_prompts`, which validates the model's CLIP token limit.
Diffusers is imported during inversion or scheduler preflight; discovering the method does not
load a model.

## Null-text Inversion

The bundled `null-text` method implements
[Null-text Inversion](https://null-text-inversion.github.io/). It deterministically
encodes the image with the VAE and builds a source-caption DDIM pivot trajectory
at guidance 1. It then optimizes one unconditional embedding per denoising step
using the shared `sampling.guidance_scale` and latent reconstruction MSE. Only
these embeddings are optimized; model weights and conditional embeddings stay
fixed. Each step starts from the previous step's optimized embedding.

The adapter reads
[`method_h_params/Null_text.yaml`](../method_h_params/Null_text.yaml). This file is
resolved relative to the fixed project root beside `src/` and loaded on the adapter's first
inversion preflight, direct inversion, or cache lookup; its validated settings are cached for that invocation's adapter instance. Missing
or malformed files fail clearly. Omitted keys use the defaults below, and unknown
keys are rejected. Optimization settings do not change the experiment YAML.

| Hyperparameter | Default | Meaning |
| --- | --- | --- |
| `num_inner_steps` | `10` | Maximum Adam updates per timestep; positive integer |
| `learning_rate` | `0.01` | Initial learning rate; finite and positive |
| `early_stop_epsilon` | `0.00001` | Initial latent-MSE stopping threshold; finite and nonnegative |
| `epsilon_increment` | `0.00002` | Threshold increase per timestep; finite and nonnegative |

For step index `i` out of `T`, the learning rate is
`learning_rate * (1 - i / (2 * T))` and the stopping threshold is
`early_stop_epsilon + i * epsilon_increment`. The optimized parameter and Adam
moments use float32, while embeddings are cast to the configured model precision
for UNet calls. Latent MSE is computed in float32.

Null-text inversion requires `sampling.eta: 0.0`, guidance greater than 1, and a
DDIM scheduler with epsilon prediction, leading timestep spacing, and disabled
sample clipping and dynamic thresholding. The inverse schedule must reverse the
editor's timesteps. Inversion supports `cpu_offload: none` and `model`;
sequential offload is rejected because optimization needs the UNet to remain
resident through backward passes. Float16, bfloat16, and float32 use the existing
hardware restrictions. Inversion uses `runtime.inversion_batch_size`; each image
keeps its own float32 Adam parameters, optimizer state, and early stopping
decision. Only active images participate in optimization model calls.
Reconstruction and editing use `runtime.batch_size`.

To compare all four bundled methods over the entire dataset with shared settings:

```powershell
uv run diffuse `
  --method ddim `
  --method null-text `
  --method direct-inversion `
  --method renoise `
  --h-params ALL
```

Select only `--method null-text` for a Null-text run. The adapter is also
exported as `NullTextInversion` from `image_editing_inversion.inversion`.

Null-text artifacts store the terminal pivot plus `null_text_embeddings` of
shape `[T, 1, token_count, embedding_dim]` in descending denoising order and the
optimization `guidance_scale` repeated in a `[T]` tensor. The replay hook supplies
the saved embedding to both source and target unconditional branches at every
step. Replay rejects incompatible guidance scales and malformed embeddings.
Use `edit_artifacts` with the same sampling guidance and compatible model/scheduler
settings; it requires neither the hyperparameter file nor further optimization.
Sequential offload can be used for replay, which runs without gradients.

## Direct Inversion

The bundled `direct-inversion` method implements Ju et al.'s
[Direct Inversion, later published as PnP Inversion](https://github.com/cure-lab/PnPInversion).
It deterministically encodes the source image with the VAE posterior mode and
builds a source-caption DDIM pivot trajectory at guidance 1, using the same
inverse scheduler approach as Null-text Inversion. It then follows the descending
schedule with an empty unconditional caption and the configured sampling guidance.
At each step, it records the latent residual
`previous_pivot - predicted_previous_latent` and advances from the corrected latent.
It performs no optimization and needs no method hyperparameter file.

```powershell
uv run diffuse --method direct-inversion --h-params ALL
```

The adapter is also exported as `DirectInversion` from
`image_editing_inversion.inversion`.

Direct Inversion requires `sampling.eta: 0.0`, finite nonnegative sampling
guidance, and a DDIM scheduler with epsilon prediction, leading timestep spacing,
and disabled sample clipping and dynamic thresholding. The inverse timesteps must
exactly reverse the editor's denoising schedule. Inversion and replay run without
gradients and support `cpu_offload: none`, `model`, and `sequential`, subject to
the existing hardware restrictions. Inversion uses `runtime.inversion_batch_size`;
reconstruction and editing use `runtime.batch_size` and the shared Prompt-to-Prompt
attention policy.

Direct Inversion artifacts store the terminal pivot plus
`direct_inversion_offsets` of shape `[T, 1, 4, height/8, width/8]` in descending
denoising order. They also store the sampling `guidance_scale` repeated in a
float64 `[T]` tensor. The replay hook adds each saved offset at full strength to
the source latent after the DDIM update; the target branch receives no direct
latent correction. Shared Prompt-to-Prompt attention transfer still affects
the target branch. Replay rejects malformed or non-finite state and incompatible
guidance, model, dataset, scheduler, or timestep settings. Use `edit_artifacts`
with the same sampling guidance and compatible configuration; it performs no
new inversion. Reconstruction can differ from the original image because of
VAE encoding/decoding and numerical precision.

## ReNoise Inversion

The bundled `renoise` method implements the core iterative noising and averaging
from [ReNoise: Real Image Inversion Through Iterative Noising](https://github.com/garibida/ReNoise-Inversion).
It deterministically encodes the source image with the VAE posterior mode and
traverses the shared DDIM timesteps in ascending order. At each timestep, the
previous latent remains fixed while successive UNet predictions refine the noisy
latent estimate. Every prediction uses the current noisy timestep, the source
caption, an empty unconditional caption, and `sampling.guidance_scale`.
The inverse update uses the editor's actual DDIM alpha products, timestep stride,
and final-step boundary. The selected predictions are averaged in float32 before
one final inverse update; UNet inputs use the configured runtime precision.

The adapter reads
[`method_h_params/ReNoise.yaml`](../method_h_params/ReNoise.yaml). The file is loaded
on the adapter's first inversion preflight, direct inversion, or cache lookup; its settings stay fixed for the invocation. Missing or malformed files fail clearly, unknown keys are rejected,
and omitted keys use these defaults:

| Hyperparameter | Default | Meaning |
| --- | --- | --- |
| `num_renoise_steps` | `9` | Refinements after the initial prediction; nonnegative integer |
| `max_num_renoise_steps_first_step` | `5` | Refinement cap for every timestep below 250; nonnegative integer |
| `average_latent_estimations` | `true` | Use the average of predictions in the selected window; boolean |
| `average_first_step_range` | `[0, 5]` | Prediction averaging window for timesteps below 250 |
| `average_step_range` | `[8, 10]` | Prediction averaging window for timesteps at or above 250 |

Averaging windows use zero-based prediction indices, including the initial
prediction at index 0, with an exclusive end. Each window must satisfy
`0 <= start < end <= effective_refinements + 1`, even when averaging is disabled.
Reduce the windows as well when reducing the iteration counts. With the default
settings, each timestep below 250 makes 6 UNet calls and each later timestep
makes 10; averaging adds no UNet call. Disabling averaging uses the last estimate.
These settings are independent of the shared experiment YAML.

```powershell
uv run diffuse --method renoise --h-params ALL
```

The adapter is also exported as `ReNoiseInversion` from
`image_editing_inversion.inversion`.

ReNoise requires `sampling.eta: 0.0`, finite nonnegative sampling guidance, and
a DDIM scheduler with epsilon prediction, leading timestep spacing, and disabled
sample clipping and dynamic thresholding. Inversion and replay run without
gradients and support `cpu_offload: none`, `model`, and `sequential`, subject to
the existing hardware restrictions. Inversion uses `runtime.inversion_batch_size`;
reconstruction and editing use `runtime.batch_size` and the shared Prompt-to-Prompt
attention policy. This adapter implements deterministic SD 1.5 inversion;
noise regularization and stochastic noise correction are not included.

ReNoise artifacts store the terminal latent plus sampling
`guidance_scale` repeated in a float64 `[T]` tensor. Replay rejects malformed or
non-finite state and incompatible guidance, model, dataset, scheduler, or timestep
settings. The replay hook validates the timestep and latent shape while leaving
the shared denoising state unchanged. Use `edit_artifacts` with the same sampling
guidance and compatible configuration; replay needs neither the hyperparameter
file nor further inversion. Reconstruction is approximate and can differ from
the source because of inversion error, VAE encoding/decoding, and precision.

See [artifact replay](artifacts.md#replay) and [adapter extensions](adapters.md).
