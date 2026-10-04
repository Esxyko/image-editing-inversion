# Image Editing Inversion

## Load the project dataset

The project dataset is the private Hugging Face repository
`beatle-ju1ce/image-editing-inversion`. Request access to it, then set `HF_TOKEN`
in your environment or add it to the project's ignored `.env` file. The
`.env.example` file shows the key to add. An existing Hugging Face CLI login
also works.

```python
from image_editing_inversion.data import load_project_dataset

records = load_project_dataset()
example = records[0]
print(example["uid"])
```

The first call downloads the repository's `train` split and saves its Hugging
Face Datasets files directly in `data/`. Later calls load that saved copy without
network access. The `data/` directory is local and ignored by Git. A full first
download needs several gigabytes of free disk space for the download cache and
saved dataset.

## Dataset contents

Each record represents an edit with a source image, a binary edit mask, and a
pair of captions describing the source and intended result. The edited target
image was used to prepare MagicBrush captions but is not stored in the Hub
dataset.

### How the records are constructed

- **PIE-Bench:** All 700 entries in `mapping_file.json` are read in sorted
  sample-ID order. Each 512 × 512 source image is converted to RGB. Its
  run-length encoded edit mask is decoded to a 512 × 512 single-channel image,
  with the image boundary marked white. `original_prompt` and `editing_prompt`
  supply the caption pair after surrounding brackets are removed. The mapping
  key becomes the record UID.
- **MagicBrush:** The public train and dev records are combined in that order.
  Only edits with 1024 × 1024 source images are retained; their target images
  and masks must have matching dimensions. Source and target images are resized
  to 512 × 512 RGB. Masks are resized and thresholded to black or white; for
  RGBA masks, transparent pixels mark the edit area. A source and target caption
  pair is generated from the processed images and edit instruction, then matched
  back to the edit using its content fingerprint. Each UID is
  `<img_id>:<turn_index>`.

The combined artifact contains PIE-Bench records first, followed by the
retained MagicBrush train and dev records. Assembly requires one matching
caption pair for every retained MagicBrush edit and rejects duplicate UIDs.

### Record format

| Column | Value |
| --- | --- |
| `uid` | Non-empty source record identifier |
| `source_img` | 512 × 512 RGB image |
| `mask_img` | 512 × 512 single-channel mask; black is 0 and white is 255 |
| `source_prompt` | Non-empty caption for the source image |
| `target_prompt` | Non-empty caption for the intended edit |

## Package modules

The package has six subpackages with public APIs exported from their
`__init__.py` files:

| Subpackage | Responsibility |
| --- | --- |
| `config` | Experiment settings, hardware validation, and YAML loading |
| `data` | Dataset loading, identity validation, and UID-based sample access |
| `artifacts` | Portable inversion artifact validation and serialization |
| `inversion` | Adapter interfaces, context, hooks, discovery, and algorithms in `inversion.methods` |
| `generation` | Model runtime, token alignment, Prompt-to-Prompt, and denoising |
| `workflows` | Workflow orchestration, batching, and run output |

See [`structure.md`](structure.md) for implementation responsibilities and
dependency boundaries. Import public classes and functions from these
subpackages rather than their implementation files.

The former `image_editing_inversion.dataset`, `image_editing_inversion.artifact`,
and `image_editing_inversion.runner` modules have been removed. Use `data` for
dataset loading, `artifacts` for `InversionArtifact`, `inversion` for adapter and
hook APIs, and `workflows` for `WorkflowRunner`, `edit_artifacts`, and
`run_methods`. Configuration and generation APIs are exported from `config`
and `generation`. External adapters must update moved imports; there are no
compatibility wrappers. Saved artifacts retain their portable format; sweep
outputs are grouped by pipeline parameter filename as described below.

## Shared editing framework

The framework bundles baseline DDIM, Null-text Inversion, Direct Inversion, and ReNoise
methods and accepts inversion artifacts produced outside the package. It uses
Stable Diffusion 1.5, DDIM sampling, and a shared Prompt-to-Prompt editor to
produce a reconstruction
with `source_prompt` and an edit with `target_prompt`. The dataset mask is reserved
for later metrics; it is not used to constrain the edit.

Install the project with `uv sync`, then edit [`config.yaml`](config.yaml) and
the YAML files in [`pipeline_h_params/`](pipeline_h_params/). Run commands from
the project root. Shared configuration is always loaded from `config.yaml` in
the working directory. Its four required sections supply shared settings;
an optional `methods` section supplies adapter settings:

- `model` selects the SD 1.5 checkpoint, revision, and image dimensions.
- `sampling` selects DDIM eta and the editing seed.
- `prompt_to_prompt` selects the edit mode. `auto` uses replacement when source and target captions
  have the same number of whitespace-separated words, and refinement otherwise.
- Optional `methods.<method_id>.prompt_to_prompt` supplies additional settings
  validated by that method's P2P subclass; it does not override the shared mode
  or replacement fractions. The settings are saved with the run.
- `runtime` selects the device, precision, number of concurrent image pairs,
  attention query chunk size, CPU offload mode, VAE slicing/tiling, data-loading
  workers, and pinned memory.

The checked-in runtime values target CUDA with float16 and `batch_size: 1`.
Increase the batch size only when GPU memory permits; a batch of `N` edits has
`N` source and `N` target branches inside the shared editor. For a smaller GPU,
keep the batch size at one and consider `cpu_offload: model` or
`cpu_offload: sequential` and a smaller `attention_query_chunk_size`. An
unsupported device or option combination fails before model loading. A run saves its resolved
configuration in each parameter file's `resolved-config.json`.

### Pipeline parameter files and artifact reuse

Each file in `pipeline_h_params/` must define these four settings, using the
same grouping as [`pipeline_h_params/default.yaml`](pipeline_h_params/default.yaml):

```yaml
sampling:
  num_inference_steps: 50
  guidance_scale: 7.5
prompt_to_prompt:
  cross_replace_fraction: 0.4
  self_replace_fraction: 0.6
```

Steps must be a positive integer, guidance must be finite and nonnegative,
and both fractions must be between 0 and 1. All four values are required;
unknown keys and definitions of these settings in `config.yaml` are rejected.
The resolved Python settings still use `config.sampling` and
`config.prompt_to_prompt`. Use `load_config("default.yaml")` to resolve one file.

Both commands accept `--pipeline-h-params FILENAME` to select one YAML file:

```powershell
uv run image-editing-inversion run `
  --method ddim `
  --uid your-dataset-uid `
  --pipeline-h-params default.yaml `
  --output data/runs
```

Omit this option to process every top-level `.yaml` or `.yml` file in filename
order, completing all requested methods and samples for one file before the
next. Missing, empty, or malformed parameter files fail before model loading.
The dataset and loaded model are shared across the sweep; scheduler timesteps
and resolved settings are updated for each file. An invocation returns one
parent run directory:

```text
data/runs/<run-id>/
|-- sweep.json
`-- default.yaml/
    |-- resolved-config.json
    |-- results.jsonl
    `-- images/<sample-id>/
        |-- reconstructed.png
        `-- edited.png
```

Generated inversions are stored independently in `data/artifacts/`, with a
`catalog.json` index and unique directories containing `artifact.json` and
`tensors.safetensors`. Repeated runs consult this catalog even when `--output`
changes. Matching includes dataset identity/UID, model and scheduler settings,
guidance, effective method hyperparameters, numerical runtime settings, and
implementation/dependency fingerprints. File checksums and replay validation
protect against damaged entries; a missing or damaged artifact is recomputed
without deleting its old files. A malformed catalog fails clearly. Catalog
updates assume one active writer and publish complete artifacts before indexing
them.

Changing only attention fractions or editing policies reuses inversion. The
bundled deterministic methods also reuse inversion when the editing seed
changes. Changing steps, guidance, or effective inversion settings creates a
new artifact. Method hyperparameters are loaded for cache matching using the
same validated settings as inversion. Each run generates its images again.
Result records include the parameter filename, artifact path, and
`artifact_reused`; cache hits have `inversion_seconds: 0.0`.

An `edit-artifact` sweep records incompatible artifact/file pairs as `skipped`
with a reason and continues with compatible pairs. Malformed artifacts and
operational failures stop execution. If every pair is skipped, the command
returns a nonzero exit status. Explicit replay continues to support existing
uncataloged artifacts and does not read method hyperparameter files.

### Run baseline DDIM inversion

The bundled `ddim` method converts the source image to RGB, resizes it through
the pipeline image processor to the configured dimensions, and uses the VAE
posterior mode to encode it deterministically. It then follows the ascending
DDIM noise schedule using `source_prompt`, an empty negative caption, and the
shared `sampling.guidance_scale`. The inverse schedule must exactly reverse the
editor's denoising timesteps. This baseline requires `sampling.eta: 0.0` and
rejects dynamic thresholding or incompatible inverse schedules.

```powershell
uv run image-editing-inversion run `
  --method ddim `
  --uid your-dataset-uid `
  --output data/runs
```

The runner saves or reuses the terminal latent and provenance through the
artifact catalog, then produces `reconstructed.png` and `edited.png` with the shared
Prompt-to-Prompt policy. DDIM artifacts have no per-step state and need no
denoising hook. Inversion processes one sample at a time; reconstruction and
editing use `runtime.batch_size`. Saved DDIM artifacts can be replayed with
`edit-artifact` using a compatible configuration. DDIM inversion is approximate,
so reconstruction need not reproduce the source image exactly.

The implementation is also available as `DDIMInversion` from
`image_editing_inversion.inversion`. Adapters can share prompt encoding through
`context.editor.encode_prompts`, which validates the model's CLIP token limit.
Diffusers is imported when inversion executes; discovering the method does not
load a model. After updating an existing installation, run `uv sync` to refresh
the bundled method's Python entry-point metadata.

### Run Null-text Inversion

The bundled `null-text` method implements
[Null-text Inversion](https://null-text-inversion.github.io/). It deterministically
encodes the image with the VAE and builds a source-caption DDIM pivot trajectory
at guidance 1. It then optimizes one unconditional embedding per denoising step
using the shared `sampling.guidance_scale` and latent reconstruction MSE. Only
these embeddings are optimized; model weights and conditional embeddings stay
fixed. Each step starts from the previous step's optimized embedding.

Run from the project root so the adapter can find
[`method_h_params/Null_text.yaml`](method_h_params/Null_text.yaml). This file is
resolved relative to the working directory and loaded on the adapter's first
inversion or cache lookup; its validated settings are cached for that adapter instance. Missing
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
hardware restrictions. Reconstruction and editing use `runtime.batch_size`.

To compare all four bundled methods on the same sample and shared settings:

```powershell
uv run image-editing-inversion run `
  --method ddim `
  --method null-text `
  --method direct-inversion `
  --method renoise `
  --uid your-dataset-uid `
  --output data/runs
```

Select only `--method null-text` for a Null-text run. Refresh an existing
installation with `uv sync` to register the new entry point. The adapter is also
exported as `NullTextInversion` from `image_editing_inversion.inversion`.

Null-text artifacts store the terminal pivot plus `null_text_embeddings` of
shape `[T, 1, token_count, embedding_dim]` in descending denoising order and the
optimization `guidance_scale` repeated in a `[T]` tensor. The replay hook supplies
the saved embedding to both source and target unconditional branches at every
step. Replay rejects incompatible guidance scales and malformed embeddings.
Use `edit-artifact` with the same sampling guidance and compatible model/scheduler
settings; it requires neither the hyperparameter file nor further optimization.
Sequential offload can be used for replay, which runs without gradients.

### Run Direct Inversion

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
uv run image-editing-inversion run `
  --method direct-inversion `
  --uid your-dataset-uid `
  --output data/runs
```

Refresh an existing installation with `uv sync` to register the entry point.
The adapter is also exported as `DirectInversion` from
`image_editing_inversion.inversion`.

Direct Inversion requires `sampling.eta: 0.0`, finite nonnegative sampling
guidance, and a DDIM scheduler with epsilon prediction, leading timestep spacing,
and disabled sample clipping and dynamic thresholding. The inverse timesteps must
exactly reverse the editor's denoising schedule. Inversion and replay run without
gradients and support `cpu_offload: none`, `model`, and `sequential`, subject to
the existing hardware restrictions. Inversion processes one sample at a time;
reconstruction and editing use `runtime.batch_size` and the shared Prompt-to-Prompt
attention policy.

Direct Inversion artifacts retain schema v1 and store the terminal pivot plus
`direct_inversion_offsets` of shape `[T, 1, 4, height/8, width/8]` in descending
denoising order. They also store the sampling `guidance_scale` repeated in a
float64 `[T]` tensor. The replay hook adds each saved offset at full strength to
the source latent after the DDIM update; the target branch receives no direct
latent correction. Shared Prompt-to-Prompt attention transfer still affects
the target branch. Replay rejects malformed or non-finite state and incompatible
guidance, model, dataset, scheduler, or timestep settings. Use `edit-artifact`
with the same sampling guidance and compatible configuration; it performs no
new inversion. Reconstruction can differ from the original image because of
VAE encoding/decoding and numerical precision.

### Run ReNoise Inversion

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

Run from the project root so the adapter can find
[`method_h_params/ReNoise.yaml`](method_h_params/ReNoise.yaml). The file is loaded
on the adapter's first inversion or cache lookup and its validated settings are cached for that
instance. Missing or malformed files fail clearly, unknown keys are rejected,
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
uv run image-editing-inversion run `
  --method renoise `
  --uid your-dataset-uid `
  --output data/runs
```

Refresh an existing installation with `uv sync` to register the entry point.
The adapter is also exported as `ReNoiseInversion` from
`image_editing_inversion.inversion`.

ReNoise requires `sampling.eta: 0.0`, finite nonnegative sampling guidance, and
a DDIM scheduler with epsilon prediction, leading timestep spacing, and disabled
sample clipping and dynamic thresholding. Inversion and replay run without
gradients and support `cpu_offload: none`, `model`, and `sequential`, subject to
the existing hardware restrictions. Inversion processes one sample at a time;
reconstruction and editing use `runtime.batch_size` and the shared Prompt-to-Prompt
attention policy. This adapter implements deterministic SD 1.5 inversion;
noise regularization and stochastic noise correction are not included.

ReNoise artifacts retain schema v1 and store the terminal latent plus sampling
`guidance_scale` repeated in a float64 `[T]` tensor. Replay rejects malformed or
non-finite state and incompatible guidance, model, dataset, scheduler, or timestep
settings. The replay hook validates the timestep and latent shape while leaving
the shared denoising state unchanged. Use `edit-artifact` with the same sampling
guidance and compatible configuration; replay needs neither the hyperparameter
file nor further inversion. Reconstruction is approximate and can differ from
the source because of inversion error, VAE encoding/decoding, and precision.

### Edit from a saved artifact

An artifact directory contains `artifact.json` and `tensors.safetensors`. Its
`terminal_latent` tensor has shape `[1, 4, height/8, width/8]`. The JSON file
records the dataset reference/fingerprint and UID, inversion method ID, exact
model and DDIM scheduler settings, eta, and descending timesteps. Optional
per-step tensors have one entry per timestep and require that method's registered
denoising hook. The artifact does not duplicate dataset images, prompts, or masks.
Construct and save one with `InversionArtifact` from
`image_editing_inversion.artifacts`. Use the same model revision, scheduler config,
and timestep sequence as the editor; the runner rejects mismatches.

For the project Hub dataset, an external inversion implementation can package
its computed `z_t` tensor as follows:

```python
from image_editing_inversion.artifacts import InversionArtifact
from image_editing_inversion.config import load_config
from image_editing_inversion.data import load_project_dataset
from image_editing_inversion.generation import Editor

records = load_project_dataset()
config = load_config("default.yaml")
editor = Editor(config)
sample = records[0]
# z_t is a [1, 4, 64, 64] tensor produced by an inversion implementation.
artifact = InversionArtifact(
    method_id="external-ddim",
    model_id=config.model.model_id,
    model_revision=config.model.revision,
    dataset_ref="beatle-ju1ce/image-editing-inversion",
    dataset_fingerprint=records._fingerprint,
    sample_uid=sample["uid"],
    scheduler_id="ddim",
    scheduler_config=editor.scheduler_config,
    eta=config.sampling.eta,
    timesteps=editor.expected_timesteps,
    terminal_latent=z_t,
)
artifact.save("data/artifacts/external-example")
```

Before replay, install an `InversionMethod` adapter registered as
`external-ddim`, even if it uses the default P2P behavior and no denoising hook.
The artifact format itself has not changed.

```powershell
uv run image-editing-inversion edit-artifact `
  --artifact data/artifacts/external-example `
  --pipeline-h-params default.yaml `
  --output data/runs
```

Repeat `--artifact` to edit several samples. The default uses the private Hub
dataset loader described above. `--dataset-path` can select a separately saved
local dataset; artifacts made for that path use `local:image-editing` as their
dataset reference. Hub artifacts use the repository ID. The dataset fingerprint
must also match. Results are written under a new parent run directory with a
`sweep.json` manifest, and a child per parameter filename containing `results.jsonl`,
`resolved-config.json`, and `edited.png`/`reconstructed.png` per sample. Run
records contain UIDs and artifact paths, not prompts, source images, or masks.

### Add an inversion method

Import adapter contracts from the inversion subpackage:

```python
from image_editing_inversion.inversion import (
    DenoisingHook,
    DenoisingStepState,
    InversionContext,
    InversionMethod,
)
```

Implement `InversionMethod.invert(sample, context)` and return an
`InversionArtifact`. `InversionContext` provides the shared editor and resolved config,
plus dataset reference and fingerprint. A method that needs per-step behavior
also implements `create_denoising_hook(artifact)` to return a `DenoisingHook`.
Override the optional `validate_replay(artifact, context)` to check method-specific
state or configuration before the runner creates that hook; the default does
nothing. Direct editor callers should invoke this validation themselves before
creating a method's hook.
Raise `ArtifactCompatibilityError` from `image_editing_inversion.artifacts`
for configuration mismatches that replay sweeps can skip; keep malformed-state
errors as ordinary exceptions.

Automatic artifact reuse is opt-in for external adapters. Override
`inversion_cache_parameters(context)` and return a JSON-compatible mapping of
effective inversion settings, including any additional inputs that can change
inversion (such as a stochastic method's seed or versions of helper modules).
The default returns `None`, so artifacts are saved centrally without automatic
reuse. All four bundled adapters opt in; Null-text and ReNoise expose the same
resolved settings used by inversion. Explicit replay does not call this method.
Hooks receive source/target latents and text embeddings before and after each
denoising step. Install the method through the
`image_editing_inversion.methods` Python entry-point group. This discovery group
remains unchanged; Python imports now use `image_editing_inversion.inversion`,
with bundled implementations under `image_editing_inversion.inversion.methods`.
The former methods package has been removed without compatibility wrappers.
Published inversion implementations can then be adapted without changing the
common editor.

To customize attention behavior, subclass `PromptToPrompt` from
`image_editing_inversion.generation` and return that class from the adapter's
`prompt_to_prompt_class` property. The base class exposes `artifacts`,
`settings`, `step_index`, and `pair_maps`. Override `build_pair_map` for token
alignment, `active` for the replacement schedule, or
`replace_cross_attention` / `replace_self_attention` for attention transfer.
`active` receives the attention type, query token count, and UNet layer name.
Each replacement method receives source and target attention probabilities,
the pair index, and the UNet layer name; it must return a tensor with the target
probabilities' shape, device, and dtype. Model loading, four-branch UNet layout,
DDIM denoising, and denoising hooks remain shared.

For example, a subclass can blend mapped source attention with the target's
original attention:

```python
from image_editing_inversion.generation import PromptToPrompt

class BlendedPromptToPrompt(PromptToPrompt):
    @classmethod
    def validate_method_settings(cls, settings):
        if set(settings) != {"target_weight"}:
            raise ValueError("target_weight is required")
        weight = settings["target_weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
            raise ValueError("target_weight must be between 0 and 1")
        return {"target_weight": float(weight)}

    def replace_cross_attention(self, source_prob, target_prob, pair_index, layer_name):
        mapped = super().replace_cross_attention(
            source_prob, target_prob, pair_index, layer_name
        )
        weight = self.settings["target_weight"]
        return (1 - weight) * mapped + weight * target_prob
```

Select `BlendedPromptToPrompt` in `YourMethod`:

```python
@property
def prompt_to_prompt_class(self):
    return BlendedPromptToPrompt
```

Then add these settings to the experiment YAML:

```yaml
methods:
  your-method-id:
    prompt_to_prompt:
      target_weight: 0.25
```

For example, an adapter package can declare:

```toml
[project.entry-points."image_editing_inversion.methods"]
your-method-id = "your_package.method:YourMethod"
```

`YourMethod` must be an `InversionMethod` instance or a zero-argument subclass,
and its `method_id` must equal the entry-point name. An adapter whose editing
hook cannot share the configured GPU batch can set `max_edit_batch_size`;
the runner rejects a larger configured value.
The runner also rejects a GPU batch that mixes P2P subclasses or different
method-specific P2P settings. Replaying saved artifacts requires a registered
adapter for every artifact's method ID.

```powershell
uv run image-editing-inversion run `
  --method your-method-id `
  --uid your-dataset-uid `
  --output data/runs
```

Repeat `--method` or `--uid` for a comparison. The bundled `ddim`, `null-text`,
`direct-inversion`, and `renoise` methods are available after installing the project;
other method IDs require registered adapters. All methods for a given parameter
file use the same model, sampling, shared Prompt-to-Prompt, and runtime configuration. Both commands
read `config.yaml` from the working directory; `--pipeline-h-params` selects a
pipeline parameter filename, and omitting it sweeps all parameter files.
