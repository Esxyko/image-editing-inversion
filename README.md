# Image Editing Inversion

## Load the project dataset

The project dataset is the private Hugging Face repository
`beatle-ju1ce/image-editing-inversion`. Request access to it, then set `HF_TOKEN`
in your environment or add it to the project's ignored `.env` file. The
`.env.example` file shows the key to add. An existing Hugging Face CLI login
also works.

```python
from image_editing_inversion.dataset import load_project_dataset

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

## Shared editing framework

The first framework milestone accepts inversion artifacts produced outside the
package. No inversion algorithm is bundled yet. It uses Stable Diffusion 1.5,
DDIM sampling, and a shared Prompt-to-Prompt editor to produce a reconstruction
with `source_prompt` and an edit with `target_prompt`. The dataset mask is reserved
for later metrics; it is not used to constrain the edit.

Install the project with `uv sync`, then select a YAML config file for each run.
The checked-in [`config/experiment.yaml`](config/experiment.yaml) is a starting
point. Its four required sections supply the model, sampling, editing, and
runtime settings; an optional `methods` section supplies adapter settings:

- `model` selects the SD 1.5 checkpoint, revision, and image dimensions.
- `sampling` selects the DDIM step count, eta, guidance scale, and seed.
- `prompt_to_prompt` selects the edit mode and the cross- and self-attention
  replacement fractions. `auto` uses replacement when source and target captions
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
configuration in `resolved-config.json`.

### Edit from a saved artifact

An artifact directory contains `artifact.json` and `tensors.safetensors`. Its
`terminal_latent` tensor has shape `[1, 4, height/8, width/8]`. The JSON file
records the dataset reference/fingerprint and UID, inversion method ID, exact
model and DDIM scheduler settings, eta, and descending timesteps. Optional
per-step tensors have one entry per timestep and require that method's registered
denoising hook. The artifact does not duplicate dataset images, prompts, or masks.
Construct and save one with `InversionArtifact` from
`image_editing_inversion.artifact`. Use the same model revision, scheduler config,
and timestep sequence as the editor; the runner rejects mismatches.

For the project Hub dataset, an external inversion implementation can package
its computed `z_t` tensor as follows:

```python
from image_editing_inversion.artifact import InversionArtifact
from image_editing_inversion.config import load_config
from image_editing_inversion.dataset import load_project_dataset
from image_editing_inversion.editing import Editor

records = load_project_dataset()
config = load_config("config/experiment.yaml")
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
artifact.save("data/inversions/example")
```

Before replay, install an `InversionMethod` adapter registered as
`external-ddim`, even if it uses the default P2P behavior and no denoising hook.
The artifact format itself has not changed.

```powershell
uv run image-editing-inversion edit-artifact `
  --config config/experiment.yaml `
  --artifact data/inversions/example `
  --output data/runs
```

Repeat `--artifact` to edit several samples. The default uses the private Hub
dataset loader described above. `--dataset-path` can select a separately saved
local dataset; artifacts made for that path use `local:image-editing` as their
dataset reference. Hub artifacts use the repository ID. The dataset fingerprint
must also match. Results are written under a new run directory with `results.jsonl`,
`resolved-config.json`, and `edited.png`/`reconstructed.png` per sample. Run
records contain UIDs and artifact paths, not prompts, source images, or masks.

### Add an inversion method

Implement `InversionMethod.invert(sample, context)` and return an
`InversionArtifact`. `InversionContext` provides the shared editor and resolved config,
plus dataset reference and fingerprint. A method that needs per-step behavior
also implements `create_denoising_hook(artifact)` to return a `DenoisingHook`.
Hooks receive source/target latents and text embeddings before and after each
denoising step. Install the method through the
`image_editing_inversion.methods` Python entry-point group. Published inversion
implementations can then be adapted without changing the common editor.

To customize attention behavior, subclass `PromptToPrompt` from
`image_editing_inversion.editing` and return that class from the adapter's
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
from image_editing_inversion.editing import PromptToPrompt

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
  --config config/experiment.yaml `
  --method your-method-id `
  --uid your-dataset-uid `
  --output data/runs
```

Repeat `--method` or `--uid` for a comparison. The `run` command will report a
missing adapter until one is installed. All methods in one run use the same
model, sampling, shared Prompt-to-Prompt, and runtime configuration. Both commands
require `--config`, so separate runs can select different YAML files.
