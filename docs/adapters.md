# Adapter extensions

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
Optional `validate_inversion_config(config)` checks settings without model weights;
optional `validate_inversion_context(context)` checks the actual scheduler before
samples are processed. Both default to no-ops for existing external adapters.
Implement them to participate fully in whole-sweep preflight, and reuse the same
checks in direct inversion calls. All bundled adapters do so. Replay invokes
neither method, so inversion-only restrictions and hyperparameter files do not
affect `edit_artifacts`.
Override the optional `validate_replay(artifact, context)` to check method-specific
state or configuration before the runner creates that hook; the default does
nothing. Direct editor callers should invoke this validation themselves before
creating a method's hook.
Adapters can override `invert_batch(samples, context)` to perform batched model
calls and must return one artifact per input, in input order. The workflow supplies
up to `runtime.inversion_batch_size` uncached samples. The default implementation
calls `invert` sequentially, so existing external adapters continue to work.
Bundled methods retain `invert` as a one-image wrapper. Shared
`context.editor.encode_images(images)` encodes a source batch, and
`context.editor.to_device(tensor, dtype=...)` applies the configured transfers;
omitting `dtype` preserves a tensor's existing dtype.
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
`image_editing_inversion.methods` Python entry-point group. Contracts are exported from `image_editing_inversion.inversion`; bundled
implementations live under `image_editing_inversion.inversion.methods`.
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

`YourMethod` must be a zero-argument subclass or factory returning a fresh
`InversionMethod`; its `method_id` must equal the entry-point name. An adapter whose editing
hook cannot share the configured GPU batch can set `max_edit_batch_size`;
the runner rejects a larger configured value.
The runner also rejects a GPU batch that mixes P2P subclasses or different
method-specific P2P settings. Replaying saved artifacts requires a registered
adapter for every artifact's method ID.

```powershell
uv run diffuse --method your-method-id --h-params ALL
```

Repeat `--method` for a comparison over the entire dataset, or use `--method ALL`
to select every discovered adapter once. The bundled `ddim`,
`null-text`, `direct-inversion`, and `renoise` methods are available after installing the project;
other method IDs require registered adapters. All methods for a given parameter
file use the same model, sampling, shared Prompt-to-Prompt, and runtime configuration. Both workflows
read the fixed `config.yaml` beside the package's `src/` directory. For `diffuse`, `--h-params` is
required: provide a pipeline parameter filename or the exact value `ALL` to
sweep every parameter file.
`edit_artifacts(artifact_id=None)` always sweeps all parameter files.

## Factory registration and lifetimes

For programmatic registration, pass a method ID and zero-argument factory:

```python
from image_editing_inversion.inversion import register_method, get_method
from your_package.method import YourMethod

register_method("your-method-id", YourMethod)
standalone_method = get_method("your-method-id")
```

Discovery registers factories without constructing adapters or reading method
hyperparameters. Each `get_method` call constructs a fresh adapter and verifies
its ID. Each workflow invocation retains one adapter per method, shared by its
preflight, inversion, and replay stages across parameter files. Factories must
return fresh instances; mutable state must not be shared between invocations.
Replay does not load inversion-only hyperparameters.

## Migration

| Previous usage | Current usage |
| --- | --- |
| `register_method(YourMethod())` | `register_method("your-method-id", YourMethod)` |
| Entry point exporting an adapter instance | Export a zero-argument class or factory |
| `get_method` returning a shared instance | Fresh instance on every call |
| `DatasetRepository(None)` or a local dataset path | `DatasetRepository()` for the fixed project dataset |
| Repository/cache root arguments | No arguments: fixed paths beside `src/` |
| Reading mutable runner fields such as `editor`, `dataset`, or `output_dir` | Use the returned run-directory `Path` and persisted snapshots/results |
| Shared inversion helpers under `inversion.methods.common` | Source/pivot reuse under `inversion.components`; algorithm helpers stay in `methods.common` |

`WorkflowRunner` can be reused: each call creates a new session, reloads the
configuration snapshot, and constructs fresh adapters and a model runtime.
The public inversion and hook signatures, entry-point group, tensor layouts,
cache identities, and saved storage schemas remain unchanged.

`config.project` provides read-only, source-relative project locations for
adapters needing fixed parameter files. It is excluded from `config.to_dict()`
and cache identity. Configuration models validate semantic constraints when
constructed directly as well as when loaded from YAML.
