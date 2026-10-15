# Null-text optimization diagnostics

Use the temporary `experiment-diagnostic` command to inspect Null-text
optimization and compare its final source latent with the actual replay result.

## Run a diagnostic experiment

Run the diagnostic workflow with the ordinary experiment settings and RGB output format:

```powershell
uv run experiment-diagnostic --method null-text --h-params high_steps.yaml --nsamples 64
```

This command supports only `null-text`; omitting `--method` selects it. Other
arguments, including `--dataset`, follow the
[ordinary experiment rules](workflows.md#small-comparisons). It always recomputes
inversion, source encodings, and conditional pivots. It publishes the
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
[diagnostic field definitions](#diagnostic-files).

## Diagnostic files

Each result record references the three diagnostic filenames.

### Optimization steps: `optimization.jsonl`

`optimization.jsonl` streams one completed optimization timestep per sample,
including UID, source/target prompts, zero-based `step_index`, actual `timestep`,
`target_pivot_index`, and `target_timestep` (`null` for the original source latent).
Metrics are defined as follows:

| Field | Meaning |
| --- | --- |
| `inner_losses` | MSE before each Adam update, in update order |
| `loss_before` | First entry of `inner_losses`, or null when guidance 1 skips optimization |
| `loss_after` | MSE measured after the last Adam update, or after the conditional-only step at guidance 1; matches that sample's `reconstruction_mse[step_index]` |
| `best_loss` | Minimum of the observed pre-update losses and final post-update loss |
| `gradient_norms` | L2 norm of the unconditional embedding's float32 gradient for each update |
| `embedding_update_norms` | L2 norm of the float32 embedding parameter change per update |
| `effective_embedding_change_fraction` | Fraction of embedding elements that differ before/after an update after both are cast to the model dtype |
| `learning_rate`, `early_stop_threshold` | Actual step-specific optimizer learning rate and stopping threshold |
| `actual_num_inner_steps` | Number of Adam updates; also the length of all per-update arrays |
| `stop_reason` | `guidance_one` when guidance 1 skips optimization; otherwise `threshold_reached` if the last pre-update loss is below the threshold, or `max_inner_steps` |
| `early_stop_loss_before_update` | Last pre-update loss, used by the ordinary early-stop check, or null when optimization is skipped |
| `loss_after_below_threshold` | Whether the final post-update loss also satisfies the threshold |

All losses compare scaled VAE latents in float32, not decoded RGB pixels. The
ordinary algorithm checks the pre-update loss after applying Adam; recording
preserves this behavior. Nonfinite diagnostic norms are serialized as strings
`nan`, `inf`, or `-inf`; the ordinary finite-value checks still apply to model
losses, gradients, embeddings, and latents. Completed steps remain available if
a later step or editing phase fails.

### Final reconstruction: `reconstruction.jsonl`

`reconstruction.jsonl` measures the final source branch before VAE decoding:

| Field | Meaning |
| --- | --- |
| `optimization_source_latent_mse` | Final optimized latent vs. original encoded source latent; matches `final_reconstruction_mse` |
| `replay_source_latent_mse` | Actual reconstruction latent from the editing pass vs. the same encoded source latent |
| `replay_optimized_latent_mse` | Actual reconstruction latent vs. the final optimized latent; isolates disagreement between optimization and replay |

### Reproducibility: `diagnostic-config.json`

`diagnostic-config.json` records both scheduler configurations and timestep
lists, model/dataset identities, model and embedding parameter dtypes,
hyperparameters, dependency/Python versions, and hashes of the relevant source
files. Diagnostics use detached observations and do not add UNet forward calls.
The temporary observer retains the encoded source and final optimized latent
for each selected sample until that sample's editing batch is prepared. Output
recording and diagnostic reductions add overhead to the reported timings.

For ordinary result fields and batch timing, see [run outputs](run-outputs.md).
For the separate Direct reconstruction suite, see
[Direct reconstruction diagnostics](direct-reconstruction-tests.md).
