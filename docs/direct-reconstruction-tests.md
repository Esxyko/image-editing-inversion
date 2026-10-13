# Direct reconstruction diagnostic tests

The opt-in GPU suite in `tests/test_direct_reconstruction.py` checks a fresh Direct
round trip through the real UNet, DDIM scheduler and Direct hooks. It also records
controlled comparisons for reconstruction drift; a hypothesis is not assumed to
be true just to make a test pass. Ordinary unittest discovery skips model runs.

```powershell
uv run python tests/test_direct_reconstruction.py --nsamples 16 --capture-batch 4 --edit-batch 16
```

Model files must already be cached. Use `--hf-home <directory>` if they are in a
different Hugging Face cache. Every invocation writes a new
`data/output/<timestamp>-direct-diagnostic/` directory and reads published
artifacts without changing them. Source encodings, pivots and fresh residuals are
computed in memory; they are never published as ordinary inversion cache entries.

The automated checks verify that the matched source-only round trip stays within
a dtype-based latent tolerance, deliberately omitting an offset is detected, and
observers restore attention processors and leave configuration/cache files intact.

| Comparison | Isolated variable |
| --- | --- |
| matched_source / joint_embedding_source | Actual capture embeddings versus joint source/target re-encoding |
| matched_source / large_batch_source | Source-only UNet batch, with identical embedding tensors |
| native_pairs / native_pairs_joint_embedding | Source embeddings on the paired production path |
| native_pairs / teacher_forced_pairs | Accumulation versus local prediction error at captured inputs |
| native_pairs / p2p_pairs | P2P processor versus native attention |
| branch_count_aligned_pairs | 4E paired branches equal 2B capture branches; sample composition still differs |
| cached_p2p_joint_embedding / fresh_p2p_joint_embedding | Published versus freshly recomputed artifacts |

For a precision comparison on selected samples, run both commands. Offload and
batch settings are the same in both, and FP32 recomputes all components rather
than casting FP16 offsets:

```powershell
uv run python tests/test_direct_reconstruction.py --uids 000000000000 000000000002 000000000005 000000000009 --capture-batch 1 --edit-batch 2 --dtype float16 --cpu-offload model --profile precision
uv run python tests/test_direct_reconstruction.py --uids 000000000000 000000000002 000000000005 000000000009 --capture-batch 1 --edit-batch 2 --dtype float32 --cpu-offload model --profile precision
```

Read `steps.jsonl` for per-image/timestep corrected latent RMSE and raw noise
prediction differences; read `scores.jsonl` and `report.md` for source PSNR and
RGB RMSE versus the same-precision VAE baseline. Teacher forcing is a local-error
probe and its output is not a free reconstruction quality estimate. Evidence for
a cause requires improvement in both trajectory error and final image fidelity.

The `precision` profile runs the matched, joint-embedding, larger source batch,
and native paired cases. Automated check outcomes are saved in `test-results.json`.

The CPU regressions cover original source embedding serialization, legacy v1/v2
compatibility, and Direct's original source-only residual addition. They do not
assert that free-running paired reconstruction must match VAE pixels exactly.
The former implementation profile based on forced source-trajectory replacement
has been removed. Historical outputs from that profile describe a withdrawn
algorithm variant, not evidence that the native reconstruction issue was fixed.
