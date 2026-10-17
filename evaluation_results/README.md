# Results — Image Editing Inversion Comparison

All outputs are generated from `data/metrics/summaries/.../summary.csv` (9 hyperparameter runs × 4 methods × 1,400 eval records from PIE-Bench + MagicBrush).

---

## Tables

### `results_tables.xlsx`

**Sheet: All Data / PIE-Bench / MagicBrush** — Per-method comparison table of all five evaluation metrics (CLIP text similarity, edited-region CLIP, masked PSNR, SSIM, LPIPS) under the default hyperparameter configuration; best value per metric is highlighted in green. Null-text inversion consistently leads on background-preservation metrics (PSNR, SSIM, LPIPS) while CLIP scores remain comparable across all four methods, indicating that inversion quality primarily affects background fidelity rather than edit semantic alignment.

**Sheet: Hyperparameter Ablation** — Full metric breakdown for all 9 hyperparameter configurations × 4 methods on All Data. Increasing inference steps and lowering the editing guidance CFG generally improves background preservation (PSNR/SSIM), while raising CFG or cross-replace fraction boosts edit strength at the cost of more background distortion.

**Sheet: Inference Timing** — Mean inversion and editing core time per image (seconds) for each method under the default configuration. Null-text inversion is ~14× slower than DDIM due to its per-image null-text optimisation loop, while DDIM, Direct Inversion, and ReNoise all complete inversion in under 1 second per image.

### `results_extra.xlsx`

**Sheet: Generalisation Gap** — Side-by-side PIE-Bench vs MagicBrush metric values for each method, with the average gap row at the bottom. All methods show lower PSNR and SSIM on MagicBrush than on PIE-Bench, suggesting MagicBrush edits are harder to localise; CLIP text similarity is slightly higher on MagicBrush, likely because its edits are more semantically distinct.

**Sheet: Best Config per Method** — The single hyperparameter configuration that maximises each method's composite score (normalised average of all five metrics). No method universally benefits from the same configuration — ReNoise peaks at a moderate setting while Null-text is relatively robust across configs — indicating that method-specific tuning is necessary for optimal performance.

---

## Figures

### `fig1_method_comparison.png`
Bar charts comparing all four methods across five metrics, split into three rows (All Data / PIE-Bench / MagicBrush). Null-text stands out on preservation metrics (PSNR, SSIM, LPIPS) in all three dataset splits, while the four methods cluster closely on CLIP scores, confirming that inversion choice matters most for background quality.

### `fig2_hyperparameter_sensitivity.png`
Line charts showing how each of four hyperparameters (editing CFG, inference steps, cross-replace fraction, self-replace fraction) affects CLIP, PSNR, SSIM, and LPIPS for each method. Steps and CFG are the most influential parameters: more steps consistently improve PSNR/SSIM, whereas higher CFG trades background fidelity for stronger edits.

### `fig3_timing.png`
Stacked bar chart of inversion + editing time per image (left) and a quality-vs-speed scatter plot (right). Null-text occupies the top-left of the scatter (best quality, slowest speed), while DDIM and ReNoise cluster at the bottom-right, highlighting a clear quality–efficiency Pareto frontier.

### `fig4_dataset_breakdown.png`
Side-by-side bars for PIE-Bench and MagicBrush showing CLIP, PSNR, SSIM, and LPIPS per method. The gap between the two datasets is largest for PSNR and SSIM, confirming that MagicBrush edits involve more structural change and are harder to preserve background fidelity on.

### `fig5_radar.png`
Radar chart with all five normalised metrics on a single plot for each method (higher = better on all axes, LPIPS inverted). Null-text has the largest area on preservation axes (PSNR, SSIM, LPIPS) while all methods have similar coverage on CLIP axes, making the background-vs-edit quality trade-off immediately visible.

### `fig6_best_config.png`
Heatmap of composite scores (normalised mean of five metrics) for all 9 configs × 4 methods (left), with the best cell per method outlined, alongside a grouped bar comparing default vs best-config composite scores (right). Gains from tuning are modest (< 0.05 composite points), suggesting the default configuration is a reasonable baseline for all methods.
