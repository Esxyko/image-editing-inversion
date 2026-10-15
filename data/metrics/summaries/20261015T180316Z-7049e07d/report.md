# 模型性能汇总

生成时间（UTC）：2026-10-15T18:03:16.961065+00:00。共 9 份输入，108 个分组汇总。

直接提取原报告统计，不合并运行、不重新评估。all 取原始整体统计，不平均数据集均值。

## 来源与评估协议

| 来源 | 报告路径 | ZIP | 数据集 | Fingerprint | Content digest | Split | 来源划分 | 协议 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | D:\Projects\image-editing-inversion\data\metrics\20261015T134317Z-66b48dd2\20261013T141357Z-5d372477.json | 20261013T141357Z-5d372477.zip | beatle-ju1ce/image-editing-inversion | 3c08518e41865876 | 3fc77b7baa92c68327fe127e6a83fbbd0786e99f5813eb4eca9e85d284732685 | eval | DatasetRepository.select_uids | P1 |
| R2 | D:\Projects\image-editing-inversion\data\metrics\20261015T135124Z-a9029d1f\20261013T141445Z-e8479caa.json | 20261013T141445Z-e8479caa.zip | beatle-ju1ce/image-editing-inversion | 3c08518e41865876 | 3fc77b7baa92c68327fe127e6a83fbbd0786e99f5813eb4eca9e85d284732685 | eval | DatasetRepository.select_uids | P1 |
| R3 | D:\Projects\image-editing-inversion\data\metrics\20261015T135941Z-82e87d4c\20261013T143816Z-054e5aca.json | 20261013T143816Z-054e5aca.zip | beatle-ju1ce/image-editing-inversion | 3c08518e41865876 | 3fc77b7baa92c68327fe127e6a83fbbd0786e99f5813eb4eca9e85d284732685 | eval | DatasetRepository.select_uids | P1 |
| R4 | D:\Projects\image-editing-inversion\data\metrics\20261015T140759Z-d6d0725e\20261014T153356Z-58d9b841.json | 20261014T153356Z-58d9b841.zip | beatle-ju1ce/image-editing-inversion | 3c08518e41865876 | 3fc77b7baa92c68327fe127e6a83fbbd0786e99f5813eb4eca9e85d284732685 | eval | DatasetRepository.select_uids | P1 |
| R5 | D:\Projects\image-editing-inversion\data\metrics\20261015T141643Z-52b30403\20261014T180131Z-d2d8e417.json | 20261014T180131Z-d2d8e417.zip | beatle-ju1ce/image-editing-inversion | 3c08518e41865876 | 3fc77b7baa92c68327fe127e6a83fbbd0786e99f5813eb4eca9e85d284732685 | eval | DatasetRepository.select_uids | P1 |
| R6 | D:\Projects\image-editing-inversion\data\metrics\20261015T142513Z-e404de34\20261014T115840Z-cd136154.json | 20261014T115840Z-cd136154.zip | beatle-ju1ce/image-editing-inversion | 3c08518e41865876 | 3fc77b7baa92c68327fe127e6a83fbbd0786e99f5813eb4eca9e85d284732685 | eval | DatasetRepository.select_uids | P1 |
| R7 | D:\Projects\image-editing-inversion\data\metrics\20261015T144843Z-8f958aa7\20261013T215240Z-96f013b2.json | 20261013T215240Z-96f013b2.zip | beatle-ju1ce/image-editing-inversion | 3c08518e41865876 | 3fc77b7baa92c68327fe127e6a83fbbd0786e99f5813eb4eca9e85d284732685 | eval | DatasetRepository.select_uids | P1 |
| R8 | D:\Projects\image-editing-inversion\data\metrics\20261015T173004Z-d9510f27\20261013T213744Z-7d52f157.json | 20261013T213744Z-7d52f157.zip | beatle-ju1ce/image-editing-inversion | 3c08518e41865876 | 3fc77b7baa92c68327fe127e6a83fbbd0786e99f5813eb4eca9e85d284732685 | eval | DatasetRepository.select_uids | P1 |
| R9 | D:\Projects\image-editing-inversion\data\metrics\20261015T173809Z-8357f074\20261013T230113Z-2c4f3c67.json | 20261013T230113Z-2c4f3c67.zip | beatle-ju1ce/image-editing-inversion | 3c08518e41865876 | 3fc77b7baa92c68327fe127e6a83fbbd0786e99f5813eb4eca9e85d284732685 | eval | DatasetRepository.select_uids | P1 |

### P1

- aggregation：arithmetic mean of per-image values; std is sample standard deviation; undefined std is null
- clip：{"input_resolution": 224, "max_text_tokens": 77, "model": "openai/clip-vit-large-patch14", "precision": "float32", "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41", "score": "raw cosine of normalized projected image/text features", "text_overflow": "truncate", "tf32": false}
- edited_region_clip：Keep mask_img==255; set every other RGB pixel to black before standard CLIP preprocessing; compare with target_prompt
- empty_region_policy：null score, excluded separately from each metric's statistics
- lpips：{"backbone": "SqueezeNet 1.1 ImageNet1K V1", "implementation": "official lpips", "input_range": "[-1,1] after RGB masking", "net": "squeeze", "spatial": false, "version": "0.1"}
- masked_background：Set mask_img==255 to black in BOTH source_img and edited; reduce over the full canvas, without background-area normalization
- psnr：{"data_range": 1.0, "formula": "-10*log10(mean((source_masked-edited_masked)^2))", "perfect_match": "+inf"}
- ssim：{"data_range": 1.0, "gaussian_kernel": true, "implementation": "torchmetrics functional", "k1": 0.01, "k2": 0.03, "kernel_size": 11, "sigma": 1.5}
- timing_protocol：{"editing": "batches.jsonl durations_seconds.editor, uniformly divided among edited participants", "grouping": "mean of per-image allocated core times, including cached zeros; cross-source batches allocated before grouping", "inversion": "batches.jsonl durations_seconds.method, uniformly divided among uncached participants; cache hits receive zero", "limitations": "batch-amortized observed core times, not individually measured latency or original cache-generation time; IO, preparation, publication and model loading excluded", "replay": "missing inversion batch ID means unknown time and cache state (null), not zero"}

## 实验设置

| 来源 | 参数 | 方法 | 生成模型 | Revision | 分辨率 | 步数 | Seed | 编辑 CFG | Cross | Self | 方法参数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | default.yaml | ddim | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.4000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R1 | default.yaml | direct-inversion | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.4000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R1 | default.yaml | null-text | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.4000 | 0.4000 | {"early_stop_epsilon": 1e-05, "epsilon_increment": 2e-05, "inversion_guidance_scale": 1.25, "learning_rate": 0.01, "num_inner_steps": 10} |
| R1 | default.yaml | renoise | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.4000 | 0.4000 | {"average_first_step_range": [0, 2], "average_latent_estimations": true, "average_step_range": [1, 2], "inversion_guidance_scale": 1.25, "max_num_renoise_steps_first_step": 1, "num_renoise_steps": 1} |
| R2 | high_editing_guidance.yaml | ddim | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 10.0000 | 0.4000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R2 | high_editing_guidance.yaml | direct-inversion | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 10.0000 | 0.4000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R2 | high_editing_guidance.yaml | null-text | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 10.0000 | 0.4000 | 0.4000 | {"early_stop_epsilon": 1e-05, "epsilon_increment": 2e-05, "inversion_guidance_scale": 1.25, "learning_rate": 0.01, "num_inner_steps": 10} |
| R2 | high_editing_guidance.yaml | renoise | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 10.0000 | 0.4000 | 0.4000 | {"average_first_step_range": [0, 2], "average_latent_estimations": true, "average_step_range": [1, 2], "inversion_guidance_scale": 1.25, "max_num_renoise_steps_first_step": 1, "num_renoise_steps": 1} |
| R3 | low_cross_replace.yaml | ddim | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.2000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R3 | low_cross_replace.yaml | direct-inversion | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.2000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R3 | low_cross_replace.yaml | null-text | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.2000 | 0.4000 | {"early_stop_epsilon": 1e-05, "epsilon_increment": 2e-05, "inversion_guidance_scale": 1.25, "learning_rate": 0.01, "num_inner_steps": 10} |
| R3 | low_cross_replace.yaml | renoise | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.2000 | 0.4000 | {"average_first_step_range": [0, 2], "average_latent_estimations": true, "average_step_range": [1, 2], "inversion_guidance_scale": 1.25, "max_num_renoise_steps_first_step": 1, "num_renoise_steps": 1} |
| R4 | high_steps.yaml | ddim | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 20 | 42 | 7.5000 | 0.4000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R4 | high_steps.yaml | direct-inversion | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 20 | 42 | 7.5000 | 0.4000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R4 | high_steps.yaml | null-text | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 20 | 42 | 7.5000 | 0.4000 | 0.4000 | {"early_stop_epsilon": 1e-05, "epsilon_increment": 2e-05, "inversion_guidance_scale": 1.25, "learning_rate": 0.01, "num_inner_steps": 10} |
| R4 | high_steps.yaml | renoise | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 20 | 42 | 7.5000 | 0.4000 | 0.4000 | {"average_first_step_range": [0, 2], "average_latent_estimations": true, "average_step_range": [1, 2], "inversion_guidance_scale": 1.25, "max_num_renoise_steps_first_step": 1, "num_renoise_steps": 1} |
| R5 | low_steps.yaml | ddim | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 10 | 42 | 7.5000 | 0.4000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R5 | low_steps.yaml | direct-inversion | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 10 | 42 | 7.5000 | 0.4000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R5 | low_steps.yaml | null-text | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 10 | 42 | 7.5000 | 0.4000 | 0.4000 | {"early_stop_epsilon": 1e-05, "epsilon_increment": 2e-05, "inversion_guidance_scale": 1.25, "learning_rate": 0.01, "num_inner_steps": 10} |
| R5 | low_steps.yaml | renoise | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 10 | 42 | 7.5000 | 0.4000 | 0.4000 | {"average_first_step_range": [0, 2], "average_latent_estimations": true, "average_step_range": [1, 2], "inversion_guidance_scale": 1.25, "max_num_renoise_steps_first_step": 1, "num_renoise_steps": 1} |
| R6 | high_cross_replace.yaml | ddim | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.6000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R6 | high_cross_replace.yaml | direct-inversion | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.6000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R6 | high_cross_replace.yaml | null-text | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.6000 | 0.4000 | {"early_stop_epsilon": 1e-05, "epsilon_increment": 2e-05, "inversion_guidance_scale": 1.25, "learning_rate": 0.01, "num_inner_steps": 10} |
| R6 | high_cross_replace.yaml | renoise | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.6000 | 0.4000 | {"average_first_step_range": [0, 2], "average_latent_estimations": true, "average_step_range": [1, 2], "inversion_guidance_scale": 1.25, "max_num_renoise_steps_first_step": 1, "num_renoise_steps": 1} |
| R7 | low_editing_guidance.yaml | ddim | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 5.0000 | 0.4000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R7 | low_editing_guidance.yaml | direct-inversion | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 5.0000 | 0.4000 | 0.4000 | {"inversion_guidance_scale": 1.25} |
| R7 | low_editing_guidance.yaml | null-text | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 5.0000 | 0.4000 | 0.4000 | {"early_stop_epsilon": 1e-05, "epsilon_increment": 2e-05, "inversion_guidance_scale": 1.25, "learning_rate": 0.01, "num_inner_steps": 10} |
| R7 | low_editing_guidance.yaml | renoise | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 5.0000 | 0.4000 | 0.4000 | {"average_first_step_range": [0, 2], "average_latent_estimations": true, "average_step_range": [1, 2], "inversion_guidance_scale": 1.25, "max_num_renoise_steps_first_step": 1, "num_renoise_steps": 1} |
| R8 | high_self_replace.yaml | ddim | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.4000 | 0.6000 | {"inversion_guidance_scale": 1.25} |
| R8 | high_self_replace.yaml | direct-inversion | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.4000 | 0.6000 | {"inversion_guidance_scale": 1.25} |
| R8 | high_self_replace.yaml | null-text | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.4000 | 0.6000 | {"early_stop_epsilon": 1e-05, "epsilon_increment": 2e-05, "inversion_guidance_scale": 1.25, "learning_rate": 0.01, "num_inner_steps": 10} |
| R8 | high_self_replace.yaml | renoise | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.4000 | 0.6000 | {"average_first_step_range": [0, 2], "average_latent_estimations": true, "average_step_range": [1, 2], "inversion_guidance_scale": 1.25, "max_num_renoise_steps_first_step": 1, "num_renoise_steps": 1} |
| R9 | low_self_replace.yaml | ddim | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.4000 | 0.2000 | {"inversion_guidance_scale": 1.25} |
| R9 | low_self_replace.yaml | direct-inversion | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.4000 | 0.2000 | {"inversion_guidance_scale": 1.25} |
| R9 | low_self_replace.yaml | null-text | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.4000 | 0.2000 | {"early_stop_epsilon": 1e-05, "epsilon_increment": 2e-05, "inversion_guidance_scale": 1.25, "learning_rate": 0.01, "num_inner_steps": 10} |
| R9 | low_self_replace.yaml | renoise | stable-diffusion-v1-5/stable-diffusion-v1-5 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 512×512 | 15 | 42 | 7.5000 | 0.4000 | 0.2000 | {"average_first_step_range": [0, 2], "average_latent_estimations": true, "average_step_range": [1, 2], "inversion_guidance_scale": 1.25, "max_num_renoise_steps_first_step": 1, "num_renoise_steps": 1} |

## 分组统计

质量单元格为均值 ± 样本标准差（n=有效样本数）。— 表示缺失或未定义，+∞ 表示无限 PSNR。Markdown 数值显示四位小数；CSV 保留原始精度。CLIP 为原始余弦相似度。

耗时单位为秒，是批次均摊的观测核心时间，并非单图实测延迟。缓存命中可使反演耗时为零，不代表原始反演成本；未知时间不当作零。耗时不含模型加载、外部 I/O、准备与发布，编辑包含源图重建和目标编辑。计时 n 与缓存已知数单独列出，评估耗时不作为实验耗时。

### all

质量：

| 来源 | 参数 | 方法 | 样本数 | CLIP ↑ | 区域 CLIP ↑ | 背景 PSNR (dB) ↑ | 背景 SSIM ↑ | 背景 LPIPS ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | default.yaml | ddim | 1400 | 0.2691 ± 0.0367 (n=1400) | 0.2118 ± 0.0485 (n=1400) | 17.8778 ± 3.4766 (n=1256) | 0.6812 ± 0.1470 (n=1256) | 0.2316 ± 0.1034 (n=1256) |
| R1 | default.yaml | direct-inversion | 1400 | 0.2694 ± 0.0373 (n=1400) | 0.2125 ± 0.0484 (n=1400) | 17.9038 ± 3.4726 (n=1256) | 0.6827 ± 0.1463 (n=1256) | 0.2286 ± 0.1027 (n=1256) |
| R1 | default.yaml | null-text | 1400 | 0.2598 ± 0.0383 (n=1400) | 0.2035 ± 0.0483 (n=1400) | 21.0741 ± 3.7572 (n=1256) | 0.7437 ± 0.1310 (n=1256) | 0.1535 ± 0.0765 (n=1256) |
| R1 | default.yaml | renoise | 1400 | 0.2700 ± 0.0366 (n=1400) | 0.2145 ± 0.0471 (n=1400) | 17.9670 ± 3.7355 (n=1256) | 0.6844 ± 0.1453 (n=1256) | 0.2271 ± 0.1011 (n=1256) |
| R2 | high_editing_guidance.yaml | ddim | 1400 | 0.2697 ± 0.0369 (n=1400) | 0.2127 ± 0.0490 (n=1400) | 16.6962 ± 3.4267 (n=1256) | 0.6524 ± 0.1544 (n=1256) | 0.2585 ± 0.1100 (n=1256) |
| R2 | high_editing_guidance.yaml | direct-inversion | 1400 | 0.2699 ± 0.0373 (n=1400) | 0.2124 ± 0.0484 (n=1400) | 16.7224 ± 3.4332 (n=1256) | 0.6535 ± 0.1536 (n=1256) | 0.2565 ± 0.1089 (n=1256) |
| R2 | high_editing_guidance.yaml | null-text | 1400 | 0.2638 ± 0.0384 (n=1400) | 0.2052 ± 0.0493 (n=1400) | 20.3892 ± 3.6853 (n=1256) | 0.7280 ± 0.1367 (n=1256) | 0.1668 ± 0.0816 (n=1256) |
| R2 | high_editing_guidance.yaml | renoise | 1400 | 0.2704 ± 0.0366 (n=1400) | 0.2154 ± 0.0469 (n=1400) | 16.7633 ± 3.6348 (n=1256) | 0.6544 ± 0.1537 (n=1256) | 0.2553 ± 0.1084 (n=1256) |
| R3 | low_cross_replace.yaml | ddim | 1400 | 0.2718 ± 0.0370 (n=1400) | 0.2137 ± 0.0489 (n=1400) | 17.8687 ± 3.4276 (n=1256) | 0.6810 ± 0.1469 (n=1256) | 0.2317 ± 0.1032 (n=1256) |
| R3 | low_cross_replace.yaml | direct-inversion | 1400 | 0.2714 ± 0.0372 (n=1400) | 0.2143 ± 0.0487 (n=1400) | 17.8884 ± 3.4460 (n=1256) | 0.6829 ± 0.1461 (n=1256) | 0.2287 ± 0.1025 (n=1256) |
| R3 | low_cross_replace.yaml | null-text | 1400 | 0.2620 ± 0.0376 (n=1400) | 0.2044 ± 0.0488 (n=1400) | 20.8516 ± 3.6990 (n=1256) | 0.7396 ± 0.1322 (n=1256) | 0.1583 ± 0.0785 (n=1256) |
| R3 | low_cross_replace.yaml | renoise | 1400 | 0.2724 ± 0.0366 (n=1400) | 0.2159 ± 0.0476 (n=1400) | 17.9459 ± 3.5741 (n=1256) | 0.6844 ± 0.1454 (n=1256) | 0.2275 ± 0.1012 (n=1256) |
| R4 | high_steps.yaml | ddim | 1400 | 0.2702 ± 0.0363 (n=1400) | 0.2134 ± 0.0483 (n=1400) | 17.3203 ± 3.5965 (n=1256) | 0.6679 ± 0.1502 (n=1256) | 0.2450 ± 0.1073 (n=1256) |
| R4 | high_steps.yaml | direct-inversion | 1400 | 0.2701 ± 0.0366 (n=1400) | 0.2137 ± 0.0482 (n=1400) | 17.3142 ± 3.5408 (n=1256) | 0.6694 ± 0.1496 (n=1256) | 0.2435 ± 0.1065 (n=1256) |
| R4 | high_steps.yaml | null-text | 1400 | 0.2592 ± 0.0374 (n=1400) | 0.2030 ± 0.0491 (n=1400) | 20.1270 ± 3.6504 (n=1256) | 0.7162 ± 0.1406 (n=1256) | 0.1800 ± 0.0852 (n=1256) |
| R4 | high_steps.yaml | renoise | 1400 | 0.2717 ± 0.0369 (n=1400) | 0.2157 ± 0.0481 (n=1400) | 17.3684 ± 3.5341 (n=1256) | 0.6695 ± 0.1492 (n=1256) | 0.2425 ± 0.1070 (n=1256) |
| R5 | low_steps.yaml | ddim | 1400 | 0.2677 ± 0.0377 (n=1400) | 0.2089 ± 0.0492 (n=1400) | 18.2646 ± 3.5900 (n=1256) | 0.6895 ± 0.1446 (n=1256) | 0.2207 ± 0.0993 (n=1256) |
| R5 | low_steps.yaml | direct-inversion | 1400 | 0.2674 ± 0.0374 (n=1400) | 0.2090 ± 0.0483 (n=1400) | 18.2993 ± 3.5723 (n=1256) | 0.6918 ± 0.1440 (n=1256) | 0.2155 ± 0.0971 (n=1256) |
| R5 | low_steps.yaml | null-text | 1400 | 0.2610 ± 0.0380 (n=1400) | 0.2037 ± 0.0480 (n=1400) | 21.7635 ± 3.9131 (n=1256) | 0.7640 ± 0.1252 (n=1256) | 0.1327 ± 0.0676 (n=1256) |
| R5 | low_steps.yaml | renoise | 1400 | 0.2681 ± 0.0374 (n=1400) | 0.2108 ± 0.0484 (n=1400) | 18.5085 ± 3.7907 (n=1256) | 0.6994 ± 0.1414 (n=1256) | 0.2119 ± 0.0956 (n=1256) |
| R6 | high_cross_replace.yaml | ddim | 1400 | 0.2672 ± 0.0370 (n=1400) | 0.2111 ± 0.0484 (n=1400) | 17.9021 ± 3.6017 (n=1256) | 0.6816 ± 0.1470 (n=1256) | 0.2311 ± 0.1036 (n=1256) |
| R6 | high_cross_replace.yaml | direct-inversion | 1400 | 0.2676 ± 0.0377 (n=1400) | 0.2117 ± 0.0480 (n=1400) | 17.9373 ± 3.6041 (n=1256) | 0.6834 ± 0.1461 (n=1256) | 0.2281 ± 0.1026 (n=1256) |
| R6 | high_cross_replace.yaml | null-text | 1400 | 0.2577 ± 0.0376 (n=1400) | 0.2020 ± 0.0478 (n=1400) | 21.2776 ± 3.7105 (n=1256) | 0.7494 ± 0.1301 (n=1256) | 0.1473 ± 0.0742 (n=1256) |
| R6 | high_cross_replace.yaml | renoise | 1400 | 0.2682 ± 0.0364 (n=1400) | 0.2131 ± 0.0471 (n=1400) | 17.9749 ± 3.6536 (n=1256) | 0.6849 ± 0.1451 (n=1256) | 0.2266 ± 0.1011 (n=1256) |
| R7 | low_editing_guidance.yaml | ddim | 1400 | 0.2674 ± 0.0371 (n=1400) | 0.2097 ± 0.0484 (n=1400) | 19.5684 ± 3.7118 (n=1256) | 0.7189 ± 0.1369 (n=1256) | 0.1931 ± 0.0933 (n=1256) |
| R7 | low_editing_guidance.yaml | direct-inversion | 1400 | 0.2676 ± 0.0373 (n=1400) | 0.2105 ± 0.0483 (n=1400) | 19.6927 ± 3.7184 (n=1256) | 0.7228 ± 0.1350 (n=1256) | 0.1885 ± 0.0913 (n=1256) |
| R7 | low_editing_guidance.yaml | null-text | 1400 | 0.2576 ± 0.0381 (n=1400) | 0.2017 ± 0.0474 (n=1400) | 21.3903 ± 3.6412 (n=1256) | 0.7522 ± 0.1273 (n=1256) | 0.1479 ± 0.0726 (n=1256) |
| R7 | low_editing_guidance.yaml | renoise | 1400 | 0.2682 ± 0.0372 (n=1400) | 0.2118 ± 0.0475 (n=1400) | 19.7694 ± 3.8270 (n=1256) | 0.7265 ± 0.1331 (n=1256) | 0.1855 ± 0.0900 (n=1256) |
| R8 | high_self_replace.yaml | ddim | 1400 | 0.2665 ± 0.0370 (n=1400) | 0.2110 ± 0.0482 (n=1400) | 17.8783 ± 3.5037 (n=1256) | 0.6810 ± 0.1472 (n=1256) | 0.2319 ± 0.1039 (n=1256) |
| R8 | high_self_replace.yaml | direct-inversion | 1400 | 0.2669 ± 0.0376 (n=1400) | 0.2111 ± 0.0479 (n=1400) | 17.9115 ± 3.4941 (n=1256) | 0.6826 ± 0.1465 (n=1256) | 0.2286 ± 0.1031 (n=1256) |
| R8 | high_self_replace.yaml | null-text | 1400 | 0.2564 ± 0.0390 (n=1400) | 0.2014 ± 0.0474 (n=1400) | 21.5470 ± 3.7348 (n=1256) | 0.7559 ± 0.1276 (n=1256) | 0.1398 ± 0.0701 (n=1256) |
| R8 | high_self_replace.yaml | renoise | 1400 | 0.2676 ± 0.0365 (n=1400) | 0.2129 ± 0.0472 (n=1400) | 17.9625 ± 3.6412 (n=1256) | 0.6845 ± 0.1453 (n=1256) | 0.2272 ± 0.1012 (n=1256) |
| R9 | low_self_replace.yaml | ddim | 1400 | 0.2722 ± 0.0362 (n=1400) | 0.2138 ± 0.0493 (n=1400) | 17.8653 ± 3.4784 (n=1256) | 0.6806 ± 0.1470 (n=1256) | 0.2316 ± 0.1032 (n=1256) |
| R9 | low_self_replace.yaml | direct-inversion | 1400 | 0.2722 ± 0.0365 (n=1400) | 0.2141 ± 0.0487 (n=1400) | 17.8657 ± 3.4639 (n=1256) | 0.6816 ± 0.1464 (n=1256) | 0.2304 ± 0.1031 (n=1256) |
| R9 | low_self_replace.yaml | null-text | 1400 | 0.2632 ± 0.0365 (n=1400) | 0.2057 ± 0.0488 (n=1400) | 20.4125 ± 3.7009 (n=1256) | 0.7291 ± 0.1349 (n=1256) | 0.1697 ± 0.0830 (n=1256) |
| R9 | low_self_replace.yaml | renoise | 1400 | 0.2735 ± 0.0362 (n=1400) | 0.2157 ± 0.0484 (n=1400) | 17.9662 ± 3.7419 (n=1256) | 0.6840 ± 0.1449 (n=1256) | 0.2270 ± 0.1009 (n=1256) |

耗时与缓存：

| 来源 | 参数 | 方法 | 反演 n | 反演总秒 | 反演秒/图 | 编辑 n | 编辑总秒 | 编辑秒/图 | 缓存已知数 | 缓存命中数 | 缓存命中率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | default.yaml | ddim | 1400 | 555.8935 | 0.3971 | 1400 | 1260.5073 | 0.9004 | 1400 | 0 | 0.0000 |
| R1 | default.yaml | direct-inversion | 1400 | 851.1642 | 0.6080 | 1400 | 1265.8773 | 0.9042 | 1400 | 0 | 0.0000 |
| R1 | default.yaml | null-text | 1400 | 7864.8209 | 5.6177 | 1400 | 1264.5117 | 0.9032 | 1400 | 0 | 0.0000 |
| R1 | default.yaml | renoise | 1400 | 1105.9158 | 0.7899 | 1400 | 1259.4378 | 0.8996 | 1400 | 0 | 0.0000 |
| R2 | high_editing_guidance.yaml | ddim | 1400 | 540.1368 | 0.3858 | 1400 | 1227.9361 | 0.8771 | 1400 | 0 | 0.0000 |
| R2 | high_editing_guidance.yaml | direct-inversion | 1400 | 823.5312 | 0.5882 | 1400 | 1232.6467 | 0.8805 | 1400 | 0 | 0.0000 |
| R2 | high_editing_guidance.yaml | null-text | 1400 | 7564.0500 | 5.4029 | 1400 | 1231.1095 | 0.8794 | 1400 | 0 | 0.0000 |
| R2 | high_editing_guidance.yaml | renoise | 1400 | 1076.2585 | 0.7688 | 1400 | 1227.6051 | 0.8769 | 1400 | 0 | 0.0000 |
| R3 | low_cross_replace.yaml | ddim | 1400 | 522.5582 | 0.3733 | 1400 | 1180.4748 | 0.8432 | 1400 | 0 | 0.0000 |
| R3 | low_cross_replace.yaml | direct-inversion | 1400 | 789.1819 | 0.5637 | 1400 | 1176.4010 | 0.8403 | 1400 | 0 | 0.0000 |
| R3 | low_cross_replace.yaml | null-text | 1400 | 7250.4235 | 5.1789 | 1400 | 1177.6670 | 0.8412 | 1400 | 0 | 0.0000 |
| R3 | low_cross_replace.yaml | renoise | 1400 | 1036.5493 | 0.7404 | 1400 | 1178.1183 | 0.8415 | 1400 | 0 | 0.0000 |
| R4 | high_steps.yaml | ddim | 1400 | 0.0000 | 0.0000 | 1400 | 1618.3039 | 1.1559 | 1400 | 1400 | 1.0000 |
| R4 | high_steps.yaml | direct-inversion | 1400 | 0.0000 | 0.0000 | 1400 | 1618.8875 | 1.1563 | 1400 | 1400 | 1.0000 |
| R4 | high_steps.yaml | null-text | 1400 | 0.0000 | 0.0000 | 1400 | 1621.2359 | 1.1580 | 1400 | 1400 | 1.0000 |
| R4 | high_steps.yaml | renoise | 1400 | 1462.8676 | 1.0449 | 1400 | 1616.5681 | 1.1547 | 1400 | 0 | 0.0000 |
| R5 | low_steps.yaml | ddim | 1400 | 415.5288 | 0.2968 | 1400 | 881.2012 | 0.6294 | 1400 | 0 | 0.0000 |
| R5 | low_steps.yaml | direct-inversion | 1400 | 564.2184 | 0.4030 | 1400 | 884.0221 | 0.6314 | 1400 | 0 | 0.0000 |
| R5 | low_steps.yaml | null-text | 1400 | 5191.2326 | 3.7080 | 1400 | 885.8454 | 0.6327 | 1400 | 0 | 0.0000 |
| R5 | low_steps.yaml | renoise | 1400 | 734.4823 | 0.5246 | 1400 | 883.9562 | 0.6314 | 1400 | 0 | 0.0000 |
| R6 | high_cross_replace.yaml | ddim | 1400 | 0.0000 | 0.0000 | 1400 | 1249.5139 | 0.8925 | 1400 | 1400 | 1.0000 |
| R6 | high_cross_replace.yaml | direct-inversion | 1400 | 554.5021 | 0.3961 | 1400 | 1248.0744 | 0.8915 | 1400 | 0 | 0.0000 |
| R6 | high_cross_replace.yaml | null-text | 1400 | 7094.7920 | 5.0677 | 1400 | 1249.5185 | 0.8925 | 1400 | 0 | 0.0000 |
| R6 | high_cross_replace.yaml | renoise | 1400 | 1086.2641 | 0.7759 | 1400 | 1247.7397 | 0.8912 | 1400 | 0 | 0.0000 |
| R7 | low_editing_guidance.yaml | ddim | 1400 | 583.0058 | 0.4164 | 1400 | 1226.1988 | 0.8759 | 1400 | 0 | 0.0000 |
| R7 | low_editing_guidance.yaml | direct-inversion | 1400 | 822.2130 | 0.5873 | 1400 | 1229.5489 | 0.8782 | 1400 | 0 | 0.0000 |
| R7 | low_editing_guidance.yaml | null-text | 1400 | 7561.1573 | 5.4008 | 1400 | 1235.9759 | 0.8828 | 1400 | 0 | 0.0000 |
| R7 | low_editing_guidance.yaml | renoise | 1400 | 1080.4516 | 0.7718 | 1400 | 1232.1215 | 0.8801 | 1400 | 0 | 0.0000 |
| R8 | high_self_replace.yaml | ddim | 1400 | 0.0000 | 0.0000 | 1400 | 1191.7200 | 0.8512 | 1400 | 1400 | 1.0000 |
| R8 | high_self_replace.yaml | direct-inversion | 1400 | 0.0000 | 0.0000 | 1400 | 1190.4485 | 0.8503 | 1400 | 1400 | 1.0000 |
| R8 | high_self_replace.yaml | null-text | 1400 | 0.0000 | 0.0000 | 1400 | 1193.0048 | 0.8521 | 1400 | 1400 | 1.0000 |
| R8 | high_self_replace.yaml | renoise | 1400 | 0.0000 | 0.0000 | 1400 | 1187.0887 | 0.8479 | 1400 | 1400 | 1.0000 |
| R9 | low_self_replace.yaml | ddim | 1400 | 0.0000 | 0.0000 | 1400 | 1189.9935 | 0.8500 | 1400 | 1400 | 1.0000 |
| R9 | low_self_replace.yaml | direct-inversion | 1400 | 0.0000 | 0.0000 | 1400 | 1189.1886 | 0.8494 | 1400 | 1400 | 1.0000 |
| R9 | low_self_replace.yaml | null-text | 1400 | 0.0000 | 0.0000 | 1400 | 1193.5973 | 0.8526 | 1400 | 1400 | 1.0000 |
| R9 | low_self_replace.yaml | renoise | 1400 | 0.0000 | 0.0000 | 1400 | 1188.9140 | 0.8492 | 1400 | 1400 | 1.0000 |

### pie-bench

质量：

| 来源 | 参数 | 方法 | 样本数 | CLIP ↑ | 区域 CLIP ↑ | 背景 PSNR (dB) ↑ | 背景 SSIM ↑ | 背景 LPIPS ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | default.yaml | ddim | 700 | 0.2632 ± 0.0367 (n=700) | 0.2278 ± 0.0493 (n=700) | 19.4366 ± 4.0984 (n=556) | 0.7451 ± 0.1506 (n=556) | 0.1770 ± 0.0959 (n=556) |
| R1 | default.yaml | direct-inversion | 700 | 0.2633 ± 0.0378 (n=700) | 0.2288 ± 0.0490 (n=700) | 19.4735 ± 4.0884 (n=556) | 0.7467 ± 0.1499 (n=556) | 0.1751 ± 0.0949 (n=556) |
| R1 | default.yaml | null-text | 700 | 0.2546 ± 0.0386 (n=700) | 0.2201 ± 0.0499 (n=700) | 22.6939 ± 4.4412 (n=556) | 0.7913 ± 0.1387 (n=556) | 0.1122 ± 0.0664 (n=556) |
| R1 | default.yaml | renoise | 700 | 0.2640 ± 0.0374 (n=700) | 0.2300 ± 0.0489 (n=700) | 19.6357 ± 4.5311 (n=556) | 0.7502 ± 0.1480 (n=556) | 0.1716 ± 0.0914 (n=556) |
| R2 | high_editing_guidance.yaml | ddim | 700 | 0.2640 ± 0.0373 (n=700) | 0.2294 ± 0.0492 (n=700) | 18.2510 ± 4.0593 (n=556) | 0.7227 ± 0.1580 (n=556) | 0.2006 ± 0.1049 (n=556) |
| R2 | high_editing_guidance.yaml | direct-inversion | 700 | 0.2639 ± 0.0380 (n=700) | 0.2288 ± 0.0493 (n=700) | 18.2691 ± 4.0740 (n=556) | 0.7234 ± 0.1567 (n=556) | 0.2001 ± 0.1041 (n=556) |
| R2 | high_editing_guidance.yaml | null-text | 700 | 0.2581 ± 0.0384 (n=700) | 0.2228 ± 0.0504 (n=700) | 22.0528 ± 4.2395 (n=556) | 0.7824 ± 0.1415 (n=556) | 0.1214 ± 0.0697 (n=556) |
| R2 | high_editing_guidance.yaml | renoise | 700 | 0.2643 ± 0.0368 (n=700) | 0.2303 ± 0.0477 (n=700) | 18.4278 ± 4.4027 (n=556) | 0.7279 ± 0.1553 (n=556) | 0.1958 ± 0.1006 (n=556) |
| R3 | low_cross_replace.yaml | ddim | 700 | 0.2652 ± 0.0372 (n=700) | 0.2298 ± 0.0494 (n=700) | 19.4221 ± 4.0008 (n=556) | 0.7451 ± 0.1501 (n=556) | 0.1768 ± 0.0954 (n=556) |
| R3 | low_cross_replace.yaml | direct-inversion | 700 | 0.2653 ± 0.0375 (n=700) | 0.2307 ± 0.0487 (n=700) | 19.4461 ± 4.0287 (n=556) | 0.7466 ± 0.1496 (n=556) | 0.1751 ± 0.0947 (n=556) |
| R3 | low_cross_replace.yaml | null-text | 700 | 0.2577 ± 0.0375 (n=700) | 0.2222 ± 0.0495 (n=700) | 22.4572 ± 4.3623 (n=556) | 0.7881 ± 0.1393 (n=556) | 0.1159 ± 0.0684 (n=556) |
| R3 | low_cross_replace.yaml | renoise | 700 | 0.2663 ± 0.0374 (n=700) | 0.2317 ± 0.0489 (n=700) | 19.5903 ± 4.2395 (n=556) | 0.7501 ± 0.1481 (n=556) | 0.1719 ± 0.0911 (n=556) |
| R4 | high_steps.yaml | ddim | 700 | 0.2640 ± 0.0366 (n=700) | 0.2295 ± 0.0485 (n=700) | 18.8833 ± 4.3298 (n=556) | 0.7344 ± 0.1541 (n=556) | 0.1881 ± 0.1005 (n=556) |
| R4 | high_steps.yaml | direct-inversion | 700 | 0.2639 ± 0.0371 (n=700) | 0.2295 ± 0.0485 (n=700) | 18.8446 ± 4.2514 (n=556) | 0.7350 ± 0.1542 (n=556) | 0.1879 ± 0.0993 (n=556) |
| R4 | high_steps.yaml | null-text | 700 | 0.2562 ± 0.0372 (n=700) | 0.2218 ± 0.0496 (n=700) | 21.6872 ± 4.2805 (n=556) | 0.7711 ± 0.1456 (n=556) | 0.1327 ± 0.0740 (n=556) |
| R4 | high_steps.yaml | renoise | 700 | 0.2659 ± 0.0370 (n=700) | 0.2316 ± 0.0488 (n=700) | 18.9877 ± 4.1937 (n=556) | 0.7381 ± 0.1523 (n=556) | 0.1835 ± 0.0977 (n=556) |
| R5 | low_steps.yaml | ddim | 700 | 0.2623 ± 0.0379 (n=700) | 0.2261 ± 0.0502 (n=700) | 19.8746 ± 4.2916 (n=556) | 0.7509 ± 0.1503 (n=556) | 0.1691 ± 0.0935 (n=556) |
| R5 | low_steps.yaml | direct-inversion | 700 | 0.2616 ± 0.0378 (n=700) | 0.2254 ± 0.0495 (n=700) | 19.9127 ± 4.2563 (n=556) | 0.7532 ± 0.1485 (n=556) | 0.1660 ± 0.0910 (n=556) |
| R5 | low_steps.yaml | null-text | 700 | 0.2558 ± 0.0377 (n=700) | 0.2199 ± 0.0500 (n=700) | 23.4328 ± 4.6443 (n=556) | 0.8054 ± 0.1349 (n=556) | 0.0972 ± 0.0592 (n=556) |
| R5 | low_steps.yaml | renoise | 700 | 0.2621 ± 0.0384 (n=700) | 0.2260 ± 0.0499 (n=700) | 20.2307 ± 4.5886 (n=556) | 0.7615 ± 0.1453 (n=556) | 0.1607 ± 0.0876 (n=556) |
| R6 | high_cross_replace.yaml | ddim | 700 | 0.2614 ± 0.0368 (n=700) | 0.2271 ± 0.0491 (n=700) | 19.4777 ± 4.3217 (n=556) | 0.7454 ± 0.1506 (n=556) | 0.1767 ± 0.0961 (n=556) |
| R6 | high_cross_replace.yaml | direct-inversion | 700 | 0.2614 ± 0.0378 (n=700) | 0.2275 ± 0.0489 (n=700) | 19.5219 ± 4.3253 (n=556) | 0.7472 ± 0.1495 (n=556) | 0.1749 ± 0.0950 (n=556) |
| R6 | high_cross_replace.yaml | null-text | 700 | 0.2533 ± 0.0381 (n=700) | 0.2185 ± 0.0501 (n=700) | 22.8598 ± 4.3693 (n=556) | 0.7953 ± 0.1379 (n=556) | 0.1072 ± 0.0629 (n=556) |
| R6 | high_cross_replace.yaml | renoise | 700 | 0.2621 ± 0.0369 (n=700) | 0.2283 ± 0.0486 (n=700) | 19.6274 ± 4.3899 (n=556) | 0.7503 ± 0.1479 (n=556) | 0.1714 ± 0.0912 (n=556) |
| R7 | low_editing_guidance.yaml | ddim | 700 | 0.2608 ± 0.0373 (n=700) | 0.2257 ± 0.0493 (n=700) | 21.2068 ± 4.4072 (n=556) | 0.7739 ± 0.1417 (n=556) | 0.1434 ± 0.0827 (n=556) |
| R7 | low_editing_guidance.yaml | direct-inversion | 700 | 0.2607 ± 0.0374 (n=700) | 0.2266 ± 0.0498 (n=700) | 21.3099 ± 4.4253 (n=556) | 0.7762 ± 0.1405 (n=556) | 0.1413 ± 0.0813 (n=556) |
| R7 | low_editing_guidance.yaml | null-text | 700 | 0.2523 ± 0.0385 (n=700) | 0.2179 ± 0.0490 (n=700) | 22.9780 ± 4.2983 (n=556) | 0.7961 ± 0.1370 (n=556) | 0.1090 ± 0.0636 (n=556) |
| R7 | low_editing_guidance.yaml | renoise | 700 | 0.2613 ± 0.0370 (n=700) | 0.2269 ± 0.0491 (n=700) | 21.4541 ± 4.6084 (n=556) | 0.7808 ± 0.1380 (n=556) | 0.1365 ± 0.0775 (n=556) |
| R8 | high_self_replace.yaml | ddim | 700 | 0.2612 ± 0.0363 (n=700) | 0.2267 ± 0.0492 (n=700) | 19.4443 ± 4.1473 (n=556) | 0.7450 ± 0.1509 (n=556) | 0.1775 ± 0.0968 (n=556) |
| R8 | high_self_replace.yaml | direct-inversion | 700 | 0.2609 ± 0.0377 (n=700) | 0.2271 ± 0.0486 (n=700) | 19.4792 ± 4.1238 (n=556) | 0.7464 ± 0.1502 (n=556) | 0.1756 ± 0.0959 (n=556) |
| R8 | high_self_replace.yaml | null-text | 700 | 0.2518 ± 0.0386 (n=700) | 0.2179 ± 0.0492 (n=700) | 23.0928 ± 4.4179 (n=556) | 0.7989 ± 0.1360 (n=556) | 0.1034 ± 0.0612 (n=556) |
| R8 | high_self_replace.yaml | renoise | 700 | 0.2619 ± 0.0374 (n=700) | 0.2285 ± 0.0486 (n=700) | 19.6144 ± 4.3685 (n=556) | 0.7501 ± 0.1483 (n=556) | 0.1719 ± 0.0917 (n=556) |
| R9 | low_self_replace.yaml | ddim | 700 | 0.2657 ± 0.0368 (n=700) | 0.2306 ± 0.0495 (n=700) | 19.4375 ± 4.1066 (n=556) | 0.7451 ± 0.1504 (n=556) | 0.1767 ± 0.0954 (n=556) |
| R9 | low_self_replace.yaml | direct-inversion | 700 | 0.2658 ± 0.0369 (n=700) | 0.2309 ± 0.0492 (n=700) | 19.4373 ± 4.0642 (n=556) | 0.7460 ± 0.1491 (n=556) | 0.1766 ± 0.0954 (n=556) |
| R9 | low_self_replace.yaml | null-text | 700 | 0.2585 ± 0.0370 (n=700) | 0.2233 ± 0.0494 (n=700) | 22.0606 ± 4.3185 (n=556) | 0.7815 ± 0.1410 (n=556) | 0.1235 ± 0.0709 (n=556) |
| R9 | low_self_replace.yaml | renoise | 700 | 0.2670 ± 0.0359 (n=700) | 0.2323 ± 0.0495 (n=700) | 19.6387 ± 4.5433 (n=556) | 0.7495 ± 0.1478 (n=556) | 0.1716 ± 0.0917 (n=556) |

耗时与缓存：

| 来源 | 参数 | 方法 | 反演 n | 反演总秒 | 反演秒/图 | 编辑 n | 编辑总秒 | 编辑秒/图 | 缓存已知数 | 缓存命中数 | 缓存命中率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | default.yaml | ddim | 700 | 277.9016 | 0.3970 | 700 | 629.3453 | 0.8991 | 700 | 0 | 0.0000 |
| R1 | default.yaml | direct-inversion | 700 | 425.9219 | 0.6085 | 700 | 632.8233 | 0.9040 | 700 | 0 | 0.0000 |
| R1 | default.yaml | null-text | 700 | 3932.0989 | 5.6173 | 700 | 631.8205 | 0.9026 | 700 | 0 | 0.0000 |
| R1 | default.yaml | renoise | 700 | 552.9415 | 0.7899 | 700 | 629.2239 | 0.8989 | 700 | 0 | 0.0000 |
| R2 | high_editing_guidance.yaml | ddim | 700 | 270.1053 | 0.3859 | 700 | 613.7674 | 0.8768 | 700 | 0 | 0.0000 |
| R2 | high_editing_guidance.yaml | direct-inversion | 700 | 411.9162 | 0.5885 | 700 | 616.0144 | 0.8800 | 700 | 0 | 0.0000 |
| R2 | high_editing_guidance.yaml | null-text | 700 | 3781.0899 | 5.4016 | 700 | 615.1574 | 0.8788 | 700 | 0 | 0.0000 |
| R2 | high_editing_guidance.yaml | renoise | 700 | 538.0192 | 0.7686 | 700 | 613.4023 | 0.8763 | 700 | 0 | 0.0000 |
| R3 | low_cross_replace.yaml | ddim | 700 | 261.6666 | 0.3738 | 700 | 590.1211 | 0.8430 | 700 | 0 | 0.0000 |
| R3 | low_cross_replace.yaml | direct-inversion | 700 | 395.0015 | 0.5643 | 700 | 588.0540 | 0.8401 | 700 | 0 | 0.0000 |
| R3 | low_cross_replace.yaml | null-text | 700 | 3628.1097 | 5.1830 | 700 | 588.3840 | 0.8405 | 700 | 0 | 0.0000 |
| R3 | low_cross_replace.yaml | renoise | 700 | 517.6533 | 0.7395 | 700 | 589.2923 | 0.8418 | 700 | 0 | 0.0000 |
| R4 | high_steps.yaml | ddim | 700 | 0.0000 | 0.0000 | 700 | 809.6181 | 1.1566 | 700 | 700 | 1.0000 |
| R4 | high_steps.yaml | direct-inversion | 700 | 0.0000 | 0.0000 | 700 | 809.1397 | 1.1559 | 700 | 700 | 1.0000 |
| R4 | high_steps.yaml | null-text | 700 | 0.0000 | 0.0000 | 700 | 809.8201 | 1.1569 | 700 | 700 | 1.0000 |
| R4 | high_steps.yaml | renoise | 700 | 729.2090 | 1.0417 | 700 | 807.2348 | 1.1532 | 700 | 0 | 0.0000 |
| R5 | low_steps.yaml | ddim | 700 | 207.6581 | 0.2967 | 700 | 440.2711 | 0.6290 | 700 | 0 | 0.0000 |
| R5 | low_steps.yaml | direct-inversion | 700 | 281.7690 | 0.4025 | 700 | 441.8859 | 0.6313 | 700 | 0 | 0.0000 |
| R5 | low_steps.yaml | null-text | 700 | 2593.3090 | 3.7047 | 700 | 442.5105 | 0.6322 | 700 | 0 | 0.0000 |
| R5 | low_steps.yaml | renoise | 700 | 367.4125 | 0.5249 | 700 | 441.7518 | 0.6311 | 700 | 0 | 0.0000 |
| R6 | high_cross_replace.yaml | ddim | 700 | 0.0000 | 0.0000 | 700 | 623.6502 | 0.8909 | 700 | 700 | 1.0000 |
| R6 | high_cross_replace.yaml | direct-inversion | 700 | 277.5188 | 0.3965 | 700 | 623.8344 | 0.8912 | 700 | 0 | 0.0000 |
| R6 | high_cross_replace.yaml | null-text | 700 | 3547.4479 | 5.0678 | 700 | 623.7348 | 0.8910 | 700 | 0 | 0.0000 |
| R6 | high_cross_replace.yaml | renoise | 700 | 543.0700 | 0.7758 | 700 | 623.4545 | 0.8906 | 700 | 0 | 0.0000 |
| R7 | low_editing_guidance.yaml | ddim | 700 | 291.4848 | 0.4164 | 700 | 612.8366 | 0.8755 | 700 | 0 | 0.0000 |
| R7 | low_editing_guidance.yaml | direct-inversion | 700 | 411.1633 | 0.5874 | 700 | 614.2819 | 0.8775 | 700 | 0 | 0.0000 |
| R7 | low_editing_guidance.yaml | null-text | 700 | 3776.7202 | 5.3953 | 700 | 617.4965 | 0.8821 | 700 | 0 | 0.0000 |
| R7 | low_editing_guidance.yaml | renoise | 700 | 540.2811 | 0.7718 | 700 | 615.5166 | 0.8793 | 700 | 0 | 0.0000 |
| R8 | high_self_replace.yaml | ddim | 700 | 0.0000 | 0.0000 | 700 | 596.4608 | 0.8521 | 700 | 700 | 1.0000 |
| R8 | high_self_replace.yaml | direct-inversion | 700 | 0.0000 | 0.0000 | 700 | 594.8580 | 0.8498 | 700 | 700 | 1.0000 |
| R8 | high_self_replace.yaml | null-text | 700 | 0.0000 | 0.0000 | 700 | 595.2772 | 0.8504 | 700 | 700 | 1.0000 |
| R8 | high_self_replace.yaml | renoise | 700 | 0.0000 | 0.0000 | 700 | 593.9450 | 0.8485 | 700 | 700 | 1.0000 |
| R9 | low_self_replace.yaml | ddim | 700 | 0.0000 | 0.0000 | 700 | 594.8502 | 0.8498 | 700 | 700 | 1.0000 |
| R9 | low_self_replace.yaml | direct-inversion | 700 | 0.0000 | 0.0000 | 700 | 593.6190 | 0.8480 | 700 | 700 | 1.0000 |
| R9 | low_self_replace.yaml | null-text | 700 | 0.0000 | 0.0000 | 700 | 596.4442 | 0.8521 | 700 | 700 | 1.0000 |
| R9 | low_self_replace.yaml | renoise | 700 | 0.0000 | 0.0000 | 700 | 593.9624 | 0.8485 | 700 | 700 | 1.0000 |

### magic-brush

质量：

| 来源 | 参数 | 方法 | 样本数 | CLIP ↑ | 区域 CLIP ↑ | 背景 PSNR (dB) ↑ | 背景 SSIM ↑ | 背景 LPIPS ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | default.yaml | ddim | 700 | 0.2751 ± 0.0358 (n=700) | 0.1959 ± 0.0420 (n=700) | 16.6398 ± 2.2128 (n=700) | 0.6304 ± 0.1222 (n=700) | 0.2750 ± 0.0875 (n=700) |
| R1 | default.yaml | direct-inversion | 700 | 0.2755 ± 0.0359 (n=700) | 0.1963 ± 0.0419 (n=700) | 16.6570 ± 2.2051 (n=700) | 0.6319 ± 0.1215 (n=700) | 0.2712 ± 0.0876 (n=700) |
| R1 | default.yaml | null-text | 700 | 0.2651 ± 0.0373 (n=700) | 0.1869 ± 0.0403 (n=700) | 19.7876 ± 2.4373 (n=700) | 0.7059 ± 0.1109 (n=700) | 0.1863 ± 0.0676 (n=700) |
| R1 | default.yaml | renoise | 700 | 0.2761 ± 0.0349 (n=700) | 0.1990 ± 0.0397 (n=700) | 16.6416 ± 2.1858 (n=700) | 0.6321 ± 0.1198 (n=700) | 0.2711 ± 0.0857 (n=700) |
| R2 | high_editing_guidance.yaml | ddim | 700 | 0.2754 ± 0.0356 (n=700) | 0.1961 ± 0.0427 (n=700) | 15.4612 ± 2.1327 (n=700) | 0.5966 ± 0.1262 (n=700) | 0.3045 ± 0.0906 (n=700) |
| R2 | high_editing_guidance.yaml | direct-inversion | 700 | 0.2759 ± 0.0355 (n=700) | 0.1961 ± 0.0415 (n=700) | 15.4939 ± 2.1379 (n=700) | 0.5979 ± 0.1260 (n=700) | 0.3013 ± 0.0903 (n=700) |
| R2 | high_editing_guidance.yaml | null-text | 700 | 0.2695 ± 0.0376 (n=700) | 0.1876 ± 0.0413 (n=700) | 19.0679 ± 2.4829 (n=700) | 0.6849 ± 0.1160 (n=700) | 0.2029 ± 0.0718 (n=700) |
| R2 | high_editing_guidance.yaml | renoise | 700 | 0.2765 ± 0.0354 (n=700) | 0.2004 ± 0.0410 (n=700) | 15.4412 ± 2.0920 (n=700) | 0.5961 ± 0.1248 (n=700) | 0.3027 ± 0.0894 (n=700) |
| R3 | low_cross_replace.yaml | ddim | 700 | 0.2785 ± 0.0357 (n=700) | 0.1977 ± 0.0428 (n=700) | 16.6348 ± 2.2227 (n=700) | 0.6301 ± 0.1224 (n=700) | 0.2753 ± 0.0871 (n=700) |
| R3 | low_cross_replace.yaml | direct-inversion | 700 | 0.2775 ± 0.0358 (n=700) | 0.1980 ± 0.0429 (n=700) | 16.6511 ± 2.2296 (n=700) | 0.6323 ± 0.1214 (n=700) | 0.2712 ± 0.0874 (n=700) |
| R3 | low_cross_replace.yaml | null-text | 700 | 0.2663 ± 0.0373 (n=700) | 0.1867 ± 0.0410 (n=700) | 19.5762 ± 2.4035 (n=700) | 0.7010 ± 0.1122 (n=700) | 0.1919 ± 0.0692 (n=700) |
| R3 | low_cross_replace.yaml | renoise | 700 | 0.2786 ± 0.0347 (n=700) | 0.2002 ± 0.0407 (n=700) | 16.6398 ± 2.1921 (n=700) | 0.6322 ± 0.1200 (n=700) | 0.2716 ± 0.0859 (n=700) |
| R4 | high_steps.yaml | ddim | 700 | 0.2764 ± 0.0349 (n=700) | 0.1974 ± 0.0424 (n=700) | 16.0789 ± 2.2026 (n=700) | 0.6152 ± 0.1239 (n=700) | 0.2901 ± 0.0895 (n=700) |
| R4 | high_steps.yaml | direct-inversion | 700 | 0.2763 ± 0.0350 (n=700) | 0.1979 ± 0.0424 (n=700) | 16.0986 ± 2.1946 (n=700) | 0.6173 ± 0.1232 (n=700) | 0.2877 ± 0.0902 (n=700) |
| R4 | high_steps.yaml | null-text | 700 | 0.2623 ± 0.0374 (n=700) | 0.1841 ± 0.0407 (n=700) | 18.8878 ± 2.4294 (n=700) | 0.6726 ± 0.1198 (n=700) | 0.2176 ± 0.0741 (n=700) |
| R4 | high_steps.yaml | renoise | 700 | 0.2776 ± 0.0358 (n=700) | 0.1998 ± 0.0417 (n=700) | 16.0822 ± 2.1721 (n=700) | 0.6150 ± 0.1218 (n=700) | 0.2893 ± 0.0895 (n=700) |
| R5 | low_steps.yaml | ddim | 700 | 0.2732 ± 0.0367 (n=700) | 0.1918 ± 0.0418 (n=700) | 16.9858 ± 2.1946 (n=700) | 0.6407 ± 0.1192 (n=700) | 0.2616 ± 0.0835 (n=700) |
| R5 | low_steps.yaml | direct-inversion | 700 | 0.2731 ± 0.0361 (n=700) | 0.1926 ± 0.0410 (n=700) | 17.0179 ± 2.1937 (n=700) | 0.6430 ± 0.1197 (n=700) | 0.2548 ± 0.0827 (n=700) |
| R5 | low_steps.yaml | null-text | 700 | 0.2662 ± 0.0376 (n=700) | 0.1875 ± 0.0398 (n=700) | 20.4377 ± 2.5276 (n=700) | 0.7311 ± 0.1061 (n=700) | 0.1608 ± 0.0603 (n=700) |
| R5 | low_steps.yaml | renoise | 700 | 0.2741 ± 0.0355 (n=700) | 0.1957 ± 0.0417 (n=700) | 17.1406 ± 2.2020 (n=700) | 0.6500 ± 0.1168 (n=700) | 0.2525 ± 0.0811 (n=700) |
| R6 | high_cross_replace.yaml | ddim | 700 | 0.2729 ± 0.0364 (n=700) | 0.1951 ± 0.0419 (n=700) | 16.6507 ± 2.2177 (n=700) | 0.6310 ± 0.1224 (n=700) | 0.2743 ± 0.0879 (n=700) |
| R6 | high_cross_replace.yaml | direct-inversion | 700 | 0.2737 ± 0.0366 (n=700) | 0.1959 ± 0.0416 (n=700) | 16.6787 ± 2.2099 (n=700) | 0.6328 ± 0.1215 (n=700) | 0.2703 ± 0.0877 (n=700) |
| R6 | high_cross_replace.yaml | null-text | 700 | 0.2621 ± 0.0365 (n=700) | 0.1855 ± 0.0389 (n=700) | 20.0210 ± 2.4473 (n=700) | 0.7130 ± 0.1109 (n=700) | 0.1792 ± 0.0667 (n=700) |
| R6 | high_cross_replace.yaml | renoise | 700 | 0.2743 ± 0.0348 (n=700) | 0.1978 ± 0.0400 (n=700) | 16.6623 ± 2.1837 (n=700) | 0.6329 ± 0.1198 (n=700) | 0.2704 ± 0.0860 (n=700) |
| R7 | low_editing_guidance.yaml | ddim | 700 | 0.2740 ± 0.0358 (n=700) | 0.1937 ± 0.0416 (n=700) | 18.2670 ± 2.3416 (n=700) | 0.6752 ± 0.1157 (n=700) | 0.2326 ± 0.0817 (n=700) |
| R7 | low_editing_guidance.yaml | direct-inversion | 700 | 0.2746 ± 0.0360 (n=700) | 0.1943 ± 0.0408 (n=700) | 18.4083 ± 2.3543 (n=700) | 0.6805 ± 0.1139 (n=700) | 0.2261 ± 0.0808 (n=700) |
| R7 | low_editing_guidance.yaml | null-text | 700 | 0.2629 ± 0.0371 (n=700) | 0.1856 ± 0.0397 (n=700) | 20.1291 ± 2.3529 (n=700) | 0.7174 ± 0.1070 (n=700) | 0.1787 ± 0.0641 (n=700) |
| R7 | low_editing_guidance.yaml | renoise | 700 | 0.2752 ± 0.0361 (n=700) | 0.1967 ± 0.0407 (n=700) | 18.4312 ± 2.3200 (n=700) | 0.6833 ± 0.1117 (n=700) | 0.2244 ± 0.0797 (n=700) |
| R8 | high_self_replace.yaml | ddim | 700 | 0.2718 ± 0.0369 (n=700) | 0.1953 ± 0.0416 (n=700) | 16.6344 ± 2.2097 (n=700) | 0.6302 ± 0.1224 (n=700) | 0.2751 ± 0.0877 (n=700) |
| R8 | high_self_replace.yaml | direct-inversion | 700 | 0.2729 ± 0.0365 (n=700) | 0.1951 ± 0.0414 (n=700) | 16.6664 ± 2.2159 (n=700) | 0.6318 ± 0.1217 (n=700) | 0.2707 ± 0.0882 (n=700) |
| R8 | high_self_replace.yaml | null-text | 700 | 0.2610 ± 0.0389 (n=700) | 0.1849 ± 0.0392 (n=700) | 20.3192 ± 2.4772 (n=700) | 0.7217 ± 0.1093 (n=700) | 0.1688 ± 0.0628 (n=700) |
| R8 | high_self_replace.yaml | renoise | 700 | 0.2733 ± 0.0346 (n=700) | 0.1974 ± 0.0401 (n=700) | 16.6504 ± 2.1810 (n=700) | 0.6325 ± 0.1197 (n=700) | 0.2710 ± 0.0858 (n=700) |
| R9 | low_self_replace.yaml | ddim | 700 | 0.2786 ± 0.0344 (n=700) | 0.1969 ± 0.0431 (n=700) | 16.6165 ± 2.1922 (n=700) | 0.6295 ± 0.1222 (n=700) | 0.2751 ± 0.0871 (n=700) |
| R9 | low_self_replace.yaml | direct-inversion | 700 | 0.2787 ± 0.0349 (n=700) | 0.1972 ± 0.0420 (n=700) | 16.6174 ± 2.2141 (n=700) | 0.6304 ± 0.1221 (n=700) | 0.2732 ± 0.0878 (n=700) |
| R9 | low_self_replace.yaml | null-text | 700 | 0.2680 ± 0.0353 (n=700) | 0.1881 ± 0.0412 (n=700) | 19.1034 ± 2.4304 (n=700) | 0.6875 ± 0.1139 (n=700) | 0.2064 ± 0.0730 (n=700) |
| R9 | low_self_replace.yaml | renoise | 700 | 0.2801 ± 0.0353 (n=700) | 0.1992 ± 0.0412 (n=700) | 16.6377 ± 2.1812 (n=700) | 0.6320 ± 0.1193 (n=700) | 0.2709 ± 0.0850 (n=700) |

耗时与缓存：

| 来源 | 参数 | 方法 | 反演 n | 反演总秒 | 反演秒/图 | 编辑 n | 编辑总秒 | 编辑秒/图 | 缓存已知数 | 缓存命中数 | 缓存命中率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | default.yaml | ddim | 700 | 277.9919 | 0.3971 | 700 | 631.1620 | 0.9017 | 700 | 0 | 0.0000 |
| R1 | default.yaml | direct-inversion | 700 | 425.2423 | 0.6075 | 700 | 633.0541 | 0.9044 | 700 | 0 | 0.0000 |
| R1 | default.yaml | null-text | 700 | 3932.7219 | 5.6182 | 700 | 632.6913 | 0.9038 | 700 | 0 | 0.0000 |
| R1 | default.yaml | renoise | 700 | 552.9743 | 0.7900 | 700 | 630.2139 | 0.9003 | 700 | 0 | 0.0000 |
| R2 | high_editing_guidance.yaml | ddim | 700 | 270.0315 | 0.3858 | 700 | 614.1687 | 0.8774 | 700 | 0 | 0.0000 |
| R2 | high_editing_guidance.yaml | direct-inversion | 700 | 411.6150 | 0.5880 | 700 | 616.6322 | 0.8809 | 700 | 0 | 0.0000 |
| R2 | high_editing_guidance.yaml | null-text | 700 | 3782.9601 | 5.4042 | 700 | 615.9521 | 0.8799 | 700 | 0 | 0.0000 |
| R2 | high_editing_guidance.yaml | renoise | 700 | 538.2393 | 0.7689 | 700 | 614.2028 | 0.8774 | 700 | 0 | 0.0000 |
| R3 | low_cross_replace.yaml | ddim | 700 | 260.8916 | 0.3727 | 700 | 590.3537 | 0.8434 | 700 | 0 | 0.0000 |
| R3 | low_cross_replace.yaml | direct-inversion | 700 | 394.1805 | 0.5631 | 700 | 588.3470 | 0.8405 | 700 | 0 | 0.0000 |
| R3 | low_cross_replace.yaml | null-text | 700 | 3622.3138 | 5.1747 | 700 | 589.2829 | 0.8418 | 700 | 0 | 0.0000 |
| R3 | low_cross_replace.yaml | renoise | 700 | 518.8960 | 0.7413 | 700 | 588.8260 | 0.8412 | 700 | 0 | 0.0000 |
| R4 | high_steps.yaml | ddim | 700 | 0.0000 | 0.0000 | 700 | 808.6857 | 1.1553 | 700 | 700 | 1.0000 |
| R4 | high_steps.yaml | direct-inversion | 700 | 0.0000 | 0.0000 | 700 | 809.7479 | 1.1568 | 700 | 700 | 1.0000 |
| R4 | high_steps.yaml | null-text | 700 | 0.0000 | 0.0000 | 700 | 811.4158 | 1.1592 | 700 | 700 | 1.0000 |
| R4 | high_steps.yaml | renoise | 700 | 733.6586 | 1.0481 | 700 | 809.3333 | 1.1562 | 700 | 0 | 0.0000 |
| R5 | low_steps.yaml | ddim | 700 | 207.8707 | 0.2970 | 700 | 440.9301 | 0.6299 | 700 | 0 | 0.0000 |
| R5 | low_steps.yaml | direct-inversion | 700 | 282.4495 | 0.4035 | 700 | 442.1362 | 0.6316 | 700 | 0 | 0.0000 |
| R5 | low_steps.yaml | null-text | 700 | 2597.9235 | 3.7113 | 700 | 443.3348 | 0.6333 | 700 | 0 | 0.0000 |
| R5 | low_steps.yaml | renoise | 700 | 367.0698 | 0.5244 | 700 | 442.2044 | 0.6317 | 700 | 0 | 0.0000 |
| R6 | high_cross_replace.yaml | ddim | 700 | 0.0000 | 0.0000 | 700 | 625.8638 | 0.8941 | 700 | 700 | 1.0000 |
| R6 | high_cross_replace.yaml | direct-inversion | 700 | 276.9834 | 0.3957 | 700 | 624.2400 | 0.8918 | 700 | 0 | 0.0000 |
| R6 | high_cross_replace.yaml | null-text | 700 | 3547.3440 | 5.0676 | 700 | 625.7836 | 0.8940 | 700 | 0 | 0.0000 |
| R6 | high_cross_replace.yaml | renoise | 700 | 543.1941 | 0.7760 | 700 | 624.2852 | 0.8918 | 700 | 0 | 0.0000 |
| R7 | low_editing_guidance.yaml | ddim | 700 | 291.5210 | 0.4165 | 700 | 613.3622 | 0.8762 | 700 | 0 | 0.0000 |
| R7 | low_editing_guidance.yaml | direct-inversion | 700 | 411.0497 | 0.5872 | 700 | 615.2670 | 0.8790 | 700 | 0 | 0.0000 |
| R7 | low_editing_guidance.yaml | null-text | 700 | 3784.4371 | 5.4063 | 700 | 618.4793 | 0.8835 | 700 | 0 | 0.0000 |
| R7 | low_editing_guidance.yaml | renoise | 700 | 540.1705 | 0.7717 | 700 | 616.6050 | 0.8809 | 700 | 0 | 0.0000 |
| R8 | high_self_replace.yaml | ddim | 700 | 0.0000 | 0.0000 | 700 | 595.2592 | 0.8504 | 700 | 700 | 1.0000 |
| R8 | high_self_replace.yaml | direct-inversion | 700 | 0.0000 | 0.0000 | 700 | 595.5905 | 0.8508 | 700 | 700 | 1.0000 |
| R8 | high_self_replace.yaml | null-text | 700 | 0.0000 | 0.0000 | 700 | 597.7276 | 0.8539 | 700 | 700 | 1.0000 |
| R8 | high_self_replace.yaml | renoise | 700 | 0.0000 | 0.0000 | 700 | 593.1437 | 0.8473 | 700 | 700 | 1.0000 |
| R9 | low_self_replace.yaml | ddim | 700 | 0.0000 | 0.0000 | 700 | 595.1434 | 0.8502 | 700 | 700 | 1.0000 |
| R9 | low_self_replace.yaml | direct-inversion | 700 | 0.0000 | 0.0000 | 700 | 595.5696 | 0.8508 | 700 | 700 | 1.0000 |
| R9 | low_self_replace.yaml | null-text | 700 | 0.0000 | 0.0000 | 700 | 597.1530 | 0.8531 | 700 | 700 | 1.0000 |
| R9 | low_self_replace.yaml | renoise | 700 | 0.0000 | 0.0000 | 700 | 594.9516 | 0.8499 | 700 | 700 | 1.0000 |
