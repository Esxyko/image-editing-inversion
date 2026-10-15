"""Pinned CLIP and masked image metrics, sharing model and text-feature caches."""

from __future__ import annotations

import importlib.metadata
import math
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import lpips
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from torchmetrics.functional.image import structural_similarity_index_measure
    from transformers import CLIPModel, CLIPProcessor
except ImportError as exc:
    raise ImportError("Archive evaluation requires the evaluation extra: uv sync --extra evaluation") from exc

from .models import CLIP_MODEL, CLIP_REVISION, EvaluationSettings, require


class MetricCalculator:
    """Keep float32 metric models alive across every archive in an invocation."""

    def __init__(self, settings: EvaluationSettings, cache: Path) -> None:
        self.settings = settings
        self.cache = cache
        self.device = torch.device(settings.device)
        require(self.device.type in ("cpu", "cuda", "mps"), "Evaluation device must be cpu, cuda[:index], or mps")
        if self.device.type == "cuda":
            require(torch.cuda.is_available(),
                    "CUDA is unavailable; select --device cpu or install a compatible PyTorch build")
        if self.device.type == "mps":
            require(torch.backends.mps.is_available(), "MPS is unavailable")
        self._text: dict[str, torch.Tensor] = {}
        self._truncated: dict[str, bool] = {}
        self._backend_flags: tuple[bool, bool] | None = None

    def __enter__(self) -> MetricCalculator:
        self._backend_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            self.processor = CLIPProcessor.from_pretrained(
                CLIP_MODEL, revision=CLIP_REVISION, cache_dir=str(self.cache / "huggingface"),
            )
            self.clip = CLIPModel.from_pretrained(
                CLIP_MODEL, revision=CLIP_REVISION, cache_dir=str(self.cache / "huggingface"),
                dtype=torch.float32, use_safetensors=True,
            ).to(self.device).eval()
            require(self.clip.config.vision_config.image_size == 224
                    and self.clip.config.vision_config.patch_size == 14
                    and self.clip.config.projection_dim == 768, "Expected CLIP ViT-L/14 at 224px")
            original_hub = torch.hub.get_dir()
            try:
                torch.hub.set_dir(str(self.cache / "torch" / "hub"))
                self.perceptual = lpips.LPIPS(
                    net="squeeze", version="0.1", spatial=False, verbose=False,
                ).to(device=self.device, dtype=torch.float32).eval()
            finally:
                torch.hub.set_dir(original_hub)
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *args: Any) -> None:
        if self._backend_flags is not None:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self._backend_flags
            self._backend_flags = None
        self._text.clear()
        for name in ("clip", "perceptual", "processor"):
            if hasattr(self, name):
                delattr(self, name)

    @staticmethod
    def _features(result: Any) -> torch.Tensor:
        values = result if isinstance(result, torch.Tensor) else result.pooler_output
        require(values.ndim == 2 and values.shape[1] == 768 and torch.isfinite(values).all().item(),
                "Invalid CLIP projected features")
        return F.normalize(values.float(), dim=-1)

    def _text_features(self, prompts: list[str]) -> torch.Tensor:
        require(all(isinstance(prompt, str) and bool(prompt.strip()) for prompt in prompts), "Empty target prompt")
        missing = list(dict.fromkeys(prompt for prompt in prompts if prompt not in self._text))
        for start in range(0, len(missing), self.settings.batch_size):
            chunk = missing[start:start + self.settings.batch_size]
            lengths = self.processor.tokenizer(chunk, truncation=False, padding=False, verbose=False)["input_ids"]
            tokens = self.processor.tokenizer(
                chunk, padding=True, truncation=True, max_length=77, return_tensors="pt",
            ).to(self.device)
            features = self._features(self.clip.get_text_features(**tokens))
            for prompt, ids, feature in zip(chunk, lengths, features, strict=True):
                self._text[prompt] = feature
                self._truncated[prompt] = len(ids) > 77
        return torch.stack([self._text[prompt] for prompt in prompts])

    def _similarity(self, pixels: torch.Tensor, text: torch.Tensor) -> list[float]:
        images = [Image.fromarray(pixel.permute(1, 2, 0).numpy()) for pixel in pixels]
        try:
            inputs = self.processor.image_processor(images=images, return_tensors="pt").to(self.device)
            values = (self._features(self.clip.get_image_features(**inputs)) * text).sum(-1)
            require(torch.isfinite(values).all().item(), "Nonfinite CLIP similarity")
            return values.cpu().tolist()
        finally:
            for image in images:
                image.close()

    @torch.inference_mode()
    def score(self, pixels: torch.Tensor, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        # Honor the protocol even when the Python caller has enabled autocast.
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self._score(pixels, samples)

    def _score(self, pixels: torch.Tensor, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        require(len(samples) == len(pixels) > 0, "Misaligned evaluation batch")
        sources, edit_masks, prompts = [], [], []
        for sample, pixel in zip(samples, pixels, strict=True):
            source_image = sample["source_img"].convert("RGB")
            mask_image = sample["mask_img"].convert("L")
            try:
                source = np.array(source_image, copy=True)
                mask = np.array(mask_image, copy=True)
            finally:
                source_image.close()
                mask_image.close()
            require(tuple(source.shape) == (pixel.shape[1], pixel.shape[2], 3)
                    and mask.shape == source.shape[:2], "Source, mask and edited image dimensions differ")
            require(set(np.unique(mask).tolist()) <= {0, 255}, "Expected binary edit mask (0 or 255)")
            sources.append(torch.from_numpy(source).permute(2, 0, 1))
            edit_masks.append(torch.from_numpy(mask == 255)[None])
            prompts.append(sample["target_prompt"])
        source_pixels = torch.stack(sources)
        edit = torch.stack(edit_masks)
        background = ~edit
        edit_counts = edit.sum((1, 2, 3)).tolist()
        background_counts = background.sum((1, 2, 3)).tolist()
        text = self._text_features(prompts)
        global_scores = self._similarity(pixels, text)
        region_scores: list[float | None] = [None] * len(samples)
        region_indices = [i for i, count in enumerate(edit_counts) if count]
        if region_indices:
            scores = self._similarity(pixels[region_indices] * edit[region_indices], text[region_indices])
            for i, value in zip(region_indices, scores, strict=True):
                region_scores[i] = value
        rows = [{
            "uid": sample["uid"], "target_prompt": prompt,
            "text_truncated": self._truncated[prompt],
            "background_pixels": background_counts[i], "edited_region_pixels": edit_counts[i],
            "clip_text_similarity": global_scores[i], "edited_region_clip": region_scores[i],
            "masked_psnr_db": None, "masked_ssim": None, "masked_lpips": None,
        } for i, (sample, prompt) in enumerate(zip(samples, prompts, strict=True))]
        valid = [i for i, count in enumerate(background_counts) if count]
        # Full-resolution SSIM/LPIPS use smaller chunks to bound temporary memory.
        for start in range(0, len(valid), 4):
            indices = valid[start:start + 4]
            masks = background[indices].to(self.device, dtype=torch.float32)
            source = source_pixels[indices].to(self.device, dtype=torch.float32).div_(255).mul_(masks)
            edited = pixels[indices].to(self.device, dtype=torch.float32).div_(255).mul_(masks)
            psnr = (-10 * torch.log10((source - edited).square().mean((1, 2, 3)))).cpu().tolist()
            ssim = structural_similarity_index_measure(
                edited, source, data_range=1.0, gaussian_kernel=True, kernel_size=11,
                sigma=1.5, k1=0.01, k2=0.03, reduction="none",
            ).reshape(-1).cpu().tolist()
            perceptual = self.perceptual(edited * 2 - 1, source * 2 - 1).reshape(-1).cpu().tolist()
            for i, p, s, l in zip(indices, psnr, ssim, perceptual, strict=True):
                require((math.isfinite(p) or p == math.inf) and math.isfinite(s) and math.isfinite(l),
                        "Invalid masked metric result")
                rows[i].update(masked_psnr_db="+inf" if p == math.inf else p,
                               masked_ssim=s, masked_lpips=l)
        return rows

    def metadata(self) -> dict[str, Any]:
        hardware = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else platform.machine()
        return {
            "clip": {"model": CLIP_MODEL, "revision": CLIP_REVISION, "input_resolution": 224,
                     "precision": "float32", "tf32": False, "max_text_tokens": 77,
                     "text_overflow": "truncate", "score": "raw cosine of normalized projected image/text features",
                     "preprocessing": self.processor.image_processor.to_dict()},
            "edited_region_clip": (
                "Keep mask_img==255; set every other RGB pixel to black before standard CLIP "
                "preprocessing; compare with target_prompt"
            ),
            "masked_background": (
                "Set mask_img==255 to black in BOTH source_img and edited; reduce over the full "
                "canvas, without background-area normalization"
            ),
            "psnr": {"data_range": 1.0, "formula": "-10*log10(mean((source_masked-edited_masked)^2))",
                     "perfect_match": "+inf"},
            "ssim": {"implementation": "torchmetrics functional", "gaussian_kernel": True, "kernel_size": 11,
                     "sigma": 1.5, "k1": 0.01, "k2": 0.03, "data_range": 1.0},
            "lpips": {"implementation": "official lpips", "net": "squeeze", "version": "0.1",
                      "backbone": "SqueezeNet 1.1 ImageNet1K V1", "spatial": False,
                      "input_range": "[-1,1] after RGB masking"},
            "empty_region_policy": "null score, excluded separately from each metric's statistics",
            "aggregation": (
                "arithmetic mean of per-image values; std is sample standard deviation; undefined std is null"
            ),
            "settings": {"device": str(self.device), "batch_size": self.settings.batch_size,
                         "background_batch_size": min(4, self.settings.batch_size)},
            "hardware": {"evaluation_device": hardware},
            "software_versions": {name: importlib.metadata.version(name) for name in (
                "torch", "torchvision", "transformers", "lpips", "torchmetrics", "datasets",
                "safetensors", "numpy", "pillow",
            )},
        }
