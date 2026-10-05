"""Identify model weight files and effective image/text processing settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ModelContentIdentity:
    """Inspect one resolved SD snapshot without hashing unused model formats."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _weights(self, component: str) -> list[Path]:
        folder = self.root / component
        for extension in ("safetensors", "bin"):
            if component == "text_encoder":
                stems = ("model",) if extension == "safetensors" else ("pytorch_model",)
            else:
                stems = ("diffusion_pytorch_model",)
            for stem in stems:
                path = folder / f"{stem}.{extension}"
                if path.is_file():
                    return [path]
                index = folder / f"{stem}.{extension}.index.json"
                if index.is_file():
                    data = json.loads(index.read_text(encoding="utf-8"))
                    names = sorted(set(data["weight_map"].values()))
                    paths = [folder / name for name in names]
                    # Hub snapshots normally symlink files to the blob store.
                    if any(Path(name).is_absolute() or len(Path(name).parts) != 1 for name in names):
                        raise ValueError(f"Model shard paths must stay in {folder}")
                    return [index, *paths]
        raise ValueError(f"No supported model weights found in {folder}")

    def describe(self, pipeline: Any) -> dict[str, Any]:
        files: dict[str, str] = {}
        for name in ("unet", "vae", "text_encoder"):
            for path in self._weights(name):
                if path.name.endswith(".index.json"):
                    weight_map = json.loads(path.read_text(encoding="utf-8"))["weight_map"]
                    content = json.dumps(weight_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    digest = hashlib.sha256(content).hexdigest()
                else:
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                files[path.relative_to(self.root).as_posix()] = digest
        # Tokenizer assets are small; exclude config version/path metadata below.
        tokenizer_folder = self.root / "tokenizer"
        for path in sorted(tokenizer_folder.iterdir()):
            if path.is_file() and path.suffix in {".json", ".txt", ".model"}:
                if path.suffix == ".json":
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if path.name == "tokenizer_config.json":
                        value = self._semantic(value)
                    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
                    files[f"tokenizer/{path.name}"] = hashlib.sha256(content).hexdigest()
                else:
                    with path.open("rb") as stream:
                        files[f"tokenizer/{path.name}"] = hashlib.file_digest(stream, "sha256").hexdigest()
        tokenizer = pipeline.tokenizer
        identity = {
            "files": files,
            "components": {name: self._semantic(self._config(getattr(pipeline, name).config))
                           for name in ("unet", "vae", "text_encoder")},
            "image_processor": self._semantic(dict(pipeline.image_processor.config)),
            "source_encoding": {"color_mode": "RGB", "vae_posterior": "mode"},
            "tokenizer": {
                "padding": "max_length", "truncation": False,
                "model_max_length": tokenizer.model_max_length,
                "padding_side": tokenizer.padding_side,
                "truncation_side": tokenizer.truncation_side,
                "special_tokens": tokenizer.special_tokens_map,
                "special_token_ids": tokenizer.all_special_ids,
            },
        }
        return json.loads(json.dumps(identity, allow_nan=False))

    @staticmethod
    def for_components(identity: dict[str, Any], components: tuple[str, ...]) -> dict[str, Any]:
        """Match only components used by a particular reusable calculation."""
        selected = {
            "files": {name: digest for name, digest in identity["files"].items()
                      if name.split("/", 1)[0] in components},
            "components": {name: config for name, config in identity["components"].items()
                           if name in components},
        }
        if "vae" in components:
            selected["image_processor"] = identity["image_processor"]
            selected["source_encoding"] = identity["source_encoding"]
        if "tokenizer" in components:
            selected["tokenizer"] = identity["tokenizer"]
        return selected

    @classmethod
    def _semantic(cls, value: Any) -> Any:
        """Remove provenance metadata while retaining effective configuration."""
        if isinstance(value, dict):
            return {key: cls._semantic(item) for key, item in value.items()
                    if key not in {
                        "_diffusers_version", "_use_default_values", "_name_or_path", "_commit_hash",
                        "_attn_implementation", "_attn_implementation_internal", "_attn_implementation_autoset",
                        "transformers_version", "torch_dtype", "dtype", "name_or_path",
                    }}
        if isinstance(value, (list, tuple)):
            return [cls._semantic(item) for item in value]
        return value

    @staticmethod
    def _config(config: Any) -> dict[str, Any]:
        return config.to_dict() if hasattr(config, "to_dict") else dict(config)
