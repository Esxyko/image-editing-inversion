"""Content identity for the files a saved dataset actually reads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, Image


class DatasetContentIdentity:
    """Hash saved shards and external image bytes without decoding images."""

    def __init__(self, dataset: Dataset, root: Path) -> None:
        self._dataset = dataset
        self._root = root

    def digest(self) -> str:
        manifest: dict[str, Any] = {
            "features": self._dataset.features.to_dict(),
            "format": self._dataset.format,
        }
        hashed: dict[Path, str] = {}

        def include(name: str, path: Path) -> None:
            resolved = path.resolve(strict=True)
            if resolved not in hashed:
                with resolved.open("rb") as stream:
                    hashed[resolved] = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest[name] = hashed[resolved]

        # Dataset descriptions/citations and JSON formatting are not computation
        # inputs. Keep the effective features/format, and use state to find shards.
        state = json.loads((self._root / "state.json").read_text(encoding="utf-8"))
        for index, entry in enumerate(state["_data_files"]):
            include(f"shards/{index}", self._root / entry["filename"])
        # In-memory loading has no cache_files; state still identifies its shards.
        for index, entry in enumerate(self._dataset.cache_files):
            if Path(entry["filename"]).resolve() in hashed:
                continue
            include(f"loaded_shards/{index}", Path(entry["filename"]))

        # Select only image columns before iterating; decode=False leaves the
        # Arrow bytes/path records intact, including externally referenced files.
        columns = [name for name, feature in self._dataset.features.items()
                   if name == "source_img" and isinstance(feature, Image)]
        if columns:
            images = self._dataset.select_columns(columns).with_format(None)
            for name in columns:
                images = images.cast_column(name, Image(decode=False))
            for index, sample in enumerate(images):
                for name in columns:
                    image = sample[name]
                    if image is not None and image["bytes"] is None:
                        path = image["path"]
                        if not isinstance(path, str) or not path:
                            raise ValueError(f"Missing external image path in {name}, row {index}")
                        include(f"images/{index}/{name}", Path(path))
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
