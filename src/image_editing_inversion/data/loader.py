"""Fetch and load the project's image-editing dataset."""

import os
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv
from huggingface_hub import get_token

from .schema import HUB_DATASET_REF, REQUIRED_COLUMNS


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _PROJECT_ROOT / "data"


def load_project_dataset() -> Dataset:
    """Return the train dataset, downloading it once into the project data root."""
    state_path = _DATA_DIR / "state.json"
    info_path = _DATA_DIR / "dataset_info.json"

    if state_path.exists() or info_path.exists():
        if not (state_path.is_file() and info_path.is_file()):
            raise RuntimeError(
                f"Incomplete dataset at {_DATA_DIR}: state.json and "
                "dataset_info.json must both be present. Repair the dataset files "
                "before loading; keep the other data/ folders."
            )
        try:
            dataset = load_from_disk(str(_DATA_DIR))
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load the saved dataset at {_DATA_DIR}. Its files may be "
                "incomplete or damaged; keep the other data/ folders when repairing it."
            ) from exc
    else:
        if (_DATA_DIR / "dataset_dict.json").exists():
            raise RuntimeError(
                f"A different Hugging Face dataset artifact exists at {_DATA_DIR}; "
                "the project requires a single train Dataset there."
            )

        load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)
        token = os.getenv("HF_TOKEN") or get_token()
        if not token:
            raise RuntimeError(
                f"{HUB_DATASET_REF} is private. Set HF_TOKEN in the environment or "
                "the project .env file, or log in with the Hugging Face CLI."
            )

        try:
            dataset = load_dataset(HUB_DATASET_REF, split="train", token=token)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {HUB_DATASET_REF}. Check dataset access, HF_TOKEN "
                "or Hugging Face login, and the network connection."
            ) from exc

        if not isinstance(dataset, Dataset) or set(dataset.column_names) != REQUIRED_COLUMNS:
            raise RuntimeError(
                f"Downloaded {HUB_DATASET_REF}, but its train split does not have the "
                "expected project columns. No dataset was saved."
            )

        try:
            dataset.save_to_disk(str(_DATA_DIR))
        except Exception as exc:
            raise RuntimeError(
                f"Could not save the dataset to {_DATA_DIR}. The local artifact "
                "may be incomplete; keep the other data/ folders when repairing it."
            ) from exc

    if not isinstance(dataset, Dataset) or set(dataset.column_names) != REQUIRED_COLUMNS:
        raise RuntimeError(
            f"The dataset at {_DATA_DIR} does not have the expected project columns."
        )
    return dataset
