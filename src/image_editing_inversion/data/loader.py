"""Fetch and load the project's image-editing dataset."""

import os

from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv
from huggingface_hub import get_token

from .._project import project_paths
from .schema import HUB_DATASET_REF, REQUIRED_COLUMNS


def load_project_dataset() -> Dataset:
    """Return the train dataset, downloading it once into the project data root."""
    paths = project_paths()
    data_dir = paths.data
    state_path = data_dir / "state.json"
    info_path = data_dir / "dataset_info.json"

    if state_path.exists() or info_path.exists():
        if not (state_path.is_file() and info_path.is_file()):
            raise RuntimeError(
                f"Incomplete dataset at {data_dir}: state.json and "
                "dataset_info.json must both be present. Repair the dataset files "
                "before loading; keep the other data/ folders."
            )
        try:
            dataset = load_from_disk(str(data_dir))
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load the saved dataset at {data_dir}. Its files may be "
                "incomplete or damaged; keep the other data/ folders when repairing it."
            ) from exc
    else:
        if (data_dir / "dataset_dict.json").exists():
            raise RuntimeError(
                f"A different Hugging Face dataset artifact exists at {data_dir}; "
                "the project requires a single train Dataset there."
            )

        load_dotenv(dotenv_path=paths.env, override=False)
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
            dataset.save_to_disk(str(data_dir))
            # Use the same saved shards on the first and subsequent runs.
            dataset = load_from_disk(str(data_dir))
        except Exception as exc:
            raise RuntimeError(
                f"Could not save the dataset to {data_dir}. The local artifact "
                "may be incomplete; keep the other data/ folders when repairing it."
            ) from exc

    if not isinstance(dataset, Dataset) or set(dataset.column_names) != REQUIRED_COLUMNS:
        raise RuntimeError(
            f"The dataset at {data_dir} does not have the expected project columns."
        )
    return dataset
