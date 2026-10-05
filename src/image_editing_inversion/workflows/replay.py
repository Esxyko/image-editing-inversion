"""Load and prepare saved inversions for either workflow."""

from __future__ import annotations

import json
import math
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from ..artifacts import (
    ArtifactCollection, ArtifactCompatibilityError, ArtifactReference,
    ArtifactSettings, InversionArtifact,
)
from ..artifacts.layout import pipeline_group_parameters
from ..config import ExperimentConfig
from ..data import DatasetRepository
from ..generation import PromptToPrompt
from ..inversion import DenoisingHook, InversionContext, InversionMethod, get_method
from .models import LoadedArtifact, LoadResult, PreparedEdit


class ReplayPreparer:
    """Own collection reads, dataset identity, policy selection, and hook preparation."""

    def __init__(self, repository: DatasetRepository) -> None:
        self._repository = repository
        self._collections: dict[Path, tuple[tuple[int, ...], ArtifactCollection]] = {}
        self._lock = Lock()

    def invalidate(self, path: Path) -> None:
        with self._lock:
            self._collections.pop(path, None)

    def load(self, reference: ArtifactReference, settings: ArtifactSettings) -> LoadedArtifact:
        stat = reference.path.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        with self._lock:
            cached = self._collections.get(reference.path)
            if cached is None or cached[0] != identity:
                cached = (identity, ArtifactCollection(reference.path))
                self._collections[reference.path] = cached
        artifact = cached[1].load(reference, settings=settings)
        self.check_identity(artifact)
        return LoadedArtifact(artifact, reference, self._repository.sample(artifact.sample_uid))

    def load_result(self, reference: ArtifactReference, settings: ArtifactSettings) -> LoadResult:
        try:
            return LoadResult(reference, loaded=self.load(reference, settings))
        except Exception as exc:
            return LoadResult(reference, error=exc)

    def check_identity(self, artifact: InversionArtifact) -> None:
        self._repository.validate_identity(
            artifact.dataset_ref, artifact.dataset_fingerprint, artifact.sample_uid
        )

    @staticmethod
    def method_for(artifact: InversionArtifact) -> InversionMethod:
        try:
            return get_method(artifact.method_id)
        except KeyError as exc:
            raise ValueError(
                f"Artifact method {artifact.method_id!r} is not registered; "
                "install its adapter before replay"
            ) from exc

    @staticmethod
    def check_batch_size(method: InversionMethod, config: ExperimentConfig) -> None:
        limit = method.max_edit_batch_size
        if limit is None:
            return
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"Method {method.method_id!r} has an invalid max_edit_batch_size")
        if config.runtime.batch_size > limit:
            raise ValueError(
                f"Method {method.method_id!r} supports edit batches of at most "
                f"{limit}; configured batch_size is {config.runtime.batch_size}"
            )

    @staticmethod
    def policy_for(
        method: InversionMethod, config: ExperimentConfig,
    ) -> tuple[type[PromptToPrompt], dict[str, Any]]:
        selected = method.prompt_to_prompt_class
        policy_class = PromptToPrompt if selected is None else selected
        if not isinstance(policy_class, type) or not issubclass(policy_class, PromptToPrompt):
            raise TypeError(
                f"Method {method.method_id!r} prompt_to_prompt_class must subclass PromptToPrompt"
            )
        try:
            settings = policy_class.validate_method_settings(
                config.method_prompt_to_prompt(method.method_id)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid Prompt-to-Prompt settings for method {method.method_id!r}: {exc}"
            ) from exc
        if not isinstance(settings, Mapping) or any(not isinstance(key, str) for key in settings):
            raise TypeError(f"Method {method.method_id!r} returned invalid Prompt-to-Prompt settings")
        try:
            normalized = json.loads(json.dumps(dict(settings), sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Method {method.method_id!r} Prompt-to-Prompt settings must be JSON-compatible"
            ) from exc
        return policy_class, normalized

    def validate(
        self, artifact: InversionArtifact, method: InversionMethod, context: InversionContext,
    ) -> None:
        self.check_identity(artifact)
        artifact.validate_terminal_latent()
        if artifact.provenance is not None:
            for name, expected in (
                ("model_content", context.editor.model_content_identity),
                ("dataset_content", context.dataset_content),
                ("numerics", context.editor.inversion_cache_settings()),
            ):
                if getattr(artifact.provenance, name) != expected:
                    raise ArtifactCompatibilityError(f"Artifact {name} does not match this run")
            guidance = artifact.provenance.guidance_scale
            if guidance is not None and not math.isclose(
                guidance, context.config.sampling.guidance_scale, rel_tol=0, abs_tol=1e-9,
            ):
                raise ArtifactCompatibilityError("Artifact production guidance does not match this run")
        method.validate_replay(artifact, context)
        context.editor.validate_artifact(artifact)

    def prepare(self, loaded: LoadedArtifact, context: InversionContext) -> PreparedEdit:
        method = self.method_for(loaded.artifact)
        self.check_batch_size(method, context.config)
        policy_class, settings = self.policy_for(method, context.config)
        self.validate(loaded.artifact, method, context)
        parameters = pipeline_group_parameters(loaded.reference.path.parent.name)
        if not math.isclose(
            parameters["guidance_scale"], context.config.sampling.guidance_scale,
            rel_tol=0, abs_tol=1e-9,
        ):
            raise ArtifactCompatibilityError("Artifact group guidance scale does not match this run")
        hook = method.create_denoising_hook(loaded.artifact)
        if loaded.artifact.requires_denoising_hook and hook is None:
            raise ValueError(f"Artifact method {method.method_id!r} needs a denoising hook.")
        if hook is not None and not isinstance(hook, DenoisingHook):
            raise TypeError("create_denoising_hook must return DenoisingHook or None")
        return PreparedEdit(loaded, method, policy_class, settings, hook)
