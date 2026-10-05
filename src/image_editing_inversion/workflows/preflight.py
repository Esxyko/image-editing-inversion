"""Validate the complete method/parameter sweep before sample processing."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from ..config import ConfigError, ExperimentConfig
from ..inversion import InversionContext, InversionMethod
from .output import SweepOutput
from .replay import ReplayPreparer


class SweepPreflight:
    """Separate model-free validation from actual scheduler validation."""

    def __init__(
        self, configs: Sequence[tuple[Path, ExperimentConfig]],
        methods: Sequence[InversionMethod], output: SweepOutput,
    ) -> None:
        self._configs = configs
        self._methods = methods
        self._sweep_output = output

    def configuration(self) -> None:
        errors: list[tuple[Path, str, str]] = []
        for path, config in self._configs:
            for method in self._methods:
                for check in (
                    lambda: method.validate_inversion_config(config),
                    lambda: ReplayPreparer.check_batch_size(method, config),
                    lambda: ReplayPreparer.policy_for(method, config),
                ):
                    try:
                        check()
                    except Exception as exc:
                        errors.append((path, method.method_id, str(exc)))
        if errors:
            self._fail(errors)

    def runtime(
        self, activate: Callable[[Path, ExperimentConfig], None],
        context_for: Callable[[], InversionContext],
    ) -> None:
        first_path, first_config = self._configs[0]
        try:
            activate(first_path, first_config)
            context_for()
        except Exception as exc:
            self._fail([(path, method.method_id, str(exc))
                        for path, _ in self._configs for method in self._methods])
        errors: list[tuple[Path, str, str]] = []
        for path, config in self._configs:
            try:
                activate(path, config)
                context = context_for()
            except Exception as exc:
                errors.extend((path, method.method_id, str(exc)) for method in self._methods)
                continue
            for method in self._methods:
                try:
                    method.validate_inversion_context(context)
                except Exception as exc:
                    errors.append((path, method.method_id, str(exc)))
        if errors:
            self._fail(errors)
        try:
            activate(first_path, first_config)
        except Exception as exc:
            self._fail([(first_path, method.method_id, str(exc)) for method in self._methods])

    def _fail(self, errors: Sequence[tuple[Path, str, str]]) -> None:
        reasons: dict[tuple[str, str], list[str]] = {}
        for path, method_id, reason in errors:
            reasons.setdefault((path.name, method_id), []).append(reason)
        message = "Inversion preflight failed:\n" + "\n".join(
            f"- {filename} / {method_id}: {'; '.join(messages)}"
            for (filename, method_id), messages in reasons.items()
        )
        error = ConfigError(message)
        counts = {"ok": 0, "skipped": 0, "error": 0}
        try:
            for (filename, method_id), messages in reasons.items():
                self._sweep_output.update_method(filename, method_id, "error", counts, "; ".join(messages))
            for filename in dict.fromkeys(filename for filename, _ in reasons):
                self._sweep_output.update(filename, "error", counts, message)
            self._sweep_output.finish("error", message)
        except BaseException as recording:
            error.add_note(f"Could not record preflight errors: {recording}")
        raise error
