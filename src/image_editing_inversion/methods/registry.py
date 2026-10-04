"""Register inversion adapters and discover installed entry points."""

from __future__ import annotations

from importlib.metadata import entry_points

from .base import InversionMethod


_METHOD_REGISTRY: dict[str, InversionMethod] = {}
_DISCOVERED_ENTRY_POINTS: set[tuple[str, str]] = set()


def register_method(method: InversionMethod) -> InversionMethod:
    """Register one inversion method instance by its stable ID."""
    if not isinstance(method, InversionMethod):
        raise TypeError("method must implement InversionMethod")
    method_id = method.method_id
    if not isinstance(method_id, str) or not method_id.strip():
        raise ValueError("method_id must be a non-empty string")
    if method_id in _METHOD_REGISTRY:
        raise ValueError(f"Inversion method {method_id!r} is already registered")
    _METHOD_REGISTRY[method_id] = method
    return method


def get_method(method_id: str) -> InversionMethod:
    """Return a registered method or raise a clear error for future adapters."""
    try:
        return _METHOD_REGISTRY[method_id]
    except KeyError as exc:
        raise KeyError(
            f"Inversion method {method_id!r} is not registered; "
            "install or implement its adapter first"
        ) from exc


def registered_methods() -> tuple[str, ...]:
    """List available adapter IDs without importing any future implementations."""
    return tuple(sorted(_METHOD_REGISTRY))


def discover_methods() -> tuple[str, ...]:
    """Load installed adapters from ``image_editing_inversion.methods``.

    Each entry point must be named for its method ID and export either an
    ``InversionMethod`` instance or a zero-argument subclass. Repeated calls
    do not reload or reregister entry points already discovered.
    """
    group = "image_editing_inversion.methods"
    for entry_point in entry_points(group=group):
        identity = (entry_point.name, entry_point.value)
        if identity in _DISCOVERED_ENTRY_POINTS:
            continue
        try:
            exported = entry_point.load()
            method = exported() if isinstance(exported, type) else exported
            if not isinstance(method, InversionMethod):
                raise TypeError(
                    "entry point must export an InversionMethod instance "
                    "or zero-argument subclass"
                )
            if method.method_id != entry_point.name:
                raise ValueError(
                    f"entry point name {entry_point.name!r} does not match "
                    f"method ID {method.method_id!r}"
                )
            register_method(method)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load inversion method entry point "
                f"{entry_point.name!r} ({entry_point.value}): {exc}"
            ) from exc
        _DISCOVERED_ENTRY_POINTS.add(identity)
    return registered_methods()
