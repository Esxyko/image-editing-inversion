"""Register adapter factories and discover installed entry points."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable

from ..artifacts.layout import method_directory_name
from .base import InversionMethod


MethodFactory = Callable[[], InversionMethod]
_METHOD_FACTORIES: dict[str, MethodFactory] = {}
_DISCOVERED_ENTRY_POINTS: set[tuple[str, str]] = set()


def register_method(method_id: str, factory: MethodFactory) -> MethodFactory:
    """Register a zero-argument factory; execution instances belong to a workflow."""
    method_directory_name(method_id)
    if isinstance(factory, InversionMethod) or not callable(factory):
        raise TypeError("factory must be a zero-argument class or callable, not a method instance")
    if isinstance(factory, type) and not issubclass(factory, InversionMethod):
        raise TypeError("method classes must subclass InversionMethod")
    if method_id in _METHOD_FACTORIES:
        raise ValueError(f"Inversion method {method_id!r} is already registered")
    _METHOD_FACTORIES[method_id] = factory
    return factory


def get_method(method_id: str) -> InversionMethod:
    """Create a fresh adapter and verify its declared ID."""
    try:
        factory = _METHOD_FACTORIES[method_id]
    except KeyError as exc:
        raise KeyError(
            f"Inversion method {method_id!r} is not registered; install or implement its adapter first"
        ) from exc
    try:
        method = factory()
        if not isinstance(method, InversionMethod):
            raise TypeError("factory must return InversionMethod")
        if method.method_id != method_id:
            raise ValueError(f"Factory ID {method_id!r} does not match method ID {method.method_id!r}")
    except Exception as exc:
        raise RuntimeError(f"Cannot construct inversion method {method_id!r}: {exc}") from exc
    return method


def registered_methods() -> tuple[str, ...]:
    """List registered IDs without discovery or adapter construction."""
    return tuple(sorted(_METHOD_FACTORIES))


def discover_methods() -> tuple[str, ...]:
    """Discover zero-argument classes/factories without constructing adapters."""
    group = "image_editing_inversion.methods"
    for entry_point in sorted(entry_points(group=group), key=lambda item: (item.name, item.value)):
        identity = (entry_point.name, entry_point.value)
        if identity in _DISCOVERED_ENTRY_POINTS:
            continue
        try:
            register_method(entry_point.name, entry_point.load())
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load inversion method entry point {entry_point.name!r} ({entry_point.value}): {exc}"
            ) from exc
        _DISCOVERED_ENTRY_POINTS.add(identity)
    return registered_methods()
