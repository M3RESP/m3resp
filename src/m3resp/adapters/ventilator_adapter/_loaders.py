"""Registry mapping a file extension to a ventilator-file reader.

Loading otherwise dispatches between two known sources by file suffix - the
EIT ``*.bin`` and the multi-channel sEMG export (see `_core.resolve_ventilator_source`).
A third kind of file, in a format this module knows nothing about, previously
needed a caller to hand-write ``loader=``/``eit_loader=`` on every
`~m3resp.adapters.ventilator_adapter.VentilatorAdapter`/``M3Session``
construction that reads it. This registry lets that be declared once instead:
register a reader for an extension, and every subsequent load of that
extension uses it automatically - the same shape as `register_channel_alias`
letting a new vendor's channel naming be declared once rather than patched
into source.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

#: ``normalized extension -> (loader, source)``. Empty by default: no
#: third-party format is assumed. `source` is one of ``"eit"``/``"emg"``/
#: ``"ventilator"`` (see `_core.resolve_ventilator_source`) and decides which
#: clock the loaded channels are treated as sharing for synchronization.
_VENTILATOR_LOADERS: dict[str, tuple[Callable[..., Any], str]] = {}

_VALID_SOURCES = frozenset({"eit", "emg", "ventilator"})


def _normalize_extension(extension: str) -> str:
    ext = extension.strip().lower()
    return ext if ext.startswith(".") else f".{ext}"


def register_ventilator_loader(
    extension: str,
    loader: Callable[..., Any],
    *,
    source: str = "ventilator",
) -> None:
    """Register a reader for ventilator files with the given extension.

    ``loader(path, **kwargs)`` should return the ``{"array", "metadata"}``
    payload shape the built-in sEMG-file path produces, unless
    ``source="eit"``, in which case it should return an
    `eitprocessing`-Sequence-shaped object (anything with ``.continuous_data``)
    matching the EIT path - `VentilatorAdapter.load` unpacks it the same way
    either source arrived.

    `source` decides which clock the loaded channels are treated as sharing
    (see `_core.resolve_ventilator_source`): ``"eit"``/``"emg"`` for a format
    that is itself carried inside that modality's own file, ``"ventilator"``
    (the default) for a standalone export whose alignment is independent. Get
    this wrong and synchronization will crop the channels against the wrong
    clock, so default to ``"ventilator"`` unless the format is genuinely
    hosted inside an EIT or sEMG file.

    A registered extension is read by this loader whenever `VentilatorAdapter`
    is asked to load a file with that extension and no explicit ``source=``
    was passed - an explicit ``source=`` bypasses the registry and forces the
    corresponding built-in path instead, the same way an explicit channel
    index overrides label-based resolution in `split_channels`.

    The registration lasts for the current process, is not tied to any one
    `VentilatorAdapter` instance, and is not persisted - call this once at
    startup (or wherever the format is first needed) rather than per session.
    """

    if source not in _VALID_SOURCES:
        raise ValueError(
            f"Ventilator loader `source` must be one of {sorted(_VALID_SOURCES)}, "
            f"got {source!r}."
        )
    _VENTILATOR_LOADERS[_normalize_extension(extension)] = (loader, source)


def registered_ventilator_loader(path: Any) -> tuple[Callable[..., Any], str] | None:
    """The registered ``(loader, source)`` for `path`'s extension, if any."""

    return _VENTILATOR_LOADERS.get(Path(str(path)).suffix.lower())


def ventilator_loaders() -> dict[str, str]:
    """A copy of the currently registered ``extension -> source`` map."""

    return {extension: source for extension, (_, source) in _VENTILATOR_LOADERS.items()}


def unregister_ventilator_loader(extension: str) -> None:
    """Remove one registered extension, if present."""

    _VENTILATOR_LOADERS.pop(_normalize_extension(extension), None)


def reset_ventilator_loaders() -> None:
    """Remove every registered loader."""

    _VENTILATOR_LOADERS.clear()
