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

    After the registration, ``VentilatorAdapter.load`` and
    ``M3Session.load_ventilator`` read every file with this extension with
    ``loader``. The registration holds for every session until Python stops.
    With ``source="eit"`` or ``source="emg"`` in the load call, the built-in
    readers read the file.

    For ``source="ventilator"`` or ``"emg"``, ``loader(path, **kwargs)``
    returns ``{"array": ..., "metadata": {"fs": ..., "labels": [...],
    "units": [...]}}``: the samples with one row per channel, the samples per
    second in Hz, and one label and one unit per row. The labels give the
    channels, for example ``"Paw"`` the airway pressure. For
    ``source="eit"``, it returns an ``EITRecording`` or ``EITFileData`` with
    the ventilator waveforms in ``signals``.

    Example:
        ``register_ventilator_loader(".csv", read_ventilator_csv)``. The docs
        guide "How to read a ventilator file in another format" gives
        ``read_ventilator_csv``.

    Args:
        extension (str): The file extension, with or without the dot, in
            upper or lower case.
        loader (Callable[..., Any]): The function that reads one file.
        source (str): The clock of the channels. ``"ventilator"`` (default):
            the file has its own clock. ``"eit"`` or ``"emg"``: the format
            holds the ventilator data of an EIT or EMG recording, and the
            channels have the clock of that recording.

    Raises:
        ValueError: If ``source`` is not ``"eit"``, ``"emg"`` or
            ``"ventilator"``.
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
