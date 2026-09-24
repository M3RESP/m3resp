"""Standalone signal-shaping helpers used by the ReSurfEMGAdapter and by
EMG pipeline steps that need the same ventilator-channel/event-index shaping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from m3resp.core.exceptions import OptionalDependencyError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from m3resp.core.events import BreathEvent


#TODO
def ventilator_signals(
    ventilator: Any | None,
    *,
    pressure_channel: int,
    flow_channel: int,
    volume_channel: int,
    fs: float | None = None,
) -> dict[str, Any] | None:
    if ventilator is None:
        return None

    try:
        import numpy as np
    except ImportError as exc:
        msg = "EMG postprocessing requires numpy."
        raise OptionalDependencyError(msg) from exc

    metadata = ventilator.get("metadata", {}) if isinstance(ventilator, dict) else {}
    array = ventilator.get("array") if isinstance(ventilator, dict) else ventilator
    if array is None:
        msg = "Ventilator postprocessing input needs an array."
        raise TypeError(msg)

    vent_fs = fs if fs is not None else metadata.get("fs")
    if vent_fs is None:
        msg = "Ventilator postprocessing input needs a sampling rate."
        raise TypeError(msg)

    array = np.asarray(array, dtype=float)
    return {
        "pressure": np.asarray(array[pressure_channel], dtype=float),
        "flow": np.asarray(array[flow_channel], dtype=float),
        "volume": np.asarray(array[volume_channel], dtype=float),
        "fs": float(vent_fs),
        "metadata": metadata,
    }

#TODO
def peak_indices_from_events(
    events: Sequence[BreathEvent] | None, fs: float
) -> list[int]:
    if events is None:
        return []
    return [
        int(event.peak_time * fs) for event in events if event.peak_time is not None
    ]
