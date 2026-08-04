"""Standalone signal-shaping helpers used by the ReSurfEMGAdapter and by
EMG pipeline steps that need the same ventilator-channel/event-index shaping."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from m3resp.core.events import BreathEvent
from m3resp.core.exceptions import OptionalDependencyError


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
        raise OptionalDependencyError("EMG postprocessing requires numpy.") from exc

    metadata = ventilator.get("metadata", {}) if isinstance(ventilator, dict) else {}
    array = ventilator.get("array") if isinstance(ventilator, dict) else ventilator
    if array is None:
        raise TypeError("Ventilator postprocessing input needs an array.")

    vent_fs = fs if fs is not None else metadata.get("fs")
    if vent_fs is None:
        raise TypeError("Ventilator postprocessing input needs a sampling rate.")

    array = np.asarray(array, dtype=float)
    return {
        "pressure": np.asarray(array[pressure_channel], dtype=float),
        "flow": np.asarray(array[flow_channel], dtype=float),
        "volume": np.asarray(array[volume_channel], dtype=float),
        "fs": float(vent_fs),
        "metadata": metadata,
    }


def peak_indices_from_events(
    events: Sequence[BreathEvent] | None, fs: float
) -> list[int]:
    if events is None:
        return []
    return [
        int(event.peak_time * fs) for event in events if event.peak_time is not None
    ]
