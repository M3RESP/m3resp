"""Standalone signal-shaping helpers used by the ReSurfEMGAdapter and by
EMG pipeline steps that need the same ventilator-channel/event-index shaping."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from m3resp.adapters.ventilator_adapter import DEFAULT_CHANNELS, split_channels
from m3resp.core.events import BreathEvent


def ventilator_signals(
    ventilator: Any | None,
    *,
    pressure_channel: int | None = None,
    flow_channel: int | None = None,
    volume_channel: int | None = None,
    channels: Any = DEFAULT_CHANNELS,
    fs: float | None = None,
) -> dict[str, Any] | None:
    """Split the ventilator channels an EMG postprocessing run needs.

    Delegates to `m3resp.adapters.ventilator_adapter.split_channels` rather
    than indexing columns itself, so the ventilator channels reaching Pocc and
    ventilator-breath detection are found by the same name resolution used
    everywhere else. A recording that labels its channels is read by those
    labels; an unlabelled array still falls back to fixed columns.
    """

    if ventilator is None:
        return None

    metadata = ventilator.get("metadata", {}) if isinstance(ventilator, dict) else {}
    array = ventilator.get("array") if isinstance(ventilator, dict) else ventilator
    if array is None:
        raise TypeError("Ventilator postprocessing input needs an array.")

    if fs is None and not metadata.get("fs"):
        raise TypeError("Ventilator postprocessing input needs a sampling rate.")

    return split_channels(
        {"array": array, "metadata": metadata},
        channels=channels,
        pressure_channel=pressure_channel,
        flow_channel=flow_channel,
        volume_channel=volume_channel,
        fs=fs,
    )


def peak_indices_from_events(
    events: Sequence[BreathEvent] | None, fs: float
) -> list[int]:
    if events is None:
        return []
    return [
        int(event.peak_time * fs) for event in events if event.peak_time is not None
    ]
