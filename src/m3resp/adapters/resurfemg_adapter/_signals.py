"""Standalone signal-shaping helpers used by the ReSurfEMGAdapter and by
EMG workflow steps that need the same ventilator-channel/event-index shaping."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from m3resp.adapters.ventilator_adapter import DEFAULT_CHANNELS, split_channels
from m3resp.data.events import BreathEvent


def ventilator_signals(
    ventilator: Any | None,
    *,
    airway_pressure_channel: int | None = None,
    flow_channel: int | None = None,
    volume_channel: int | None = None,
    channels: Any = DEFAULT_CHANNELS,
    fs: float | None = None,
) -> dict[str, Any] | None:
    """Extract the ventilator channels used for EMG postprocessing.

    Args:
        ventilator (Any | None): Dict with array and metadata, or a numeric array
            with shape (channels, samples). None indicates unavailable data.
        airway_pressure_channel (int | None): Explicit zero-based pressure index.
        flow_channel (int | None): Explicit zero-based flow index.
        volume_channel (int | None): Explicit zero-based volume index.
        channels (Any): Names to extract; defaults to airway_pressure, flow and
            volume. Labels and fallback positions follow `split_channels`.
        fs (float | None): Sampling rate in Hz; overrides metadata fs.

    Returns:
        dict[str, Any] | None: Channel bundle from `split_channels`, including
            per-channel units and fs in Hz, or None when ventilator is None.

    Raises:
        TypeError: If the array or sampling rate is missing.
        ValueError: If a channel name or index selection is invalid.
        UnresolvedChannelError: If a requested channel cannot be identified.
        IndexError: If a resolved index exceeds the available channels.
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
        airway_pressure_channel=airway_pressure_channel,
        flow_channel=flow_channel,
        volume_channel=volume_channel,
        fs=fs,
    )


def peak_indices_from_events(
    events: Sequence[BreathEvent] | None, fs: float
) -> list[int]:
    """Convert breath turning-point times in seconds to sample indices.

    Multiply each available ``extremum_time`` by ``fs`` (Hz) and truncate
    toward zero with ``int``. Events with a missing turning-point time are
    skipped. Returns an empty list for None; preserves the input event order.
    """

    if events is None:
        return []
    return [
        int(event.extremum_time * fs)
        for event in events
        if event.extremum_time is not None
    ]
