"""Splitting a raw ventilator recording into named per-quantity channels."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.data.categories import normalize_category

#: Ventilator channel name -> the physical quantity it carries. The keys are
#: the names used throughout the ventilator bundle dicts and as
#: ``Signal.channel``; the values are ``Signal.category`` (see
#: :mod:`m3resp.data.categories`).
CHANNEL_CATEGORIES: dict[str, str] = {
    "pressure": "airway_pressure",
    "flow": "airflow",
    "volume": "volume",
}

#: Fallback units per channel, used only when the recording's metadata does not
#: carry a unit for that channel.
DEFAULT_CHANNEL_UNITS: dict[str, str] = {
    "pressure": "cmH2O",
    "flow": "L/min",
    "volume": "L",
}


def recording_payload(recording: Any) -> dict[str, Any] | None:
    """The ``{"array", "metadata"}`` payload of a ventilator recording.

    Accepts a :class:`~m3resp.modalities.ventilator.VentilatorRecording`, a bare
    payload dict, or a raw array-like.
    """

    if isinstance(recording, dict) and "array" in recording:
        return recording
    data = getattr(recording, "data", None)
    if isinstance(data, dict) and "array" in data:
        return data
    return None


def split_channels(
    recording: Any,
    *,
    pressure_channel: int = 0,
    flow_channel: int = 1,
    volume_channel: int = 2,
    fs: float | None = None,
) -> dict[str, Any]:
    """Split a ventilator recording into ``pressure``/``flow``/``volume`` arrays.

    Returns the bundle every ventilator step consumes: the three named channel
    arrays plus ``fs``, ``metadata``, the channel indices used, and each
    channel's unit resolved from the recording metadata where available.
    """

    payload = recording_payload(recording)
    if payload is not None:
        array = payload["array"]
        metadata = payload.get("metadata") or {}
    else:
        # A bare array-like, with the sample rate supplied by the caller.
        array = recording
        metadata = {}
    if array is None:
        raise TypeError("Ventilator preprocessing input needs an array.")

    sample_frequency = fs if fs is not None else metadata.get("fs")
    if sample_frequency is None:
        raise TypeError(
            "Ventilator preprocessing input needs a sampling rate. Pass `fs=` "
            "or include metadata['fs'] in the recording."
        )

    array = np.asarray(array, dtype=float)
    indices = {
        "pressure": int(pressure_channel),
        "flow": int(flow_channel),
        "volume": int(volume_channel),
    }
    units = list(metadata.get("units") or [])
    labels = list(metadata.get("labels") or [])

    channels: dict[str, Any] = {}
    channel_units: dict[str, str | None] = {}
    channel_labels: dict[str, str | None] = {}
    for name, index in indices.items():
        if array.ndim > 1:
            if index >= array.shape[0]:
                raise IndexError(
                    f"Ventilator {name} channel index {index} is out of range "
                    f"for a recording with {array.shape[0]} channels."
                )
            channels[name] = np.asarray(array[index], dtype=float)
        else:
            channels[name] = np.asarray(array, dtype=float)
        channel_units[name] = (
            units[index]
            if index < len(units) and units[index]
            else DEFAULT_CHANNEL_UNITS[name]
        )
        channel_labels[name] = labels[index] if index < len(labels) else None

    return {
        **channels,
        "fs": float(sample_frequency),
        "metadata": metadata,
        "channel_indices": indices,
        "units": channel_units,
        "labels": channel_labels,
        "categories": {
            name: normalize_category(category)
            for name, category in CHANNEL_CATEGORIES.items()
        },
    }
