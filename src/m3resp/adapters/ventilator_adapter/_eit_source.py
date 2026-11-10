"""Read ventilator waveforms from EIT recordings.

Draeger stores ventilator channels in its Medibus fields and Timpel in text
columns next to the EIT images. The channel keys of EITRecording.signals or
EITFileData.signals are their vendor labels.

Channel labels are resolved to airway_pressure, flow, volume and the
additional pressure-pod channels. The result has one channel per row and
metadata with units, sampling frequency and the original time axis.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.exceptions import UnsupportedWorkflowError
from m3resp.data.signals import Signal
from m3resp.eit.io.file_data import EITFileData
from m3resp.modalities.eit import EITRecording

from ._channels import CHANNEL_CATEGORIES, ChannelSpec, resolve_channels

#: The channels loaded unless the caller asks for others.
DEFAULT_EIT_CHANNELS: tuple[str, ...] = ("airway_pressure", "flow", "volume")

#: What `Signal.source` records for a channel read out of an EIT recording.
EIT_ORIGIN = "eit"

#: Below this, a sample is a "not measured" marker rather than a reading.
#:
#: A Draeger recording writes every channel it can carry, connected sensor or
#: not, and fills an unmeasured sample with a large negative number standing in
#: for NaN. The exact number differs between software versions, so a cutoff is
#: used instead of an equality test - no physiological pressure, flow or volume
#: comes anywhere near it. Same threshold `eitprocessing` applies when it reads
#: the impedance frames.
SENTINEL_CUTOFF = -1e30


def _eit_waveforms(recording: Any) -> list[tuple[str, Signal]]:
    """Return the vendor labels and waveforms of an EIT recording."""

    if not isinstance(recording, (EITRecording, EITFileData)):
        raise TypeError(
            "Loading ventilator data from an EIT recording expects an "
            f"EITRecording or EITFileData, got {type(recording).__name__}."
        )
    return list(recording.signals.items())


def _eit_channel_specs(
    recording: Any, requested: Any = tuple(CHANNEL_CATEGORIES)
) -> tuple[list[ChannelSpec], list[str], list[Signal]]:
    """Find the ventilator channels of an EIT recording by their vendor labels.

    Returns the specs with the labels and waveforms of the recording, in the
    same order, so each spec can be paired with its waveform.
    """

    items = _eit_waveforms(recording)
    labels = [label for label, _ in items]
    values = [data for _, data in items]
    specs = resolve_channels(
        labels,
        requested=requested,
        origin=EIT_ORIGIN,
        units=[getattr(data, "unit", None) for data in values],
        # An EIT recording names every channel it holds, so there is no
        # column layout to fall back to: a channel is either named or absent.
        fallback_positions=False,
    )
    return specs, labels, values


def available_ventilator_channels(recording: Any) -> dict[str, str]:
    """List ventilator channel keys and vendor labels in an EIT recording.

    Args:
        recording (Any): An EITRecording or EITFileData.

    Returns:
        dict[str, str]: Resolvable channel key to vendor label. Multiple airway
            pressures have distinct keys, such as airway_pressure and
            airway_pressure__pod. The list describes channel labels, including
            channels whose sensors were disconnected. Reading their values through
            the adapter converts missing-measurement sentinels to NaN and rejects
            channels containing only missing measurements.

    Raises:
        TypeError: If ``recording`` is not an EITRecording or EITFileData.
    """

    specs, _, _ = _eit_channel_specs(recording)
    return {spec.key: spec.label for spec in specs if spec.label is not None}


def _sample_frequency(channels: list[Any], fs: float | None) -> float:
    if fs is not None:
        return float(fs)
    for data in channels:
        sample_frequency = getattr(data, "sample_frequency", None)
        if sample_frequency:
            return float(sample_frequency)
    # No upstream rate: fall back to the median step of the shared time axis,
    # which vendors write per frame.
    for data in channels:
        time = np.asarray(getattr(data, "time", ()), dtype=float)
        if time.size > 1:
            step = float(np.median(np.diff(time)))
            if step > 0:
                return 1.0 / step
    raise TypeError(
        "Ventilator channels from this EIT recording carry no sample "
        "frequency and no usable time axis. Pass `fs=` explicitly."
    )


def ventilator_payload_from_eit(
    recording: Any,
    *,
    channels: tuple[str, ...] | list[str] = DEFAULT_EIT_CHANNELS,
    fs: float | None = None,
) -> dict[str, Any]:
    """Pack the ventilator waveforms of an EIT recording into a ventilator payload.

    Returns the ``{"array", "metadata"}`` shape the ventilator path already
    consumes: ``array`` has one row per requested channel, in the requested
    order, and ``metadata`` carries ``fs``, per-row ``labels``/``units``, the
    shared ``time`` axis, and ``available_channels`` (everything this recording
    could have offered, so a caller that got the default three can discover the
    pressure-pod channels without reloading).

    Samples below `SENTINEL_CUTOFF` are the device's "not measured" marker and
    become NaN on the way in, so ``nan_samples`` counts everything the file did
    not actually record.

    Raises `UnsupportedWorkflowError` when a requested channel is absent or
    carries no measurement at all - a Medibus field the ventilator never
    populated, or a pod channel recorded without a pod attached. Such a channel
    is rejected here rather than filtered downstream into an all-NaN result.
    """

    requested = tuple(channels)
    if not requested:
        raise ValueError("Ventilator channel selection cannot be empty.")

    unknown = [name for name in requested if name not in CHANNEL_CATEGORIES]
    if unknown:
        raise ValueError(
            f"Unknown ventilator channel(s) {unknown}. Known channels: "
            f"{sorted(CHANNEL_CATEGORIES)}."
        )

    specs, _, values_by_index = _eit_channel_specs(recording, requested)
    available = available_ventilator_channels(recording)

    missing = [name for name in requested if name not in {s.channel for s in specs}]
    if missing:
        raise UnsupportedWorkflowError(
            f"This EIT recording has no ventilator channel(s) {missing}. "
            f"Available: {sorted(available) or 'none'}. Draeger files carry "
            "ventilator waveforms only when Medibus was connected during "
            "recording; the pressure-pod channels need a pressure pod."
        )

    resolved = [values_by_index[spec.index or 0] for spec in specs]
    sample_frequency = _sample_frequency(resolved, fs)

    rows: list[np.ndarray] = []
    for spec, data in zip(specs, resolved, strict=True):
        values = np.asarray(getattr(data, "values", data), dtype=float)
        # A channel the device exposed but never measured is written as the
        # negative sentinel, not omitted; turn it into NaN so the emptiness
        # check below sees it and no downstream step mistakes it for a reading.
        values = np.where(values < SENTINEL_CUTOFF, np.nan, values)
        if values.size == 0 or np.all(np.isnan(values)):
            raise UnsupportedWorkflowError(
                f"Ventilator channel {spec.key!r} ({spec.label!r}) in this "
                "EIT recording holds no measurement. The device lists a "
                "channel whether or not the matching sensor was recording, "
                "so this usually means the ventilator was not connected, or "
                "that no pressure pod was attached."
            )
        rows.append(values)

    lengths = {row.size for row in rows}
    if len(lengths) > 1:
        raise UnsupportedWorkflowError(
            "Ventilator channels from this EIT recording have differing "
            f"lengths {sorted(lengths)}; they are expected to share one time "
            "axis."
        )

    time = np.asarray(getattr(resolved[0], "time", ()), dtype=float)
    for spec, data in zip(specs, resolved, strict=True):
        other_time = np.asarray(getattr(data, "time", ()), dtype=float)
        if not np.array_equal(time, other_time):
            raise UnsupportedWorkflowError(
                f"Ventilator channel {spec.key!r} has a different time axis "
                "from the other channels in this EIT recording."
            )
    return {
        "array": np.vstack(rows),
        "metadata": {
            "fs": sample_frequency,
            "labels": [spec.label for spec in specs],
            "units": [spec.unit for spec in specs],
            "channels": [spec.key for spec in specs],
            "time": time if time.size else None,
            "nan_samples": {
                spec.key: int(np.count_nonzero(np.isnan(row)))
                for spec, row in zip(specs, rows, strict=True)
            },
            "source": EIT_ORIGIN,
            "available_channels": available,
        },
    }
