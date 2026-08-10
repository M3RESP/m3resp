"""Reading ventilator waveforms out of an EIT recording.

Ventilator data reaches m3resp from two independent sources, not one:

* the multi-channel file it shares with the sEMG (a Biopac export and
  friends), read by :class:`~m3resp.adapters.resurfemg_adapter.ReSurfEMGAdapter`;
* the EIT ``*.bin`` file itself, where the device stores the ventilator
  waveforms alongside the impedance frames - Draeger via its Medibus fields,
  Timpel as dedicated columns. ``eitprocessing`` already parses both and
  exposes them as ``ContinuousData`` on the loaded ``Sequence``.

This module covers the second source. It resolves the vendor's channel labels
onto m3resp's canonical ``pressure``/``flow``/``volume`` names (plus the
esophageal/transpulmonary/gastric channels a Draeger pressure pod adds) and
packs them into the same ``{"array", "metadata"}`` payload the EMG-file path
produces, so everything downstream of loading - `split_channels`, the
preprocessing defaults, cropping, `to_signals` - stays source-agnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.exceptions import UnsupportedWorkflowError

from ._channels import CHANNEL_CATEGORIES, ChannelSpec, resolve_channels

#: The channels loaded unless the caller asks for others.
DEFAULT_EIT_CHANNELS: tuple[str, ...] = ("pressure", "flow", "volume")

#: What `Signal.source` records for a channel read out of an EIT recording.
EIT_ORIGIN = "eit"


def _continuous_data_items(sequence: Any) -> list[tuple[str, Any]]:
    """``(label, ContinuousData)`` pairs from a sequence, or a clear error."""

    collection = getattr(sequence, "continuous_data", None)
    if collection is None:
        raise TypeError(
            "Loading ventilator data from an EIT recording expects an "
            "eitprocessing Sequence (with `.continuous_data`), got "
            f"{type(sequence).__name__}."
        )
    try:
        return list(collection.items())
    except AttributeError:  # pragma: no cover - defensive
        return [(getattr(value, "label", ""), value) for value in collection]


def _sequence_channel_specs(
    sequence: Any, requested: Any = tuple(CHANNEL_CATEGORIES)
) -> tuple[list[ChannelSpec], list[str], list[Any]]:
    """Resolve a sequence's ventilator channels via the shared alias registry.

    Returns the specs alongside the sequence's labels and `ContinuousData`
    objects, indexed the same way, so callers can pair a spec with its data.
    """

    items = _continuous_data_items(sequence)
    labels = [label for label, _ in items]
    values = [data for _, data in items]
    specs = resolve_channels(
        labels,
        requested=requested,
        origin=EIT_ORIGIN,
        units=[getattr(data, "unit", None) for data in values],
        # An EIT sequence names every channel it carries, so there is no
        # column layout to fall back to: a channel is either named or absent.
        fallback_positions=False,
    )
    return specs, labels, values


def available_ventilator_channels(sequence: Any) -> dict[str, str]:
    """Map each resolvable channel key to the upstream label carrying it.

    Useful on its own to see what a given ``*.bin`` actually contains before
    asking for it: a Draeger file recorded without a pressure pod exposes only
    ``pressure``/``flow``/``volume``, and one recorded without Medibus
    connected may expose none. A file carrying both the ventilator's airway
    pressure and a pod's reports them as two keys (``pressure`` and
    ``pressure__pod``) rather than dropping one.
    """

    specs, _, _ = _sequence_channel_specs(sequence)
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


def ventilator_payload_from_sequence(
    sequence: Any,
    *,
    channels: tuple[str, ...] | list[str] = DEFAULT_EIT_CHANNELS,
    fs: float | None = None,
) -> dict[str, Any]:
    """Pack an EIT sequence's ventilator channels into a ventilator payload.

    Returns the ``{"array", "metadata"}`` shape the ventilator path already
    consumes: ``array`` has one row per requested channel, in the requested
    order, and ``metadata`` carries ``fs``, per-row ``labels``/``units``, the
    shared ``time`` axis, and ``available_channels`` (everything this recording
    could have offered, so a caller that got the default three can discover the
    pressure-pod channels without reloading).

    Raises `UnsupportedWorkflowError` when a requested channel is absent or
    present-but-empty. Draeger writes NaN for a Medibus field the ventilator
    never populated, so an all-NaN channel means "not recorded" and is rejected
    here rather than filtered downstream into an all-NaN result.
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

    specs, _, values_by_index = _sequence_channel_specs(sequence, requested)
    available = available_ventilator_channels(sequence)

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
        if values.size == 0 or np.all(np.isnan(values)):
            raise UnsupportedWorkflowError(
                f"Ventilator channel {spec.key!r} ({spec.label!r}) in this "
                "EIT recording is empty. That usually means the ventilator "
                "was not connected while recording."
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
