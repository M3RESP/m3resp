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

import re
from typing import Any

import numpy as np

from m3resp.core.exceptions import UnsupportedWorkflowError

#: Canonical ventilator channel -> upstream ``ContinuousData`` labels that mean
#: it, most specific first. Labels are matched after normalization (see
#: :func:`_normalize_label`), so ``"airway_pressure_(timpel)"`` and Draeger's
#: ``"airway pressure"`` both resolve through the same entry.
#:
#: The pressure entries are deliberately several: a Draeger pressure pod
#: records up to five distinct pressures, and collapsing them onto one
#: "pressure" channel would silently mislabel esophageal or transpulmonary
#: traces as airway pressure. Only the first three names here are loaded by
#: default; the rest are opt-in via ``channels=``.
VENTILATOR_CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {
    "pressure": ("airway pressure", "airway pressure (pod)", "paw", "pressure"),
    "flow": ("flow", "airway flow"),
    "volume": ("volume",),
    "esophageal_pressure": ("esophageal pressure (pod)", "esophageal pressure", "pes"),
    "transpulmonary_pressure": (
        "transpulmonary pressure (pod)",
        "transpulmonary pressure",
    ),
    "gastric_pressure": (
        "gastric pressure/auxiliary pressure (pod)",
        "gastric pressure",
        "auxiliary pressure",
        "pga",
    ),
}

#: The channels loaded unless the caller asks for others, in the array order
#: `split_channels`' default indices expect (pressure 0, flow 1, volume 2).
DEFAULT_EIT_CHANNELS: tuple[str, ...] = ("pressure", "flow", "volume")

#: Parenthesised tags that name the *file's* origin rather than a distinct
#: sensor, and so are dropped before matching. ``(pod)`` is deliberately absent:
#: it marks a physically separate transducer.
_VENDOR_TAGS = frozenset({"raw", "timpel", "draeger", "dräger", "sentec"})

_WHITESPACE = re.compile(r"\s+")


def _normalize_label(label: Any) -> str:
    """Normalize an upstream channel label for alias matching.

    Lowercases, treats underscores as spaces, and strips the vendor/provenance
    tag vendors append (``"airway_pressure_(timpel)"`` -> ``"airway
    pressure"``), while keeping meaningful tags like ``(pod)``.
    """

    text = str(label).replace("_", " ").strip().lower()
    text = _WHITESPACE.sub(" ", text)
    while text.endswith(")"):
        start = text.rfind("(")
        if start == -1:
            break
        tag = text[start + 1 : -1].strip()
        if tag not in _VENDOR_TAGS:
            break
        text = text[:start].strip()
    return text


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


def available_ventilator_channels(sequence: Any) -> dict[str, str]:
    """Map each resolvable canonical channel to the upstream label carrying it.

    Useful on its own to see what a given ``*.bin`` actually contains before
    asking for it: a Draeger file recorded without a pressure pod exposes only
    ``pressure``/``flow``/``volume``, and one recorded without Medibus
    connected may expose none.
    """

    by_normalized = {
        _normalize_label(label): label for label, _ in _continuous_data_items(sequence)
    }
    resolved: dict[str, str] = {}
    for channel, aliases in VENTILATOR_CHANNEL_ALIASES.items():
        for alias in aliases:
            match = by_normalized.get(_normalize_label(alias))
            if match is not None:
                resolved[channel] = match
                break
    return resolved


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

    unknown = [name for name in requested if name not in VENTILATOR_CHANNEL_ALIASES]
    if unknown:
        raise ValueError(
            f"Unknown ventilator channel(s) {unknown}. Known channels: "
            f"{sorted(VENTILATOR_CHANNEL_ALIASES)}."
        )

    by_label = dict(_continuous_data_items(sequence))
    available = available_ventilator_channels(sequence)

    missing = [name for name in requested if name not in available]
    if missing:
        raise UnsupportedWorkflowError(
            f"This EIT recording has no ventilator channel(s) {missing}. "
            f"Available: {sorted(available) or 'none'}. Draeger files carry "
            "ventilator waveforms only when Medibus was connected during "
            "recording; the pressure-pod channels need a pressure pod."
        )

    resolved = [by_label[available[name]] for name in requested]
    sample_frequency = _sample_frequency(resolved, fs)

    rows: list[np.ndarray] = []
    for name, data in zip(requested, resolved, strict=True):
        values = np.asarray(getattr(data, "values", data), dtype=float)
        if values.size == 0 or np.all(np.isnan(values)):
            raise UnsupportedWorkflowError(
                f"Ventilator channel {name!r} ({available[name]!r}) in this "
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
            "labels": [available[name] for name in requested],
            "units": [getattr(data, "unit", None) for data in resolved],
            "channels": list(requested),
            "time": time if time.size else None,
            "nan_samples": {
                name: int(np.count_nonzero(np.isnan(row)))
                for name, row in zip(requested, rows, strict=True)
            },
            "source": "eit",
            "available_channels": available,
        },
    }
