"""Raw-modality time cropping and synchronization trace extraction.

Operates on `M3Session`'s loaded EIT/EMG/ventilator recordings to support
`M3Session.synchronize_raw_modalities`: computing per-modality offsets
relative to a reference, cropping the raw arrays in place, and extracting
before/after traces for diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from m3resp.modalities.eit import EITRecording
from m3resp.modalities.emg import EMGRecording

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


#: Canonical name for the ventilator modality. ``"vent"`` was the Stage 1
#: internal spelling while ``"ventilator"`` was used in the docs, by
#: ``M3Session.link_breaths``, and by ``Signal.modality`` - so a breath could
#: carry ``modality="vent"`` while the ``LinkedBreath`` holding it was keyed
#: ``"ventilator"``. Everything now canonicalizes here.
VENTILATOR = "ventilator"

#: Accepted spellings that normalize to :data:`VENTILATOR`. ``"vent"`` is kept
#: indefinitely: it shipped as a ``session.raw`` key and as a pipeline-spec
#: parameter value, so specs and user code still pass it.
_VENTILATOR_ALIASES = frozenset({"vent", "ventilator", "ventilation"})


def resolve_alignment_offsets(
    offset_seconds: float | Mapping[str, float],
) -> dict[str, float]:
    if isinstance(offset_seconds, Mapping):
        offsets = {"eit": 0.0, "emg": 0.0, VENTILATOR: 0.0}
        for modality, offset in offset_seconds.items():
            offsets[normalize_modality(modality)] = float(offset)
        return offsets
    return {"eit": 0.0, "emg": float(offset_seconds), VENTILATOR: 0.0}


def offsets_relative_to_reference(
    offsets: Mapping[str, float],
    reference_modality: str,
) -> dict[str, float]:
    reference = normalize_modality(reference_modality)
    reference_offset = float(offsets.get(reference, 0.0))
    return {
        normalize_modality(modality): float(offset) - reference_offset
        for modality, offset in offsets.items()
    }


def normalize_modality(modality: str) -> str:
    """Canonicalize a modality name, mapping every ventilator spelling to
    :data:`VENTILATOR`."""

    normalized = str(modality).lower()
    if normalized in _VENTILATOR_ALIASES:
        return VENTILATOR
    return normalized


def ventilator_raw(session: M3Session) -> Any:
    """The loaded ventilator recording, under either key.

    ``session.raw`` stores it under both ``"ventilator"`` (canonical) and
    ``"vent"`` (the key Stage 1 shipped), pointing at the same object. Reading
    through here means code stays correct whichever key a caller populated -
    including a user who assigned ``session.raw["vent"]`` directly.
    """

    recording = session.raw.get(VENTILATOR)
    if recording is None:
        recording = session.raw.get("vent")
    return recording


def ventilator_payload(recording: Any) -> dict[str, Any] | None:
    """The ``{"array", "metadata"}`` payload of a ventilator recording.

    Accepts either a :class:`~m3resp.modalities.ventilator.VentilatorRecording`
    (which keeps it under ``.data``, like every other modality) or a bare dict,
    which is what Stage 1 stored directly in ``session.raw["vent"]``.
    """

    if isinstance(recording, dict) and "array" in recording:
        return recording
    data = getattr(recording, "data", None)
    if isinstance(data, dict) and "array" in data:
        return data
    return None


def crop_loaded_modality(session: M3Session, modality: str, offset: float) -> int:
    if offset == 0.0:
        return 0
    if modality == "emg" and session.emg is not None:
        return _crop_emg_recording(session.emg, offset)
    if modality == VENTILATOR:
        return _crop_ventilator_recording(ventilator_raw(session), offset)
    if modality == "eit" and session.eit is not None:
        return _crop_eit_recording(session.eit, offset)
    return 0


def raw_synchronization_traces(session: M3Session, modality: str) -> dict[str, Any]:
    if modality == "emg" and session.emg is not None:
        trace = _emg_raw_trace(session.emg)
        return {"emg": trace} if trace is not None else {}
    if modality == "eit" and session.eit is not None:
        trace = _eit_raw_trace(session.eit)
        return {"eit": trace} if trace is not None else {}
    if modality == VENTILATOR:
        return _ventilator_raw_traces(ventilator_payload(ventilator_raw(session)))
    return {}


def _crop_ventilator_recording(recording: Any, offset: float) -> int:
    """Crop a ventilator recording in place, mirroring `_crop_emg_recording`.

    The payload dict is mutated rather than replaced, so both ``session.raw``
    keys - which reference the same object - stay consistent.
    """

    cropped = _crop_recording_dict(ventilator_payload(recording), offset)
    if not cropped:
        return 0
    data = getattr(recording, "data", None)
    if isinstance(data, dict):
        recording.raw = data.get("array")
        recording.metadata = data.get("metadata")
    return cropped


def _emg_raw_trace(recording: EMGRecording) -> dict[str, Any] | None:
    data = recording.data if isinstance(recording.data, dict) else None
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    labels = metadata.get("labels") or []
    units = metadata.get("units") or []
    label = labels[0] if labels else "emg_0"
    unit = units[0] if units else "a.u."
    return _recording_dict_trace(
        data,
        title=f"EMG raw ({label})",
        ylabel=f"EMG amplitude ({unit})" if unit else "EMG amplitude",
    )


def _recording_dict_trace(
    recording: Any,
    *,
    title: str,
    ylabel: str,
    channel: int = 0,
) -> dict[str, Any] | None:
    if not isinstance(recording, dict) or "array" not in recording:
        return None
    metadata = recording.get("metadata") or {}
    fs = metadata.get("fs")
    if fs is None:
        return None
    array = np.asarray(recording["array"], dtype=float)
    if array.ndim == 0 or array.size == 0:
        return None
    if array.ndim > 1 and channel >= array.shape[0]:
        return None
    values = array[channel] if array.ndim > 1 else array
    time = np.arange(len(values), dtype=float) / float(fs)
    return {
        "title": title,
        "time": time.tolist(),
        "values": np.asarray(values, dtype=float).tolist(),
        "ylabel": ylabel,
    }


def _ventilator_raw_traces(recording: Any) -> dict[str, Any]:
    pressure = _recording_dict_trace(
        recording,
        title="Ventilator pressure",
        ylabel="Pressure",
        channel=0,
    )
    flow = _recording_dict_trace(
        recording,
        title="Ventilator flow",
        ylabel="Flow",
        channel=1,
    )
    volume = _recording_dict_trace(
        recording,
        title="Ventilator volume",
        ylabel="Volume",
        channel=2,
    )
    traces: dict[str, Any] = {}
    if pressure is not None:
        traces["vent_pressure"] = pressure
    if flow is not None:
        traces["vent_flow"] = flow
    if volume is not None:
        traces["vent_volume"] = volume
    return traces


def _eit_raw_trace(recording: EITRecording) -> dict[str, Any] | None:
    signal = recording.global_impedance
    if signal is None:
        return None
    time = getattr(signal, "time", None)
    values = getattr(signal, "values", None)
    if time is None or values is None:
        return None
    return {
        "title": "EIT raw global impedance",
        "time": np.asarray(time, dtype=float).tolist(),
        "values": np.asarray(values, dtype=float).tolist(),
        "ylabel": getattr(signal, "label", "global_impedance_(raw)"),
    }


def _crop_emg_recording(recording: EMGRecording, offset: float) -> int:
    cropped = _crop_recording_dict(recording.data, offset)
    if not cropped:
        return 0
    if isinstance(recording.data, dict):
        recording.raw = recording.data.get("array")
        recording.metadata = recording.data.get("metadata")
    return cropped


def _crop_recording_dict(recording: Any, offset: float) -> int:
    if not isinstance(recording, dict) or "array" not in recording:
        return 0
    metadata = recording.get("metadata") or {}
    fs = metadata.get("fs")
    if fs is None:
        return 0
    array, n_samples = _crop_sample_array(recording["array"], float(fs), offset)
    if not n_samples:
        return 0
    recording["array"] = array
    return n_samples


def _crop_eit_recording(recording: EITRecording, offset: float) -> int:
    sequence = recording.data
    fs = _infer_eit_sample_frequency(recording)
    if fs is None:
        return 0

    n_samples = 0
    if recording.raw is not None:
        n_samples = max(n_samples, _crop_eit_like_object(recording.raw, fs, offset))
    if recording.global_impedance is not None:
        n_samples = max(
            n_samples,
            _crop_eit_like_object(recording.global_impedance, fs, offset),
        )
    for collection_name in ("eit_data", "continuous_data"):
        collection = getattr(sequence, collection_name, None)
        if isinstance(collection, Mapping):
            for item in collection.values():
                n_samples = max(n_samples, _crop_eit_like_object(item, fs, offset))
    return n_samples


def _crop_eit_like_object(obj: Any, fs: float, offset: float) -> int:
    n_samples = 0
    for attr in ("pixel_impedance", "values", "time"):
        if not hasattr(obj, attr):
            continue
        values = getattr(obj, attr)
        cropped, cropped_samples = _crop_sample_array(
            values,
            fs,
            offset,
            sample_axis=0,
        )
        if cropped_samples:
            setattr(obj, attr, cropped)
            n_samples = max(n_samples, cropped_samples)
    return n_samples


def _crop_sample_array(
    values: Any,
    fs: float,
    offset: float,
    sample_axis: int | None = None,
) -> tuple[Any, int]:
    array = np.asarray(values)
    if array.size == 0:
        return values, 0

    if sample_axis is None:
        sample_axis = 1 if array.ndim > 1 and array.shape[1] >= array.shape[0] else 0
    n_samples = round(abs(offset) * float(fs))
    axis_len = array.shape[sample_axis]
    if n_samples <= 0:
        return values, 0
    if n_samples >= axis_len:
        raise ValueError(
            "Crop offset would remove every sample: "
            f"offset={offset!r} seconds, sample_frequency={fs!r} Hz, "
            f"requested_samples={n_samples}, axis_length={axis_len}."
        )

    if offset < 0:
        sample_slice = slice(n_samples, None)
    else:
        sample_slice = slice(None, axis_len - n_samples)
    slices = [slice(None)] * array.ndim
    slices[sample_axis] = sample_slice
    cropped = array[tuple(slices)]
    if isinstance(values, list):
        return cropped.tolist(), n_samples
    return cropped, n_samples


def _infer_eit_sample_frequency(recording: EITRecording) -> float | None:
    for obj in (recording.raw, recording.global_impedance, recording.data):
        for attr in ("sample_frequency", "sample_frequency_hz", "fs"):
            value = getattr(obj, attr, None)
            if value is not None:
                return float(value)
        metadata = getattr(obj, "metadata", None)
        if isinstance(metadata, Mapping):
            for key in ("sample_frequency", "sample_frequency_hz", "fs"):
                value = metadata.get(key)
                if value is not None:
                    return float(value)
    return None
