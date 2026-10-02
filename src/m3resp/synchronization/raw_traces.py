"""Before/after traces for the raw synchronization plot.

`M3Session.synchronize_raw_modalities` stores one trace per loaded raw signal
(EIT global impedance, the first EMG channel, and ventilator pressure, flow
and volume), each on its recording's own clock, for
`plot_synchronization_comparison` to draw before and after the start times
are applied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from m3resp.modalities.eit import EITRecording
from m3resp.modalities.emg import EMGRecording
from m3resp.modalities.names import VENTILATOR
from m3resp.modalities.ventilator import ventilator_payload, ventilator_raw

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


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
