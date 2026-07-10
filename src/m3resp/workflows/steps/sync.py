"""Registered synchronization pipeline steps.

``sync.estimate_offset`` estimates the constant time offset that aligns the EIT
recording to the Biopac/EMG clock directly from the raw signals (see
:mod:`m3resp.synchronization.offset_estimation`) and writes it into the pipeline
context. Downstream, ``sync.apply_estimated_offset`` consumes that value and
crops the modalities, keeping estimation and application as separate,
declarative steps.

Run this *after* the ``*.load`` steps but *before* ``session.sync_raw``: the
interference anchor needs the full, un-cropped sEMG including the EIT-off tail,
and the raw recordings live in ``session.raw`` / ``session.emg`` / ``session.eit``
before any cropping happens.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.data.timeseries import TimeSeries
from m3resp.synchronization.offset_estimation import estimate_sync_offset
from m3resp.workflows.registry import register_step


def _dict_array(obj: Any) -> dict[str, Any] | None:
    """Return the ``{"array", "metadata"}`` payload of a raw recording, if any.

    The EMG recording keeps it under ``recording.data``; the ventilator recording
    is stored as the dict itself. Both carry ``metadata["fs"]``.
    """

    if isinstance(obj, dict) and "array" in obj:
        return obj
    data = getattr(obj, "data", None)
    if isinstance(data, dict) and "array" in data:
        return data
    return None


def _channel_timeseries(obj: Any, channel: int) -> TimeSeries | None:
    """Build a :class:`TimeSeries` from one channel of a dict-array recording."""

    data = _dict_array(obj)
    if data is None:
        return None
    metadata = data.get("metadata") or {}
    sample_frequency = metadata.get("fs")
    if sample_frequency is None:
        return None
    array = np.asarray(data["array"], dtype=float)
    if array.size == 0:
        return None
    if array.ndim > 1:
        if channel >= array.shape[0]:
            return None
        values = array[channel]
    else:
        values = array
    time = np.arange(values.shape[0], dtype=float) / float(sample_frequency)
    return TimeSeries(
        values=np.asarray(values, dtype=float),
        time=time,
        sample_frequency=float(sample_frequency),
    )


def _eit_global_impedance_timeseries(session: M3Session) -> TimeSeries | None:
    """Build a :class:`TimeSeries` from the raw EIT global impedance."""

    eit = getattr(session, "eit", None)
    signal = getattr(eit, "global_impedance", None) if eit is not None else None
    if signal is None:
        return None
    time = getattr(signal, "time", None)
    values = getattr(signal, "values", None)
    if time is None or values is None:
        return None
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.size == 0:
        return None
    sample_frequency = None
    if time.size >= 2:
        step = float(np.median(np.diff(time)))
        if step > 0:
            sample_frequency = 1.0 / step
    # Shift onto a target-relative clock (t=0 at recording start).
    return TimeSeries(
        values=values, time=time - time[0], sample_frequency=sample_frequency
    )


def _raw_source(
    session: M3Session, source: str, channel: int, *, required: bool
) -> TimeSeries | None:
    """Return the raw :class:`TimeSeries` for ``source`` (``eit``/``emg``/``vent``)."""

    if source == "eit":
        signal = _eit_global_impedance_timeseries(session)
    elif source == "emg":
        signal = _channel_timeseries(getattr(session, "emg", None), channel)
    elif source == "vent":
        signal = _channel_timeseries(session.raw.get("vent"), channel)
    else:
        raise ValueError(
            f"Unknown signal source {source!r}; expected 'eit', 'emg', or 'vent'."
        )
    if signal is None and required:
        raise ValueError(
            f"sync.estimate_offset could not extract a raw '{source}' signal "
            f"(channel {channel}). Ensure the modality was loaded before this "
            f"step and that its sample rate is known."
        )
    return signal


@register_step(
    "sync.estimate_offset",
    reads={"session": "session"},
    writes=("estimated_offset_seconds", "offset_estimation"),
    summary="Estimate the EIT-to-Biopac time offset from the raw signals.",
)
def estimate_offset(
    session: M3Session,
    *,
    method: str = "interference",
    emg_source: str = "emg",
    emg_channel: int = 0,
    target_source: str = "eit",
    target_channel: int = 0,
    reference_source: str = "vent",
    reference_channel: int = 0,
    reference_duration_seconds: float | None = None,
    manual_offset_seconds: float = 0.0,
    interference_kwargs: dict[str, Any] | None = None,
    crosscorrelation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate the sync offset and expose it for ``session.sync_raw``.

    Reads the raw, un-cropped signals from the session:

    * ``emg_source``/``emg_channel`` -- the interference-bearing sEMG (the diaphragm
      sEMG channel; ``'emg'`` for a dedicated EMG recording or ``'vent'`` if the
      sEMG shares the Biopac ventilator file).
    * ``target_source``/``target_channel`` -- the breathing target, normally EIT
      global impedance (``'eit'``).
    * ``reference_source``/``reference_channel`` -- the reference breathing signal
      for cross-correlation, normally airway pressure (``'vent'`` channel 0).

    Writes two context artifacts: ``estimated_offset_seconds`` (a float, ready to
    bind onto ``session.sync_raw``'s ``offset_seconds``) and ``offset_estimation``
    (a JSON-friendly summary of what each estimator found, for provenance/QA).
    """

    needs_emg = method in ("interference", "interference+crosscorrelation")
    needs_breathing = method in ("crosscorrelation", "interference+crosscorrelation")

    # The target (EIT) is also needed by the interference method when no explicit
    # reference_duration_seconds was supplied, since its duration backs out the edge.
    needs_target = needs_breathing or (needs_emg and reference_duration_seconds is None)

    emg = _raw_source(session, emg_source, emg_channel, required=needs_emg)
    target = _raw_source(session, target_source, target_channel, required=needs_target)
    reference = _raw_source(
        session, reference_source, reference_channel, required=needs_breathing
    )

    result = estimate_sync_offset(
        method=method,
        emg=emg,
        target=target,
        reference=reference,
        reference_duration_seconds=reference_duration_seconds,
        manual_offset_seconds=manual_offset_seconds,
        interference_kwargs=interference_kwargs,
        crosscorrelation_kwargs=crosscorrelation_kwargs,
    )

    summary: dict[str, Any] = {
        "method": result.method,
        "offset_seconds": result.offset_seconds,
        "source": result.source,
    }
    if result.interference is not None:
        interference = result.interference
        summary["interference"] = {
            "detected": interference.detected,
            "offset_seconds": interference.offset_seconds,
            "edge_time_seconds": interference.edge_time_seconds,
            "reference_duration_seconds": interference.reference_duration_seconds,
            "high_power": interference.high_power,
            "low_power": interference.low_power,
        }
    if result.crosscorrelation is not None:
        crosscorrelation = result.crosscorrelation
        summary["crosscorrelation"] = {
            "lag_seconds": crosscorrelation.lag_seconds,
            "refined_offset_seconds": crosscorrelation.refined_offset_seconds,
            "peak_correlation": crosscorrelation.peak_correlation,
        }

    session.parameters["offset_estimation"] = summary
    return {
        "estimated_offset_seconds": result.offset_seconds,
        "offset_estimation": summary,
    }


@register_step(
    "sync.apply_estimated_offset",
    reads={
        "session": "session",
        "offset_seconds": "estimated_offset_seconds",
    },
    writes=("sync_summary",),
    summary="Apply an estimated EIT-to-source offset to the raw modalities.",
)
def apply_estimated_offset(
    session: M3Session,
    offset_seconds: float,
    *,
    target_modality: str = "eit",
    source_modalities: tuple[str, ...] | list[str] = ("emg", "vent"),
) -> dict[str, Any]:
    """Crop source-clock recordings so they start with the target recording.

    ``sync.estimate_offset`` reports where target ``t=0`` falls on the source
    clock. The corresponding Stage-1 crop is therefore the negative estimate
    on each source modality, with the target modality held at zero.
    """

    target = str(target_modality).lower()
    sources = tuple(str(modality).lower() for modality in source_modalities)
    if not sources:
        raise ValueError("source_modalities must contain at least one modality")
    if target in sources:
        raise ValueError("target_modality cannot also be a source modality")

    estimate = float(offset_seconds)
    configured_offsets = {target: 0.0}
    configured_offsets.update({modality: -estimate for modality in sources})
    summary = session.synchronize_raw_modalities(
        method="manual_offset",
        offset_seconds=configured_offsets,
        reference_modality=target,
    )
    session.parameters["raw_alignment"]["estimated_offset_seconds"] = estimate
    session.parameters["raw_alignment"]["estimate_direction"] = (
        "target_t0_on_source_clock"
    )
    return {"sync_summary": summary}
