"""Registered EMG pipeline steps.

Loading/preprocessing steps wrap the ``M3Session`` EMG stage methods.
Postprocessing steps keep the one-step-per-operation structure used by
``eit.py``. Factored signal-processing primitives are used where available;
remaining upstream imports are deferred to call time so the package installs
without the optional ``resurfemg`` dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from m3resp.adapters.resurfemg_adapter import (
    _peak_indices_from_events,
    _ventilator_signals,
)
from m3resp.core.session import (
    M3Session,
    _iter_ventilator_detections,
    _normalize_ventilator_breath,
)
from m3resp.core.events import BreathEvent, Event
from m3resp.data import ParameterResult, Signal
from m3resp.processing.intervals import (
    onoff_from_baseline_crossings,
    onoff_from_slope,
)
from m3resp.processing.metrics import (
    amplitude_at_peaks,
    area_under_baseline as _area_under_baseline,
    pseudo_slope as _pseudo_slope,
    respiratory_rate_from_indices,
    time_to_peak as _time_to_peak,
    window_integral,
)
from m3resp.processing.peaks import (
    detect_occluded_breath_peaks,
    detect_ventilator_breath_peaks,
)
from m3resp.processing.windows import rolling_arv
from m3resp.workflows.registry import register_step


def _resurfemg_version() -> str | None:
    """Installed `resurfemg` version, read from package metadata without
    importing the package itself (so this stays optional-dependency-safe)."""

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("resurfemg")
    except PackageNotFoundError:
        return None


def _upstream_metadata(
    *,
    source_function: str,
    operation: str,
    parameters: dict[str, Any],
    source_package: str = "resurfemg",
    implementation: str = "upstream_adapter",
) -> dict[str, Any]:
    """Build the Stage 2 EMG provenance metadata schema shared by native
    `Signal`/`ParameterResult` outputs (see `plan/stage2/
    2_resurfemg_gap_migration_implementation_plan.md`, "Use one provenance
    schema"). Mirrors `m3resp.workflows.steps.eit._upstream_metadata`.

    `source_package`/`implementation` default to the ReSurfEMG-adapter case;
    pass `source_package="m3resp"`, `implementation="m3resp.processing.<module>"`
    for a step whose value comes from a native primitive instead.
    """

    return {
        "source_package": source_package,
        "source_function": source_function,
        "implementation": implementation,
        "parameters": parameters,
        "operation": operation,
    }


def _record_step(
    session: M3Session, step_name: str, *, metadata: dict[str, Any]
) -> None:
    """Record per-step EMG provenance through the existing
    `M3Session._record()` seam, reusing the step's declared reads/writes
    from the registry rather than a second EMG-only history mechanism."""

    from m3resp.workflows.registry import get_step

    definition = get_step(step_name)
    session._record(
        step_name,
        "emg",
        parameters={
            "step": step_name,
            "reads": sorted(definition.reads),
            "writes": list(definition.writes),
            "upstream_version": _resurfemg_version(),
            **metadata,
        },
    )


@register_step(
    "emg.load",
    reads={"session": "session"},
    writes=("emg_recording", "raw_emg_signals"),
    summary="Load an EMG recording into the session.",
)
def load(
    session: M3Session,
    *,
    file: str,
    loader_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if loader_options is not None and not isinstance(loader_options, Mapping):
        raise TypeError(
            "emg.load 'loader_options' must be a mapping of keyword arguments "
            f"for M3Session.load_emg(), got {type(loader_options).__name__}."
        )

    session.load_emg(file, verbose=False, **dict(loader_options or {}))
    recording = session.emg
    assert recording is not None

    metadata = dict(recording.metadata or {})
    fs = metadata.get("fs")
    labels = list(metadata.get("labels") or [])
    units = list(metadata.get("units") or [])
    raw_array = np.asarray(recording.raw) if recording.raw is not None else None

    raw_emg_signals: list[Signal] = []
    if raw_array is not None and fs:
        time = np.arange(raw_array.shape[1], dtype=float) / float(fs)
        for index in range(raw_array.shape[0]):
            label = labels[index] if index < len(labels) else f"emg_{index}"
            unit = units[index] if index < len(units) else None
            signal = Signal(
                values=raw_array[index],
                time=time,
                sample_frequency=float(fs),
                unit=unit,
                name=label,
                modality="emg",
                channel=label,
                source=str(recording.path),
                processing_state="raw",
                metadata={"channel_index": index, "file_metadata": metadata},
            )
            session.signals.add(signal)
            raw_emg_signals.append(signal)

    _record_step(
        session,
        "emg.load",
        metadata=_upstream_metadata(
            source_function="resurfemg.data_connector.converter_functions.load_file",
            operation="emg.load",
            parameters={"loader_options": dict(loader_options or {})},
        ),
    )
    return {"emg_recording": recording, "raw_emg_signals": raw_emg_signals}


@register_step(
    "emg.load_ventilator",
    reads={"session": "session"},
    writes=("ventilator_raw",),
    summary="Load a ventilator recording into the session.",
)
def load_ventilator(session: M3Session, *, file: str) -> dict[str, Any]:
    recording = session.emg_adapter.load(str(file), verbose=False)
    # Stored on the session too so `session.sync_raw` (which crops
    # `session.raw["vent"]` in place) keeps this same dict object in sync.
    session.raw["vent"] = recording
    return {"ventilator_raw": recording}


@register_step(
    "emg.ventilator_channels",
    reads={"ventilator_raw": "ventilator_raw"},
    writes=("ventilator_signals",),
    summary="Split a raw ventilator recording into pressure/flow/volume channels.",
)
def ventilator_channels(
    ventilator_raw: Any,
    *,
    pressure_channel: int = 0,
    flow_channel: int = 1,
    volume_channel: int = 2,
    fs: float | None = None,
) -> dict[str, Any]:
    signals = _ventilator_signals(
        ventilator_raw,
        pressure_channel=pressure_channel,
        flow_channel=flow_channel,
        volume_channel=volume_channel,
        fs=fs,
    )
    return {"ventilator_signals": signals}


@register_step(
    "emg.preprocess",
    reads={"session": "session"},
    writes=("processed_emg",),
    summary="Filter EMG and compute its envelope.",
)
def preprocess(session: M3Session, **kwargs: Any) -> dict[str, Any]:
    return {"processed_emg": session.preprocess_emg(**kwargs)}


@register_step(
    "emg.detect_breaths",
    reads={"session": "session"},
    writes=("emg_breath_events",),
    summary="Detect EMG breaths from the envelope.",
)
def detect_breaths(session: M3Session, **kwargs: Any) -> dict[str, Any]:
    events = session.detect_emg_breaths(**kwargs)
    return {"emg_breath_events": events}


@register_step(
    "emg.peak_indices",
    reads={"events": "emg_breath_events", "processed_emg": "processed_emg"},
    writes=("peak_indices",),
    summary="Derive EMG breath peak sample indices from detected breath events.",
)
def peak_indices(events: Any, processed_emg: Any) -> dict[str, Any]:
    import numpy as np

    fs = float(processed_emg["fs"])
    return {
        "peak_indices": np.asarray(_peak_indices_from_events(events, fs), dtype=int)
    }


# --- baseline -----------------------------------------------------------
#
# `emg.moving_baseline` and `emg.slopesum_baseline` both write `baseline`.
# A pipeline picks exactly one (or renames one via `out:`) - this makes the
# baseline choice an explicit YAML decision instead of the previous silent
# "whichever ran last wins when both are enabled" behavior.


def _require_positive_seconds(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite, positive number; got {value!r}.")


def _require_percentile(name: str, value: float) -> None:
    if not (0.0 <= float(value) <= 100.0):
        raise ValueError(f"{name} must be between 0 and 100; got {value!r}.")


def _processed_channel_label_and_unit(processed_emg: Any) -> tuple[str, str | None]:
    channel_index = processed_emg.get("channel")
    metadata = processed_emg.get("metadata") or {}
    labels = list(metadata.get("labels") or [])
    units = list(metadata.get("units") or [])
    label = (
        labels[channel_index]
        if isinstance(channel_index, int) and channel_index < len(labels)
        else str(channel_index)
    )
    unit = (
        units[channel_index]
        if isinstance(channel_index, int) and channel_index < len(units)
        else None
    )
    return label, unit


@register_step(
    "emg.moving_baseline",
    reads={"session": "session", "processed_emg": "processed_emg"},
    writes=("baseline", "baseline_signal"),
    summary="Compute a moving-percentile EMG baseline.",
)
def moving_baseline(
    session: M3Session,
    processed_emg: Any,
    *,
    window_seconds: float = 30.0,
    step_seconds: float = 1.0,
    percentile: float = 33.0,
) -> dict[str, Any]:
    _require_positive_seconds("window_seconds", window_seconds)
    _require_positive_seconds("step_seconds", step_seconds)
    _require_percentile("percentile", percentile)

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    window_samples = max(1, int(window_seconds * fs))
    step_samples = max(1, int(step_seconds * fs))
    baseline = session.emg_adapter.moving_baseline(
        envelope,
        window_samples=window_samples,
        step_samples=step_samples,
        percentile=percentile,
    )

    label, unit = _processed_channel_label_and_unit(processed_emg)
    time = np.arange(len(envelope), dtype=float) / fs
    baseline_signal = Signal(
        values=baseline,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_moving_baseline",
        modality="emg",
        channel=label,
        source="resurfemg",
        processing_state="derived",
        derived_from="processed",
        method="resurfemg.moving_baseline",
        metadata={
            "requested_window_seconds": window_seconds,
            "requested_step_seconds": step_seconds,
            "effective_window_samples": window_samples,
            "effective_step_samples": step_samples,
            "percentile": percentile,
        },
    )
    session.signals.add(baseline_signal)
    _record_step(
        session,
        "emg.moving_baseline",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.baseline.moving_baseline",
            operation="emg.moving_baseline",
            parameters={
                "window_seconds": window_seconds,
                "step_seconds": step_seconds,
                "percentile": percentile,
                "effective_window_samples": window_samples,
                "effective_step_samples": step_samples,
            },
        ),
    )
    return {"baseline": baseline, "baseline_signal": baseline_signal}


@register_step(
    "emg.slopesum_baseline",
    reads={"session": "session", "processed_emg": "processed_emg"},
    writes=(
        "baseline",
        "slopesum_baseline_detail",
        "baseline_signal",
        "slopesum_baseline_native_detail",
        "baseline_running_mean_signal",
        "baseline_running_std_signal",
    ),
    summary="Compute a slope-sum EMG baseline.",
)
def slopesum_baseline(
    session: M3Session,
    processed_emg: Any,
    *,
    window_seconds: float = 30.0,
    step_seconds: float = 1.0,
    percentile: float = 33.0,
    augmented_percentile: float = 25.0,
    moving_average_seconds: float = 0.5,
    percentile_window_seconds: float = 1.0,
) -> dict[str, Any]:
    _require_positive_seconds("window_seconds", window_seconds)
    _require_positive_seconds("step_seconds", step_seconds)
    _require_positive_seconds("moving_average_seconds", moving_average_seconds)
    _require_positive_seconds("percentile_window_seconds", percentile_window_seconds)
    _require_percentile("percentile", percentile)
    _require_percentile("augmented_percentile", augmented_percentile)

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    window_samples = max(1, int(window_seconds * fs))
    step_samples = max(1, int(step_seconds * fs))
    moving_average_samples = max(1, int(moving_average_seconds * fs))
    percentile_window_samples = max(1, int(percentile_window_seconds * fs))
    baseline, running_mean, running_std, series = session.emg_adapter.slopesum_baseline(
        envelope,
        window_samples=window_samples,
        step_samples=step_samples,
        sample_frequency=fs,
        percentile=percentile,
        augmented_percentile=augmented_percentile,
        moving_average_samples=moving_average_samples,
        percentile_window_samples=percentile_window_samples,
    )

    label, unit = _processed_channel_label_and_unit(processed_emg)
    time = np.arange(len(envelope), dtype=float) / fs
    effective_samples = {
        "requested_window_seconds": window_seconds,
        "requested_step_seconds": step_seconds,
        "requested_moving_average_seconds": moving_average_seconds,
        "requested_percentile_window_seconds": percentile_window_seconds,
        "effective_window_samples": window_samples,
        "effective_step_samples": step_samples,
        "effective_moving_average_samples": moving_average_samples,
        "effective_percentile_window_samples": percentile_window_samples,
        "percentile": percentile,
        "augmented_percentile": augmented_percentile,
    }
    baseline_signal = Signal(
        values=baseline,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_slopesum_baseline",
        modality="emg",
        channel=label,
        source="resurfemg",
        processing_state="derived",
        derived_from="processed",
        method="resurfemg.slopesum_baseline",
        metadata=dict(effective_samples),
    )
    running_mean_signal = Signal(
        values=running_mean,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_slopesum_baseline_running_mean",
        modality="emg",
        channel=label,
        source="resurfemg",
        processing_state="derived",
        derived_from="processed",
        method="resurfemg.slopesum_baseline",
    )
    running_std_signal = Signal(
        values=running_std,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_slopesum_baseline_running_std",
        modality="emg",
        channel=label,
        source="resurfemg",
        processing_state="derived",
        derived_from="processed",
        method="resurfemg.slopesum_baseline",
    )
    session.signals.add(baseline_signal)
    session.signals.add(running_mean_signal)
    session.signals.add(running_std_signal)

    _record_step(
        session,
        "emg.slopesum_baseline",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.baseline.slopesum_baseline",
            operation="emg.slopesum_baseline",
            parameters=effective_samples,
        ),
    )
    return {
        "baseline": baseline,
        "slopesum_baseline_detail": {
            "running_mean": running_mean,
            "running_std": running_std,
            "series": series,
        },
        "baseline_signal": baseline_signal,
        "slopesum_baseline_native_detail": {
            "running_mean": running_mean,
            "running_std": running_std,
        },
        "baseline_running_mean_signal": running_mean_signal,
        "baseline_running_std_signal": running_std_signal,
    }


def _select_ecg_source(
    session: M3Session,
    processed_emg: Any,
    *,
    ecg_channel: int | None,
    source: str,
) -> tuple[np.ndarray, float, str]:
    """Return `(array, sample_frequency, source_label)` for ECG detection.

    `ecg_channel` (a raw channel-major index) takes priority over `source`
    (a key into `processed_emg`, e.g. "raw_channel"/"filtered"/"envelope").
    """

    if ecg_channel is not None:
        recording = session.emg
        if recording is None or recording.raw is None:
            raise ValueError("emg.ecg_detect_peaks needs a loaded EMG recording.")
        raw = np.asarray(recording.raw)
        if not (0 <= ecg_channel < raw.shape[0]):
            raise ValueError(
                f"ecg_channel {ecg_channel!r} is out of range; the loaded "
                f"recording has channels 0..{raw.shape[0] - 1}."
            )
        raw_fs = (recording.metadata or {}).get("fs")
        if raw_fs is None:
            raise ValueError("emg.ecg_detect_peaks needs recording.metadata['fs'].")
        return raw[ecg_channel], float(raw_fs), f"raw_channel[{ecg_channel}]"

    if source not in processed_emg:
        available = sorted(
            key
            for key, value in processed_emg.items()
            if isinstance(value, np.ndarray) or hasattr(value, "__len__")
        )
        raise ValueError(
            f"emg.ecg_detect_peaks source {source!r} is not present in "
            f"processed_emg; available keys: {available}."
        )
    array = np.asarray(processed_emg[source], dtype=float)
    fs = float(processed_emg["fs"])
    return array, fs, source


@register_step(
    "emg.ecg_detect_peaks",
    reads={"session": "session", "processed_emg": "processed_emg"},
    writes=("ecg_peak_indices", "ecg_peak_events", "ecg_peak_count_result"),
    summary="Detect ECG peak sample indices in an EMG/ECG channel.",
)
def ecg_detect_peaks(
    session: M3Session,
    processed_emg: Any,
    *,
    ecg_channel: int | None = None,
    source: str = "raw_channel",
    peak_fraction: float = 0.4,
    peak_width_seconds: float | None = None,
    peak_distance_seconds: float | None = None,
    bandpass_filter: bool = True,
) -> dict[str, Any]:
    array, fs, source_label = _select_ecg_source(
        session, processed_emg, ecg_channel=ecg_channel, source=source
    )
    peak_width_samples = (
        max(1, int(peak_width_seconds * fs)) if peak_width_seconds is not None else None
    )
    peak_distance_samples = (
        max(1, int(peak_distance_seconds * fs))
        if peak_distance_seconds is not None
        else None
    )

    indices = session.emg_adapter.detect_ecg_peaks(
        array,
        sample_frequency=fs,
        peak_fraction=peak_fraction,
        peak_width_samples=peak_width_samples,
        peak_distance_samples=peak_distance_samples,
        bandpass_filter=bandpass_filter,
    )

    detection_parameters = {
        "source": source_label,
        "peak_fraction": peak_fraction,
        "requested_peak_width_seconds": peak_width_seconds,
        "requested_peak_distance_seconds": peak_distance_seconds,
        "effective_peak_width_samples": peak_width_samples,
        "effective_peak_distance_samples": peak_distance_samples,
        "bandpass_filter": bandpass_filter,
    }
    events = [
        Event(
            name="ecg_peak",
            modality="emg",
            time=float(index) / fs,
            sample_index=int(index),
            metadata=dict(detection_parameters),
        )
        for index in indices
    ]
    session.add_events("ecg_peaks", events)

    count_result = ParameterResult(
        name="ecg_peak_count",
        value=float(len(indices)),
        modality="emg",
        method="resurfemg.detect_ecg_peaks",
        metadata=detection_parameters,
    )

    _record_step(
        session,
        "emg.ecg_detect_peaks",
        metadata=_upstream_metadata(
            source_function="resurfemg.preprocessing.ecg_removal.detect_ecg_peaks",
            operation="emg.ecg_detect_peaks",
            parameters=detection_parameters,
        ),
    )
    return {
        "ecg_peak_indices": indices,
        "ecg_peak_events": events,
        "ecg_peak_count_result": count_result,
    }


def _build_gate_mask(
    n_samples: int, peak_indices: Any, *, gate_width_samples: int
) -> np.ndarray:
    """A boolean mask marking the (clipped-to-bounds) gated region around
    each peak. Purely descriptive - built from the same effective gate
    width used for the cleaned array, but never fed back into it."""

    mask = np.zeros(n_samples, dtype=bool)
    half_width = gate_width_samples // 2
    for peak in peak_indices:
        start = max(0, int(peak) - half_width)
        end = min(n_samples, int(peak) + half_width)
        mask[start:end] = True
    return mask


def _update_session_after_ecg_removal(
    session: M3Session,
    processed_emg_after_ecg: dict[str, Any],
) -> None:
    """Update `session.processed["emg"]` and the `EMGRecording` filtered/
    envelope fields so existing breath detection operates on the
    ECG-cleaned data (mirrors what `M3Session.preprocess_emg` does)."""

    session.processed["emg"] = processed_emg_after_ecg
    if session.emg is not None:
        session.emg.filtered = processed_emg_after_ecg.get("filtered")
        session.emg.envelope = processed_emg_after_ecg.get("envelope")


@register_step(
    "emg.ecg_gating",
    reads={
        "session": "session",
        "processed_emg": "processed_emg",
        "ecg_peak_indices": "ecg_peak_indices",
    },
    writes=(
        "ecg_gated_emg",
        "processed_emg_after_ecg",
        "ecg_gated_signal",
        "ecg_gate_mask_result",
    ),
    summary="Remove ECG peaks from EMG by gating (zero/interpolate/replace).",
)
def ecg_gating(
    session: M3Session,
    processed_emg: Any,
    ecg_peak_indices: Any,
    *,
    source: str = "filtered",
    gate_width_seconds: float | None = None,
    gate_width_samples: int | None = None,
    fill_method: int = 1,
    envelope_window_seconds: float | None = None,
) -> dict[str, Any]:
    if gate_width_seconds is not None and gate_width_samples is not None:
        raise ValueError(
            "emg.ecg_gating: set only one of gate_width_seconds or "
            "gate_width_samples, not both."
        )
    if source not in processed_emg:
        raise ValueError(
            f"emg.ecg_gating source {source!r} is not present in processed_emg; "
            f"available keys: {sorted(processed_emg.keys())}."
        )

    array = np.asarray(processed_emg[source], dtype=float)
    fs = float(processed_emg["fs"])
    if gate_width_seconds is not None:
        effective_gate_width_samples = max(1, int(gate_width_seconds * fs))
    elif gate_width_samples is not None:
        effective_gate_width_samples = gate_width_samples
    else:
        effective_gate_width_samples = 205  # resurfemg's own default

    gated = session.emg_adapter.gate_ecg(
        array,
        ecg_peak_indices,
        gate_width_samples=effective_gate_width_samples,
        fill_method=fill_method,
    )
    gate_mask = _build_gate_mask(
        len(array), ecg_peak_indices, gate_width_samples=effective_gate_width_samples
    )

    original_window_seconds = (processed_emg.get("filter") or {}).get(
        "envelope_window_seconds"
    )
    effective_envelope_window_seconds = (
        envelope_window_seconds
        if envelope_window_seconds is not None
        else original_window_seconds
    )
    envelope = processed_emg.get("envelope")
    if effective_envelope_window_seconds is not None:
        envelope_window_samples = max(1, int(effective_envelope_window_seconds * fs))
        envelope = rolling_arv(gated, window_length=envelope_window_samples)

    processed_emg_after_ecg = {**processed_emg, "filtered": gated, "envelope": envelope}
    _update_session_after_ecg_removal(session, processed_emg_after_ecg)

    label, unit = _processed_channel_label_and_unit(processed_emg)
    time = np.arange(len(gated), dtype=float) / fs
    gating_parameters = {
        "source": source,
        "requested_gate_width_seconds": gate_width_seconds,
        "requested_gate_width_samples": gate_width_samples,
        "effective_gate_width_samples": effective_gate_width_samples,
        "fill_method": fill_method,
        "effective_envelope_window_seconds": effective_envelope_window_seconds,
    }
    ecg_gated_signal = Signal(
        values=gated,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_ecg_gated",
        modality="emg",
        channel=label,
        source="resurfemg",
        processing_state="filtered",
        derived_from="processed",
        method="resurfemg.gating",
        metadata=dict(gating_parameters),
    )
    session.signals.add(ecg_gated_signal)

    gate_mask_result = ParameterResult(
        name="ecg_gate_mask",
        value=gate_mask,
        modality="emg",
        channel=label,
        method="resurfemg.gating",
        metadata=dict(gating_parameters),
    )

    _record_step(
        session,
        "emg.ecg_gating",
        metadata=_upstream_metadata(
            source_function="resurfemg.preprocessing.ecg_removal.gating",
            operation="emg.ecg_gating",
            parameters=gating_parameters,
        ),
    )
    return {
        "ecg_gated_emg": gated,
        "processed_emg_after_ecg": processed_emg_after_ecg,
        "ecg_gated_signal": ecg_gated_signal,
        "ecg_gate_mask_result": gate_mask_result,
    }


@register_step(
    "emg.ecg_wavelet_denoising",
    reads={
        "session": "session",
        "processed_emg": "processed_emg",
        "ecg_peak_indices": "ecg_peak_indices",
    },
    writes=(
        "ecg_wavelet_cleaned_emg",
        "processed_emg_after_ecg",
        "ecg_wavelet_cleaned_signal",
        "wavelet_decomposition_result",
        "wavelet_thresholds_result",
        "wavelet_gate_mask_result",
    ),
    summary="Remove ECG peaks from EMG by a-trous wavelet shrinkage.",
)
def ecg_wavelet_denoising(
    session: M3Session,
    processed_emg: Any,
    ecg_peak_indices: Any,
    *,
    source: str = "filtered",
    hard_thresholding: bool = True,
    levels: int = 4,
    wavelet_type: str = "db2",
    fixed_threshold: float = 4.5,
    envelope_window_seconds: float | None = None,
) -> dict[str, Any]:
    if source not in processed_emg:
        raise ValueError(
            f"emg.ecg_wavelet_denoising source {source!r} is not present in "
            f"processed_emg; available keys: {sorted(processed_emg.keys())}."
        )

    array = np.asarray(processed_emg[source], dtype=float)
    fs = float(processed_emg["fs"])
    original_length = len(array)
    padded_length = int(np.ceil(original_length / 2**levels) * 2**levels)

    cleaned, decomposition, thresholds, gate_mask = (
        session.emg_adapter.wavelet_denoise_ecg(
            array,
            ecg_peak_indices,
            sample_frequency=fs,
            hard_thresholding=hard_thresholding,
            levels=levels,
            wavelet_type=wavelet_type,
            fixed_threshold=fixed_threshold,
        )
    )

    original_window_seconds = (processed_emg.get("filter") or {}).get(
        "envelope_window_seconds"
    )
    effective_envelope_window_seconds = (
        envelope_window_seconds
        if envelope_window_seconds is not None
        else original_window_seconds
    )
    envelope = processed_emg.get("envelope")
    if effective_envelope_window_seconds is not None:
        envelope_window_samples = max(1, int(effective_envelope_window_seconds * fs))
        envelope = rolling_arv(cleaned, window_length=envelope_window_samples)

    processed_emg_after_ecg = {
        **processed_emg,
        "filtered": cleaned,
        "envelope": envelope,
    }
    _update_session_after_ecg_removal(session, processed_emg_after_ecg)

    label, unit = _processed_channel_label_and_unit(processed_emg)
    time = np.arange(len(cleaned), dtype=float) / fs
    wavelet_parameters = {
        "source": source,
        "hard_thresholding": hard_thresholding,
        "levels": levels,
        "wavelet_type": wavelet_type,
        "fixed_threshold": fixed_threshold,
        "original_length": original_length,
        "padded_length": padded_length,
        "effective_envelope_window_seconds": effective_envelope_window_seconds,
    }
    ecg_wavelet_cleaned_signal = Signal(
        values=cleaned,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_ecg_wavelet_cleaned",
        modality="emg",
        channel=label,
        source="resurfemg",
        processing_state="filtered",
        derived_from="processed",
        method="resurfemg.wavelet_denoising",
        metadata=dict(wavelet_parameters),
    )
    session.signals.add(ecg_wavelet_cleaned_signal)

    decomposition_result = ParameterResult(
        name="ecg_wavelet_decomposition",
        value=decomposition,
        modality="emg",
        channel=label,
        method="resurfemg.wavelet_denoising",
        metadata={**wavelet_parameters, "axes": ["level", "sample"]},
    )
    thresholds_result = ParameterResult(
        name="ecg_wavelet_thresholds",
        value=thresholds,
        modality="emg",
        channel=label,
        method="resurfemg.wavelet_denoising",
        metadata={**wavelet_parameters, "axes": ["level", "sample"]},
    )
    gate_mask_result = ParameterResult(
        name="ecg_wavelet_gate_mask",
        value=gate_mask,
        modality="emg",
        channel=label,
        method="resurfemg.wavelet_denoising",
        metadata=dict(wavelet_parameters),
    )

    _record_step(
        session,
        "emg.ecg_wavelet_denoising",
        metadata=_upstream_metadata(
            source_function="resurfemg.preprocessing.ecg_removal.wavelet_denoising",
            operation="emg.ecg_wavelet_denoising",
            parameters=wavelet_parameters,
        ),
    )
    return {
        "ecg_wavelet_cleaned_emg": cleaned,
        "processed_emg_after_ecg": processed_emg_after_ecg,
        "ecg_wavelet_cleaned_signal": ecg_wavelet_cleaned_signal,
        "wavelet_decomposition_result": decomposition_result,
        "wavelet_thresholds_result": thresholds_result,
        "wavelet_gate_mask_result": gate_mask_result,
    }


# --- event_detection ------------------------------------------------------


@register_step(
    "emg.detect_ventilator_breath",
    reads={"ventilator_signals": "ventilator_signals"},
    writes=("ventilator_breath_indices",),
    summary="Detect ventilator breaths from the ventilator volume channel.",
)
def detect_ventilator_breath(
    ventilator_signals: Any, *, breath_width_seconds: float = 0.5
) -> dict[str, Any]:
    import numpy as np

    volume = ventilator_signals["volume"]
    fs = float(ventilator_signals["fs"])
    width_samples = max(1, int(breath_width_seconds * fs))
    indices = detect_ventilator_breath_peaks(
        volume,
        start_index=0,
        end_index=len(volume) - 1,
        width_samples=width_samples,
    )
    return {"ventilator_breath_indices": np.asarray(indices, dtype=int)}


@register_step(
    "emg.find_occluded_breaths",
    reads={"ventilator_signals": "ventilator_signals"},
    writes=("pocc_indices",),
    summary="Detect occluded (Pocc) breaths from the ventilator pressure channel.",
)
def find_occluded_breaths(
    ventilator_signals: Any, *, peep: float | None = None
) -> dict[str, Any]:
    import numpy as np

    pressure = ventilator_signals["pressure"]
    fs = float(ventilator_signals["fs"])
    if peep is None:
        peep = float(np.nanmedian(pressure))
    indices = detect_occluded_breath_peaks(
        pressure,
        sample_frequency=fs,
        peep=peep,
    )
    return {"pocc_indices": np.asarray(indices, dtype=int)}


@register_step(
    "emg.pocc_intervals",
    reads={
        "session": "session",
        "ventilator_signals": "ventilator_signals",
        "pocc_indices": "pocc_indices",
    },
    writes=(
        "pocc_start_indices",
        "pocc_end_indices",
        "pocc_interval_validity",
        "pocc_events",
    ),
    summary="Find Pocc manoeuvre start/end indices from the pressure channel.",
)
def pocc_intervals(
    session: M3Session,
    ventilator_signals: Any,
    pocc_indices: Any,
    *,
    peep: float | None = None,
) -> dict[str, Any]:
    pressure = np.asarray(ventilator_signals["pressure"], dtype=float)
    fs = float(ventilator_signals["fs"])
    peaks = np.asarray(pocc_indices, dtype=int)

    # Same PEEP rule as emg.find_occluded_breaths, so pocc_indices (detected
    # against this same baseline) and these intervals stay consistent.
    effective_peep = peep if peep is not None else float(np.nanmedian(pressure))
    baseline = np.full(pressure.shape, effective_peep)

    starts, ends, valid_starts, valid_ends, valid_peaks = onoff_from_baseline_crossings(
        pressure, baseline, peaks
    )

    events: list[BreathEvent] = []
    for index, peak in enumerate(peaks):
        events.append(
            BreathEvent(
                modality="pressure",
                start_time=float(starts[index]) / fs,
                end_time=float(ends[index]) / fs,
                peak_time=float(peak) / fs,
                start_index=int(starts[index]),
                peak_index=int(peak),
                end_index=int(ends[index]),
                sample_frequency=fs,
                signal_name="pressure",
                source="m3resp.processing.intervals.onoff_from_baseline_crossings",
                metadata={
                    "event_type": "pocc",
                    "peep": effective_peep,
                    "valid": bool(valid_peaks[index]),
                    "valid_start": bool(valid_starts[index]),
                    "valid_end": bool(valid_ends[index]),
                },
            )
        )
    session.add_events("pocc_breaths", events)

    _record_step(
        session,
        "emg.pocc_intervals",
        metadata=_upstream_metadata(
            source_function="m3resp.processing.intervals.onoff_from_baseline_crossings",
            operation="emg.pocc_intervals",
            parameters={"peep": effective_peep, "requested_peep": peep},
            source_package="m3resp",
            implementation="m3resp.processing.intervals",
        ),
    )
    return {
        "pocc_start_indices": starts,
        "pocc_end_indices": ends,
        "pocc_interval_validity": np.asarray(valid_peaks, dtype=bool),
        "pocc_events": events,
    }


@register_step(
    "emg.pocc_time_product",
    reads={
        "session": "session",
        "ventilator_signals": "ventilator_signals",
        "pocc_start_indices": "pocc_start_indices",
        "pocc_end_indices": "pocc_end_indices",
    },
    writes=("pocc_time_products", "pocc_time_product_result"),
    summary="Compute the pressure-time product for each Pocc manoeuvre.",
)
def pocc_time_product(
    session: M3Session,
    ventilator_signals: Any,
    pocc_start_indices: Any,
    pocc_end_indices: Any,
    *,
    peep: float | None = None,
) -> dict[str, Any]:
    pressure = np.asarray(ventilator_signals["pressure"], dtype=float)
    fs = float(ventilator_signals["fs"])
    effective_peep = peep if peep is not None else float(np.nanmedian(pressure))
    baseline = np.full(pressure.shape, effective_peep)

    time_products = window_integral(
        pressure, fs, pocc_start_indices, pocc_end_indices, baseline
    )

    pressure_unit = ventilator_signals.get("unit") or "cmH2O"
    parameters = {"peep": effective_peep, "requested_peep": peep}
    result = ParameterResult(
        name="pocc_time_product",
        value=time_products,
        modality="pressure",
        unit=f"{pressure_unit}*s",
        method="m3resp.processing.metrics.window_integral",
        metadata={
            **parameters,
            "start_indices": np.asarray(pocc_start_indices, dtype=int).tolist(),
            "end_indices": np.asarray(pocc_end_indices, dtype=int).tolist(),
        },
    )

    _record_step(
        session,
        "emg.pocc_time_product",
        metadata=_upstream_metadata(
            source_function="m3resp.processing.metrics.window_integral",
            operation="emg.pocc_time_product",
            parameters=parameters,
            source_package="m3resp",
            implementation="m3resp.processing.metrics",
        ),
    )
    return {"pocc_time_products": time_products, "pocc_time_product_result": result}


@register_step(
    "emg.onoffpeak_baseline_crossing",
    reads={
        "processed_emg": "processed_emg",
        "baseline": "baseline",
        "peak_indices": "peak_indices",
    },
    writes=("start_indices", "end_indices"),
    summary="Find EMG breath on/offset indices by baseline crossing.",
)
def onoffpeak_baseline_crossing(
    processed_emg: Any, baseline: Any, peak_indices: Any
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    start_indices, end_indices, *_rest = onoff_from_baseline_crossings(
        envelope, baseline, peak_indices
    )
    return {"start_indices": start_indices, "end_indices": end_indices}


@register_step(
    "emg.onoffpeak_slope_extrapolation",
    reads={"processed_emg": "processed_emg", "peak_indices": "peak_indices"},
    writes=("onoffpeak_slope_result",),
    summary="Find EMG breath on/offset indices by slope extrapolation.",
)
def onoffpeak_slope_extrapolation(
    processed_emg: Any, peak_indices: Any, *, slope_window_seconds: float = 0.5
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    slope_window_samples = max(1, int(slope_window_seconds * fs))
    result = onoff_from_slope(
        envelope,
        sample_frequency=fs,
        peak_indices=peak_indices,
        slope_window=slope_window_samples,
    )
    return {"onoffpeak_slope_result": result}


# --- features ---------------------------------------------------------


@register_step(
    "emg.time_to_peak",
    reads={
        "processed_emg": "processed_emg",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
    },
    writes=("time_to_peak",),
    summary="Compute EMG breath time-to-peak.",
)
def time_to_peak(
    processed_emg: Any, start_indices: Any, end_indices: Any
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    return {"time_to_peak": _time_to_peak(envelope, start_indices, end_indices)}


@register_step(
    "emg.pseudo_slope",
    reads={
        "processed_emg": "processed_emg",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
    },
    writes=("pseudo_slope",),
    summary="Compute EMG breath pseudo-slope.",
)
def pseudo_slope(
    processed_emg: Any, start_indices: Any, end_indices: Any
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    return {"pseudo_slope": _pseudo_slope(envelope, start_indices, end_indices)}


@register_step(
    "emg.amplitude",
    reads={
        "processed_emg": "processed_emg",
        "peak_indices": "peak_indices",
        "baseline": "baseline",
    },
    writes=("amplitude",),
    summary="Compute EMG breath amplitude above baseline.",
)
def amplitude(processed_emg: Any, peak_indices: Any, baseline: Any) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    return {"amplitude": amplitude_at_peaks(envelope, peak_indices, baseline)}


@register_step(
    "emg.time_product",
    reads={
        "processed_emg": "processed_emg",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "baseline": "baseline",
    },
    writes=("time_product",),
    summary="Compute EMG breath time-product (area above baseline).",
)
def time_product(
    processed_emg: Any, start_indices: Any, end_indices: Any, baseline: Any
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    return {
        "time_product": window_integral(
            envelope, fs, start_indices, end_indices, baseline
        )
    }


@register_step(
    "emg.area_under_baseline",
    reads={
        "processed_emg": "processed_emg",
        "peak_indices": "peak_indices",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "baseline": "baseline",
    },
    writes=("area_under_baseline",),
    summary="Compute EMG area under baseline around each breath peak.",
)
def area_under_baseline(
    processed_emg: Any,
    peak_indices: Any,
    start_indices: Any,
    end_indices: Any,
    baseline: Any,
    *,
    window_seconds: float = 5.0,
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    window_samples = max(1, int(window_seconds * fs))
    result = _area_under_baseline(
        envelope,
        fs,
        peak_indices,
        start_indices,
        end_indices,
        window_samples,
        baseline,
    )
    return {"area_under_baseline": result}


@register_step(
    "emg.respiratory_rate",
    reads={"peak_indices": "peak_indices", "processed_emg": "processed_emg"},
    writes=("respiratory_rate",),
    summary="Compute respiratory rate from detected EMG breath peaks.",
)
def respiratory_rate(peak_indices: Any, processed_emg: Any) -> dict[str, Any]:
    fs = float(processed_emg["fs"])
    return {"respiratory_rate": respiratory_rate_from_indices(peak_indices, fs)}


@register_step(
    "emg.ventilator_respiratory_rate",
    reads={
        "ventilator_breath_indices": "ventilator_breath_indices",
        "ventilator_signals": "ventilator_signals",
    },
    writes=("ventilator_respiratory_rate",),
    summary="Compute respiratory rate from detected ventilator breaths.",
)
def ventilator_respiratory_rate(
    ventilator_breath_indices: Any, ventilator_signals: Any
) -> dict[str, Any]:
    fs = float(ventilator_signals["fs"])
    return {
        "ventilator_respiratory_rate": respiratory_rate_from_indices(
            ventilator_breath_indices, fs
        )
    }


# --- quality_assessment -------------------------------------------------


@register_step(
    "emg.snr_pseudo",
    reads={
        "session": "session",
        "processed_emg": "processed_emg",
        "peak_indices": "peak_indices",
        "baseline": "baseline",
    },
    writes=("snr_pseudo",),
    summary="Compute a pseudo signal-to-noise ratio for detected EMG breaths.",
)
def snr_pseudo(
    session: M3Session, processed_emg: Any, peak_indices: Any, baseline: Any
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    result = session.emg_adapter.snr_pseudo(
        envelope, peak_indices, baseline, sample_frequency=fs
    )
    return {"snr_pseudo": result}


@register_step(
    "emg.percentage_under_baseline",
    reads={
        "session": "session",
        "processed_emg": "processed_emg",
        "peak_indices": "peak_indices",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "baseline": "baseline",
    },
    writes=("percentage_under_baseline",),
    summary="Compute the percentage of each EMG breath spent under baseline.",
)
def percentage_under_baseline(
    session: M3Session,
    processed_emg: Any,
    peak_indices: Any,
    start_indices: Any,
    end_indices: Any,
    baseline: Any,
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    result = session.emg_adapter.percentage_under_baseline(
        envelope,
        peak_indices,
        start_indices,
        end_indices,
        baseline,
        sample_frequency=fs,
    )
    return {"percentage_under_baseline": result}


@register_step(
    "emg.detect_local_high_aub",
    reads={"session": "session", "area_under_baseline": "area_under_baseline"},
    writes=("detect_local_high_aub",),
    summary="Flag EMG breaths with locally elevated area-under-baseline.",
)
def detect_local_high_aub(
    session: M3Session, area_under_baseline: Any
) -> dict[str, Any]:
    aubs = area_under_baseline[0]
    return {"detect_local_high_aub": session.emg_adapter.detect_local_high_aub(aubs)}


@register_step(
    "emg.detect_extreme_time_products",
    reads={"session": "session", "time_product": "time_product"},
    writes=("detect_extreme_time_products",),
    summary="Flag EMG breaths with extreme time-products.",
)
def detect_extreme_time_products(
    session: M3Session, time_product: Any
) -> dict[str, Any]:
    result = session.emg_adapter.detect_extreme_time_products(time_product)
    return {"detect_extreme_time_products": result}


@register_step(
    "emg.detect_non_consecutive_manoeuvres",
    reads={
        "session": "session",
        "ventilator_breath_indices": "ventilator_breath_indices",
        "pocc_indices": "pocc_indices",
    },
    writes=("detect_non_consecutive_manoeuvres",),
    summary="Flag non-consecutive occlusion manoeuvres against ventilator breaths.",
)
def detect_non_consecutive_manoeuvres(
    session: M3Session, ventilator_breath_indices: Any, pocc_indices: Any
) -> dict[str, Any]:
    result = session.emg_adapter.detect_non_consecutive_manoeuvres(
        ventilator_breath_indices, pocc_indices
    )
    return {"detect_non_consecutive_manoeuvres": result}


@register_step(
    "emg.evaluate_bell_curve_error",
    reads={
        "session": "session",
        "peak_indices": "peak_indices",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "processed_emg": "processed_emg",
        "time_product": "time_product",
    },
    writes=("evaluate_bell_curve_error",),
    summary="Score how well each EMG breath matches a bell-curve shape.",
)
def evaluate_bell_curve_error(
    session: M3Session,
    peak_indices: Any,
    start_indices: Any,
    end_indices: Any,
    processed_emg: Any,
    time_product: Any,
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    result = session.emg_adapter.evaluate_bell_curve_error(
        peak_indices,
        start_indices,
        end_indices,
        envelope,
        time_product,
        sample_frequency=fs,
    )
    return {"evaluate_bell_curve_error": result}


@register_step(
    "emg.evaluate_event_timing",
    reads={
        "session": "session",
        "peak_indices": "peak_indices",
        "processed_emg": "processed_emg",
        "ventilator_breath_indices": "ventilator_breath_indices",
        "ventilator_signals": "ventilator_signals",
    },
    writes=("evaluate_event_timing",),
    summary="Score the timing agreement between EMG and ventilator breaths.",
)
def evaluate_event_timing(
    session: M3Session,
    peak_indices: Any,
    processed_emg: Any,
    ventilator_breath_indices: Any,
    ventilator_signals: Any,
) -> dict[str, Any]:
    fs = float(processed_emg["fs"])
    vent_fs = float(ventilator_signals["fs"])
    paired_count = min(len(peak_indices), len(ventilator_breath_indices))
    result = session.emg_adapter.evaluate_event_timing(
        peak_indices[:paired_count] / fs,
        ventilator_breath_indices[:paired_count] / vent_fs,
    )
    return {"evaluate_event_timing": result}


@register_step(
    "emg.evaluate_respiratory_rates",
    reads={
        "session": "session",
        "peak_indices": "peak_indices",
        "processed_emg": "processed_emg",
        "ventilator_respiratory_rate": "ventilator_respiratory_rate",
    },
    writes=("evaluate_respiratory_rates",),
    summary="Score agreement between EMG-derived and ventilator-derived respiratory rate.",
)
def evaluate_respiratory_rates(
    session: M3Session,
    peak_indices: Any,
    processed_emg: Any,
    ventilator_respiratory_rate: Any,
) -> dict[str, Any]:
    fs = float(processed_emg["fs"])
    envelope = processed_emg["envelope"]
    rr_vent = ventilator_respiratory_rate[0]
    result = session.emg_adapter.evaluate_respiratory_rates(
        peak_indices, len(envelope) / fs, rr_vent
    )
    return {"evaluate_respiratory_rates": result}


# --- event normalization -------------------------------------------------


@register_step(
    "emg.normalize_ventilator_breaths",
    reads={
        "ventilator_breath_indices": "ventilator_breath_indices",
        "ventilator_signals": "ventilator_signals",
        "session": "session",
    },
    writes=(),
    summary="Normalize detected ventilator breath indices into session events.",
)
def normalize_ventilator_breaths(
    ventilator_breath_indices: Any,
    ventilator_signals: Any,
    session: M3Session,
    *,
    breath_width_seconds: float = 0.5,
) -> dict[str, Any]:
    fs = float(ventilator_signals["fs"])
    events = [
        _normalize_ventilator_breath(
            detection, fs=fs, width_seconds=breath_width_seconds
        )
        for detection in _iter_ventilator_detections(ventilator_breath_indices)
    ]
    session.add_events("ventilator_breaths", events)
    return {}
