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
from m3resp.data import ParameterResult, QualityFlag, Signal
from m3resp.data.quality import Severity
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


# shared per-item identity helpers for native quality results/flags --------
def _require_equal_length(**named_arrays: Any) -> None:
    """Raise a clear error instead of silently truncating with
    `min(len(...))` when paired arrays disagree in length (plan Phase 5.4:
    "Do not truncate arrays... without reporting unmatched events")."""

    lengths = {name: len(array) for name, array in named_arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Arrays must have equal length; got {lengths}.")


def _breath_metadata(peak_index: Any, *, fs: float | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"peak_sample_index": int(peak_index)}
    if fs is not None:
        metadata["peak_time"] = float(peak_index) / fs
    return metadata


def _per_breath_flags(
    name: str,
    valid: Any,
    *,
    modality: str,
    peak_indices: Any,
    severity: Severity = "info",
    fs: float | None = None,
    threshold: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[QualityFlag]:
    """One `QualityFlag` per breath - `breath_id=str(position)` until a
    stable event ID is available, with the source peak sample index
    recorded in metadata (plan Phase 5.4)."""

    _require_equal_length(valid=valid, peak_indices=peak_indices)
    flags = []
    for position, (is_valid, peak_index) in enumerate(zip(valid, peak_indices)):
        metadata = _breath_metadata(peak_index, fs=fs)
        if extra_metadata:
            metadata.update(extra_metadata)
        flags.append(
            QualityFlag(
                name=name,
                passed=bool(is_valid),
                severity=severity,
                modality=modality,
                breath_id=str(position),
                threshold=threshold,
                metadata=metadata,
            )
        )
    return flags


def _per_breath_results(
    name: str,
    values: Any,
    *,
    modality: str,
    peak_indices: Any,
    unit: str | None = None,
    method: str | None = None,
    fs: float | None = None,
    extra_metadata_per_item: list[dict[str, Any]] | None = None,
) -> list[ParameterResult]:
    _require_equal_length(values=values, peak_indices=peak_indices)
    results = []
    for position, (value, peak_index) in enumerate(zip(values, peak_indices)):
        metadata = _breath_metadata(peak_index, fs=fs)
        if extra_metadata_per_item is not None:
            metadata.update(extra_metadata_per_item[position])
        results.append(
            ParameterResult(
                name=name,
                value=value if np.ndim(value) > 0 else float(value),
                modality=modality,
                unit=unit,
                breath_id=str(position),
                method=method,
                metadata=metadata,
            )
        )
    return results


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
    # Array-valued, so this reuses the shared parameter_result_arrays.npz
    # exporter (plan Phase 6.3) rather than a competing EMG-specific one -
    # session.export_summary() already routes any array-valued
    # ParameterResult there.
    session.parameter_results.add(gate_mask_result)

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
    # All three are array-valued (decomposition/thresholds are 2D: level x
    # sample), so they reuse the shared parameter_result_arrays.npz exporter
    # (plan Phase 6.3) via session.export_summary() rather than a competing
    # EMG-specific array format.
    for array_result in (decomposition_result, thresholds_result, gate_mask_result):
        session.parameter_results.add(array_result)

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


_POCC_CRITERIA_ROW_NAMES = ("dp_up_10", "dp_up_90", "dp_up_90_norm")


@register_step(
    "emg.pocc_quality",
    reads={
        "session": "session",
        "ventilator_signals": "ventilator_signals",
        "pocc_indices": "pocc_indices",
        "pocc_end_indices": "pocc_end_indices",
        "pocc_time_products": "pocc_time_products",
    },
    writes=(
        "pocc_quality",
        "pocc_quality_criteria",
        "pocc_quality_results",
        "pocc_quality_flags",
    ),
    summary="Evaluate Pocc manoeuvre quality from the pressure upslope (Warnaar et al. 2024).",
)
def pocc_quality(
    session: M3Session,
    ventilator_signals: Any,
    pocc_indices: Any,
    pocc_end_indices: Any,
    pocc_time_products: Any,
    *,
    dp_up_10_threshold: float = 0.0,
    dp_up_90_threshold: float = 2.0,
    dp_up_90_norm_threshold: float = 0.8,
) -> dict[str, Any]:
    pressure = np.asarray(ventilator_signals["pressure"], dtype=float)
    pressure_unit = ventilator_signals.get("unit") or "cmH2O"

    valid, criteria = session.emg_adapter.pocc_quality(
        pressure,
        pocc_indices,
        pocc_end_indices,
        pocc_time_products,
        dp_up_10_threshold=dp_up_10_threshold,
        dp_up_90_threshold=dp_up_90_threshold,
        dp_up_90_norm_threshold=dp_up_90_norm_threshold,
    )

    thresholds_by_row = {
        "dp_up_10": dp_up_10_threshold,
        "dp_up_90": dp_up_90_threshold,
        "dp_up_90_norm": dp_up_90_norm_threshold,
    }
    flags = _per_breath_flags(
        "pocc_quality",
        valid,
        modality="pressure",
        peak_indices=pocc_indices,
        extra_metadata={"pressure_sample_index_end": None},
    )
    # Link each flag to its Pocc end index too, not just its peak.
    for flag, end_index in zip(flags, pocc_end_indices):
        flag.metadata["pressure_sample_index_end"] = int(end_index)

    results: list[ParameterResult] = []
    for row_name, row_values in zip(_POCC_CRITERIA_ROW_NAMES, criteria):
        results.extend(
            _per_breath_results(
                f"pocc_quality_{row_name}",
                row_values,
                modality="pressure",
                peak_indices=pocc_indices,
                unit=pressure_unit,
                method="resurfemg.pocc_quality",
                extra_metadata_per_item=[
                    {"threshold": thresholds_by_row[row_name], "criterion": row_name}
                    for _ in row_values
                ],
            )
        )

    for result in results:
        session.parameter_results.add(result)
    for flag in flags:
        session.quality.add(flag)

    _record_step(
        session,
        "emg.pocc_quality",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.pocc_quality",
            operation="emg.pocc_quality",
            parameters={
                "dp_up_10_threshold": dp_up_10_threshold,
                "dp_up_90_threshold": dp_up_90_threshold,
                "dp_up_90_norm_threshold": dp_up_90_norm_threshold,
            },
        ),
    )
    return {
        "pocc_quality": valid,
        "pocc_quality_criteria": criteria,
        "pocc_quality_results": results,
        "pocc_quality_flags": flags,
    }


@register_step(
    "emg.interpeak_dist",
    reads={
        "session": "session",
        "ecg_peak_indices": "ecg_peak_indices",
        "peak_indices": "peak_indices",
        "processed_emg": "processed_emg",
    },
    writes=("interpeak_dist", "interpeak_dist_result", "interpeak_dist_flag"),
    summary="Check the ECG-to-EMG median interpeak distance ratio (Warnaar et al. 2024).",
)
def interpeak_dist(
    session: M3Session,
    ecg_peak_indices: Any,
    peak_indices: Any,
    processed_emg: Any,
    *,
    threshold: float = 1.1,
) -> dict[str, Any]:
    # Upstream compares distances in raw samples, which is only meaningful
    # if both peak sets share a time base - both are indices into the same
    # processed_emg channel here, so that holds without conversion.
    fs = float(processed_emg["fs"])
    valid = session.emg_adapter.interpeak_distance(
        ecg_peak_indices, peak_indices, threshold=threshold
    )

    ecg_peaks = np.asarray(ecg_peak_indices, dtype=int)
    emg_peaks = np.asarray(peak_indices, dtype=int)
    ecg_median_samples = float(np.median(np.diff(ecg_peaks)))
    emg_median_samples = float(np.median(np.diff(emg_peaks)))
    # Degenerate peak sets (e.g. duplicate indices) give a zero median
    # interval; match upstream's own behavior (a RuntimeWarning plus an inf/
    # nan ratio, not a raised error) via NumPy division instead of Python's
    # float division, which would raise ZeroDivisionError here.
    ratio = float(np.divide(emg_median_samples, ecg_median_samples))

    shared_metadata = {
        "threshold": threshold,
        "time_base": "shared (both peak sets index the same EMG channel)",
        "sample_frequency": fs,
    }
    results = [
        ParameterResult(
            name="interpeak_dist_ecg_median",
            value=ecg_median_samples / fs,
            modality="emg",
            unit="s",
            method="resurfemg.interpeak_dist",
            metadata=dict(shared_metadata),
        ),
        ParameterResult(
            name="interpeak_dist_emg_median",
            value=emg_median_samples / fs,
            modality="emg",
            unit="s",
            method="resurfemg.interpeak_dist",
            metadata=dict(shared_metadata),
        ),
        ParameterResult(
            name="interpeak_dist_ratio",
            value=ratio,
            modality="emg",
            method="resurfemg.interpeak_dist",
            metadata=dict(shared_metadata),
        ),
    ]
    flag = QualityFlag(
        name="interpeak_dist",
        passed=bool(valid),
        severity="info",
        modality="emg",
        value=ratio,
        threshold=threshold,
        metadata=dict(shared_metadata),
    )

    for result in results:
        session.parameter_results.add(result)
    session.quality.add(flag)

    _record_step(
        session,
        "emg.interpeak_dist",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.interpeak_dist",
            operation="emg.interpeak_dist",
            parameters={"threshold": threshold},
        ),
    )
    return {
        "interpeak_dist": valid,
        "interpeak_dist_result": results,
        "interpeak_dist_flag": flag,
    }


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
    writes=("snr_pseudo", "snr_pseudo_results", "snr_pseudo_flags"),
    summary="Compute a pseudo signal-to-noise ratio for detected EMG breaths.",
)
def snr_pseudo(
    session: M3Session,
    processed_emg: Any,
    peak_indices: Any,
    baseline: Any,
    *,
    minimum_snr: float | None = None,
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    result = session.emg_adapter.snr_pseudo(
        envelope, peak_indices, baseline, sample_frequency=fs
    )

    results = _per_breath_results(
        "snr_pseudo",
        result,
        modality="emg",
        peak_indices=peak_indices,
        method="resurfemg.snr_pseudo",
        fs=fs,
    )
    # A measurement only becomes a criterion when a threshold is actually
    # configured - no invented pass/fail otherwise (see
    # m3resp.processing.quality.quality_flag_from_result's docstring).
    flags = (
        _per_breath_flags(
            "snr_pseudo",
            result >= minimum_snr,
            modality="emg",
            peak_indices=peak_indices,
            fs=fs,
            threshold=minimum_snr,
        )
        if minimum_snr is not None
        else []
    )

    for parameter_result in results:
        session.parameter_results.add(parameter_result)
    for flag in flags:
        session.quality.add(flag)

    _record_step(
        session,
        "emg.snr_pseudo",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.snr_pseudo",
            operation="emg.snr_pseudo",
            parameters={"minimum_snr": minimum_snr},
        ),
    )
    return {
        "snr_pseudo": result,
        "snr_pseudo_results": results,
        "snr_pseudo_flags": flags,
    }


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
    writes=(
        "percentage_under_baseline",
        "percentage_under_baseline_results",
        "percentage_under_baseline_flags",
    ),
    summary="Compute the percentage of each EMG breath spent under baseline.",
)
def percentage_under_baseline(
    session: M3Session,
    processed_emg: Any,
    peak_indices: Any,
    start_indices: Any,
    end_indices: Any,
    baseline: Any,
    *,
    aub_window_seconds: float | None = None,
    aub_threshold: float = 40.0,
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    aub_window_samples = (
        max(1, int(aub_window_seconds * fs)) if aub_window_seconds is not None else None
    )
    result = session.emg_adapter.percentage_under_baseline(
        envelope,
        peak_indices,
        start_indices,
        end_indices,
        baseline,
        sample_frequency=fs,
        aub_window_samples=aub_window_samples,
        aub_threshold=aub_threshold,
    )
    valid, percentages, reference_values = result

    results = _per_breath_results(
        "percentage_under_baseline",
        percentages,
        modality="emg",
        peak_indices=peak_indices,
        unit="%",
        method="resurfemg.percentage_under_baseline",
        fs=fs,
        extra_metadata_per_item=[
            {"reference_value": float(reference)} for reference in reference_values
        ],
    )
    flags = _per_breath_flags(
        "percentage_under_baseline",
        valid,
        modality="emg",
        peak_indices=peak_indices,
        fs=fs,
        threshold=aub_threshold,
    )

    for parameter_result in results:
        session.parameter_results.add(parameter_result)
    for flag in flags:
        session.quality.add(flag)

    _record_step(
        session,
        "emg.percentage_under_baseline",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.percentage_under_baseline",
            operation="emg.percentage_under_baseline",
            parameters={
                "aub_window_seconds": aub_window_seconds,
                "aub_threshold": aub_threshold,
                "effective_aub_window_samples": aub_window_samples,
            },
        ),
    )
    return {
        "percentage_under_baseline": result,
        "percentage_under_baseline_results": results,
        "percentage_under_baseline_flags": flags,
    }


@register_step(
    "emg.detect_local_high_aub",
    reads={
        "session": "session",
        "area_under_baseline": "area_under_baseline",
        "peak_indices": "peak_indices",
    },
    writes=(
        "detect_local_high_aub",
        "detect_local_high_aub_flags",
        "detect_local_high_aub_threshold_result",
    ),
    summary="Flag EMG breaths with locally elevated area-under-baseline.",
)
def detect_local_high_aub(
    session: M3Session,
    area_under_baseline: Any,
    peak_indices: Any,
    *,
    threshold_percentile: float = 75.0,
    threshold_factor: float = 4.0,
) -> dict[str, Any]:
    aubs = area_under_baseline[0]
    result = session.emg_adapter.detect_local_high_aub(
        aubs,
        threshold_percentile=threshold_percentile,
        threshold_factor=threshold_factor,
    )
    # Upstream's own formula (resurfemg.postprocessing.quality_assessment.
    # detect_local_high_aub) - recomputed here since it only returns the
    # boolean array, not the threshold it compared against.
    effective_threshold = float(
        threshold_factor
        * np.percentile(np.asarray(aubs, dtype=float), threshold_percentile)
    )

    flags = _per_breath_flags(
        "detect_local_high_aub",
        result,
        modality="emg",
        peak_indices=peak_indices,
        threshold=effective_threshold,
    )
    threshold_result = ParameterResult(
        name="detect_local_high_aub_threshold",
        value=effective_threshold,
        modality="emg",
        method="resurfemg.detect_local_high_aub",
        metadata={
            "threshold_percentile": threshold_percentile,
            "threshold_factor": threshold_factor,
        },
    )

    for flag in flags:
        session.quality.add(flag)
    session.parameter_results.add(threshold_result)

    _record_step(
        session,
        "emg.detect_local_high_aub",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.detect_local_high_aub",
            operation="emg.detect_local_high_aub",
            parameters={
                "threshold_percentile": threshold_percentile,
                "threshold_factor": threshold_factor,
            },
        ),
    )
    return {
        "detect_local_high_aub": result,
        "detect_local_high_aub_flags": flags,
        "detect_local_high_aub_threshold_result": threshold_result,
    }


@register_step(
    "emg.detect_extreme_time_products",
    reads={
        "session": "session",
        "time_product": "time_product",
        "peak_indices": "peak_indices",
    },
    writes=(
        "detect_extreme_time_products",
        "detect_extreme_time_products_flags",
        "detect_extreme_time_products_bounds_result",
    ),
    summary="Flag EMG breaths with extreme time-products.",
)
def detect_extreme_time_products(
    session: M3Session,
    time_product: Any,
    peak_indices: Any,
    *,
    upper_percentile: float = 95.0,
    upper_factor: float = 10.0,
    lower_percentile: float = 5.0,
    lower_factor: float = 0.1,
) -> dict[str, Any]:
    result = session.emg_adapter.detect_extreme_time_products(
        time_product,
        upper_percentile=upper_percentile,
        upper_factor=upper_factor,
        lower_percentile=lower_percentile,
        lower_factor=lower_factor,
    )
    values = np.asarray(time_product, dtype=float)
    upper_bound = float(upper_factor * np.percentile(values, upper_percentile))
    lower_bound = float(lower_factor * np.percentile(values, lower_percentile))

    flags = _per_breath_flags(
        "detect_extreme_time_products",
        result,
        modality="emg",
        peak_indices=peak_indices,
    )
    bounds_result = ParameterResult(
        name="detect_extreme_time_products_bounds",
        value=np.array([lower_bound, upper_bound]),
        modality="emg",
        method="resurfemg.detect_extreme_time_products",
        metadata={
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "upper_percentile": upper_percentile,
            "upper_factor": upper_factor,
            "lower_percentile": lower_percentile,
            "lower_factor": lower_factor,
        },
    )

    for flag in flags:
        session.quality.add(flag)
    session.parameter_results.add(bounds_result)

    _record_step(
        session,
        "emg.detect_extreme_time_products",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.detect_extreme_time_products",
            operation="emg.detect_extreme_time_products",
            parameters={
                "upper_percentile": upper_percentile,
                "upper_factor": upper_factor,
                "lower_percentile": lower_percentile,
                "lower_factor": lower_factor,
            },
        ),
    )
    return {
        "detect_extreme_time_products": result,
        "detect_extreme_time_products_flags": flags,
        "detect_extreme_time_products_bounds_result": bounds_result,
    }


@register_step(
    "emg.detect_non_consecutive_manoeuvres",
    reads={
        "session": "session",
        "ventilator_breath_indices": "ventilator_breath_indices",
        "pocc_indices": "pocc_indices",
    },
    writes=(
        "detect_non_consecutive_manoeuvres",
        "detect_non_consecutive_manoeuvres_flags",
    ),
    summary="Flag non-consecutive occlusion manoeuvres against ventilator breaths.",
)
def detect_non_consecutive_manoeuvres(
    session: M3Session, ventilator_breath_indices: Any, pocc_indices: Any
) -> dict[str, Any]:
    result = session.emg_adapter.detect_non_consecutive_manoeuvres(
        ventilator_breath_indices, pocc_indices
    )
    flags = _per_breath_flags(
        "detect_non_consecutive_manoeuvres",
        result,
        modality="pressure",
        peak_indices=pocc_indices,
    )
    for flag in flags:
        session.quality.add(flag)
    _record_step(
        session,
        "emg.detect_non_consecutive_manoeuvres",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.detect_non_consecutive_manoeuvres",
            operation="emg.detect_non_consecutive_manoeuvres",
            parameters={},
        ),
    )
    return {
        "detect_non_consecutive_manoeuvres": result,
        "detect_non_consecutive_manoeuvres_flags": flags,
    }


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
    writes=(
        "evaluate_bell_curve_error",
        "evaluate_bell_curve_error_results",
        "evaluate_bell_curve_error_flags",
    ),
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
    valid_peak, percentage_bell_error, bell_error, y_min, fitted_parameters = result

    results = _per_breath_results(
        "evaluate_bell_curve_error",
        percentage_bell_error,
        modality="emg",
        peak_indices=peak_indices,
        unit="%",
        method="resurfemg.evaluate_bell_curve_error",
        fs=fs,
        extra_metadata_per_item=[
            {"bell_error": float(bell_error[index]), "y_min": float(y_min[index])}
            for index in range(len(percentage_bell_error))
        ],
    )
    # Array-valued (one fitted bell-curve parameter vector per breath), so
    # this is its own ParameterResult rather than buried in metadata - it
    # then reuses the shared parameter_result_arrays.npz exporter (plan
    # Phase 6.3) instead of a competing EMG-specific array format.
    results.extend(
        _per_breath_results(
            "evaluate_bell_curve_error_fitted_parameters",
            list(np.asarray(fitted_parameters)),
            modality="emg",
            peak_indices=peak_indices,
            method="resurfemg.evaluate_bell_curve_error",
            fs=fs,
        )
    )
    flags = _per_breath_flags(
        "evaluate_bell_curve_error",
        valid_peak,
        modality="emg",
        peak_indices=peak_indices,
        fs=fs,
    )

    for parameter_result in results:
        session.parameter_results.add(parameter_result)
    for flag in flags:
        session.quality.add(flag)

    _record_step(
        session,
        "emg.evaluate_bell_curve_error",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.evaluate_bell_curve_error",
            operation="emg.evaluate_bell_curve_error",
            parameters={},
        ),
    )
    return {
        "evaluate_bell_curve_error": result,
        "evaluate_bell_curve_error_results": results,
        "evaluate_bell_curve_error_flags": flags,
    }


@register_step(
    "emg.evaluate_event_timing",
    reads={
        "session": "session",
        "peak_indices": "peak_indices",
        "processed_emg": "processed_emg",
        "ventilator_breath_indices": "ventilator_breath_indices",
        "ventilator_signals": "ventilator_signals",
    },
    writes=(
        "evaluate_event_timing",
        "evaluate_event_timing_results",
        "evaluate_event_timing_flags",
        "evaluate_event_timing_unmatched_count",
    ),
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
    # Keep the raw output's existing truncation behavior (Phase 5.1: "existing
    # pipeline consumers do not break"), but report the truncation instead of
    # silently dropping the unmatched events (Phase 5.4).
    paired_count = min(len(peak_indices), len(ventilator_breath_indices))
    unmatched_count = abs(len(peak_indices) - len(ventilator_breath_indices))
    paired_emg_peaks = peak_indices[:paired_count]
    paired_vent_peaks = ventilator_breath_indices[:paired_count]
    result = session.emg_adapter.evaluate_event_timing(
        paired_emg_peaks / fs,
        paired_vent_peaks / vent_fs,
    )
    correct_timing, delta_time = result

    results = _per_breath_results(
        "evaluate_event_timing_delta",
        delta_time,
        modality="emg",
        peak_indices=paired_emg_peaks,
        unit="s",
        method="resurfemg.evaluate_event_timing",
        fs=fs,
        extra_metadata_per_item=[
            {
                "emg_sample_index": int(paired_emg_peaks[index]),
                "ventilator_sample_index": int(paired_vent_peaks[index]),
                "emg_sample_frequency": fs,
                "ventilator_sample_frequency": vent_fs,
            }
            for index in range(paired_count)
        ],
    )
    flags = _per_breath_flags(
        "evaluate_event_timing",
        correct_timing,
        modality="emg",
        peak_indices=paired_emg_peaks,
        fs=fs,
    )
    if unmatched_count:
        flags.append(
            QualityFlag(
                name="evaluate_event_timing_unmatched",
                passed=False,
                severity="warning",
                modality="emg",
                message=(
                    f"{unmatched_count} event(s) had no paired counterpart and "
                    "were not assessed."
                ),
                metadata={
                    "unmatched_count": unmatched_count,
                    "emg_event_count": len(peak_indices),
                    "ventilator_event_count": len(ventilator_breath_indices),
                },
            )
        )

    for parameter_result in results:
        session.parameter_results.add(parameter_result)
    for flag in flags:
        session.quality.add(flag)

    _record_step(
        session,
        "emg.evaluate_event_timing",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.evaluate_event_timing",
            operation="emg.evaluate_event_timing",
            parameters={"unmatched_count": unmatched_count},
        ),
    )
    return {
        "evaluate_event_timing": result,
        "evaluate_event_timing_results": results,
        "evaluate_event_timing_flags": flags,
        "evaluate_event_timing_unmatched_count": unmatched_count,
    }


@register_step(
    "emg.evaluate_respiratory_rates",
    reads={
        "session": "session",
        "peak_indices": "peak_indices",
        "processed_emg": "processed_emg",
        "ventilator_respiratory_rate": "ventilator_respiratory_rate",
    },
    writes=(
        "evaluate_respiratory_rates",
        "evaluate_respiratory_rates_result",
        "evaluate_respiratory_rates_flag",
    ),
    summary="Score agreement between EMG-derived and ventilator-derived respiratory rate.",
)
def evaluate_respiratory_rates(
    session: M3Session,
    peak_indices: Any,
    processed_emg: Any,
    ventilator_respiratory_rate: Any,
    *,
    minimum_fraction: float = 0.1,
) -> dict[str, Any]:
    fs = float(processed_emg["fs"])
    envelope = processed_emg["envelope"]
    rr_vent = ventilator_respiratory_rate[0]
    result = session.emg_adapter.evaluate_respiratory_rates(
        peak_indices, len(envelope) / fs, rr_vent, minimum_fraction=minimum_fraction
    )
    detected_fraction, criterion_met = result

    parameter_result = ParameterResult(
        name="evaluate_respiratory_rates_detected_fraction",
        value=detected_fraction,
        modality="emg",
        method="resurfemg.evaluate_respiratory_rates",
        metadata={
            "minimum_fraction": minimum_fraction,
            "ventilator_rr": float(rr_vent),
        },
    )
    flag = QualityFlag(
        name="evaluate_respiratory_rates",
        passed=bool(criterion_met),
        severity="info",
        modality="emg",
        value=detected_fraction,
        threshold=minimum_fraction,
        metadata={"ventilator_rr": float(rr_vent)},
    )

    session.parameter_results.add(parameter_result)
    session.quality.add(flag)

    _record_step(
        session,
        "emg.evaluate_respiratory_rates",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.evaluate_respiratory_rates",
            operation="emg.evaluate_respiratory_rates",
            parameters={"minimum_fraction": minimum_fraction},
        ),
    )
    return {
        "evaluate_respiratory_rates": result,
        "evaluate_respiratory_rates_result": parameter_result,
        "evaluate_respiratory_rates_flag": flag,
    }


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
