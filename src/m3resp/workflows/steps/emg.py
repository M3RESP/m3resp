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
    peak_indices_from_events,
    ventilator_signals,
)
from m3resp.core.session import M3Session
from m3resp.core.events import BreathEvent, Event
from m3resp.synchronization.ventilator import (
    iter_ventilator_detections,
    normalize_ventilator_breath,
)
from m3resp.data import ParameterResult, QualityFlag, Signal
from m3resp.data.quality import Severity
from m3resp.processing.intervals import (
    onoff_from_baseline_crossings,
    onoff_from_slope,
)
from m3resp.processing.ecg import (
    estimated_ecg_subtraction as _estimated_ecg_subtraction,
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
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

#: Steps that call resurfemg directly or through ReSurfEMGAdapter declare this.
_RESURFEMG = ("resurfemg",)

_SESSION_ARTIFACT = StepArtifact(
    name="session",
    artifact_type="m3session",
    default_context_key="session",
    description="Backing M3Session the step reads from and/or records provenance onto.",
    public=False,
)


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
    description="Load an EMG recording file through ReSurfEMGAdapter and expose its raw channels as native Signals.",
    category="loading",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="file",
            value_type="path",
            required=True,
            path_kind="file",
            description="EMG recording file to load.",
        ),
        StepParameter(
            name="loader_options",
            value_type="mapping",
            required=False,
            default=None,
            description="Extra keyword arguments forwarded to M3Session.load_emg().",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="emg_recording",
            artifact_type="emg_recording",
            description="Raw upstream EMGRecording object.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="raw_emg_signals",
            artifact_type="signal_list",
            description="Native Signal per raw EMG channel.",
        ),
    ),
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
    description="Load a ventilator recording file through ReSurfEMGAdapter.",
    category="loading",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="file",
            value_type="path",
            required=True,
            path_kind="file",
            description="Ventilator recording file to load.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ventilator_raw",
            artifact_type="ventilator_recording",
            description="Raw upstream ventilator recording dict.",
            compatibility_only=True,
        ),
    ),
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
    description="Split a raw ventilator recording into named pressure/flow/volume channel arrays plus sample frequency.",
    category="preprocessing",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="ventilator_raw",
            artifact_type="ventilator_recording",
            description="Raw ventilator recording from 'emg.load_ventilator'.",
            compatibility_only=True,
        ),
    ),
    parameters=(
        StepParameter(
            name="pressure_channel",
            value_type="integer",
            default=0,
            minimum=0,
            description="Channel index of airway pressure.",
        ),
        StepParameter(
            name="flow_channel",
            value_type="integer",
            default=1,
            minimum=0,
            description="Channel index of flow.",
        ),
        StepParameter(
            name="volume_channel",
            value_type="integer",
            default=2,
            minimum=0,
            description="Channel index of volume.",
        ),
        StepParameter(
            name="fs",
            value_type="number",
            required=False,
            default=None,
            unit="Hz",
            description="Sample frequency override; defaults to the recording's own metadata.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Mapping of 'pressure'/'flow'/'volume' arrays plus 'fs'.",
        ),
    ),
)
def ventilator_channels(
    ventilator_raw: Any,
    *,
    pressure_channel: int = 0,
    flow_channel: int = 1,
    volume_channel: int = 2,
    fs: float | None = None,
) -> dict[str, Any]:
    signals = ventilator_signals(
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
    description="Filter the loaded EMG recording and compute its envelope via ReSurfEMGAdapter.preprocess. Accepts arbitrary adapter keyword arguments beyond 'variant'.",
    category="preprocessing",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="variant",
            value_type="string",
            required=False,
            default=None,
            description="Store this preprocessing under a named variant instead of the default slot.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Mapping of 'filtered'/'envelope'/'fs'/'channel'/'metadata'.",
        ),
    ),
)
def preprocess(session: M3Session, **kwargs: Any) -> dict[str, Any]:
    return {"processed_emg": session.preprocess_emg(**kwargs)}


@register_step(
    "emg.detect_breaths",
    reads={"session": "session"},
    writes=("emg_breath_events",),
    summary="Detect EMG breaths from the envelope.",
    description="Detect EMG breaths from the processed envelope via ReSurfEMGAdapter.detect_breaths. Accepts arbitrary adapter keyword arguments beyond 'variant'.",
    category="detection",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="variant",
            value_type="string",
            required=False,
            default=None,
            description="Detect breaths on a named preprocessing variant instead of the default slot.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="emg_breath_events",
            artifact_type="breath_event_list",
            description="Detected EMG breath events.",
        ),
    ),
)
def detect_breaths(session: M3Session, **kwargs: Any) -> dict[str, Any]:
    events = session.detect_emg_breaths(**kwargs)
    return {"emg_breath_events": events}


@register_step(
    "emg.peak_indices",
    reads={"events": "emg_breath_events", "processed_emg": "processed_emg"},
    writes=("peak_indices",),
    summary="Derive EMG breath peak sample indices from detected breath events.",
    description="Convert detected EMG breath events into peak sample indices into the processed envelope.",
    category="detection",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="events",
            artifact_type="breath_event_list",
            default_context_key="emg_breath_events",
            description="Detected EMG breath events from 'emg.detect_breaths'.",
        ),
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying 'fs'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak sample indices.",
        ),
    ),
)
def peak_indices(events: Any, processed_emg: Any) -> dict[str, Any]:
    import numpy as np

    fs = float(processed_emg["fs"])
    return {"peak_indices": np.asarray(peak_indices_from_events(events, fs), dtype=int)}


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
    description="Compute a moving-percentile baseline of the EMG envelope via ReSurfEMGAdapter.moving_baseline.",
    category="baseline",
    modality="emg",
    optional_packages=_RESURFEMG,
    alternatives=("emg.slopesum_baseline",),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="window_seconds",
            value_type="number",
            default=30.0,
            unit="s",
            minimum=0,
            description="Moving-percentile window length.",
        ),
        StepParameter(
            name="step_seconds",
            value_type="number",
            default=1.0,
            unit="s",
            minimum=0,
            description="Step between successive baseline windows.",
        ),
        StepParameter(
            name="percentile",
            value_type="number",
            default=33.0,
            minimum=0,
            maximum=100,
            description="Percentile of the envelope computed within each window.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            unit=None,
            description="Moving-percentile baseline array, one value per envelope sample.",
        ),
        StepArtifact(
            name="baseline_signal",
            artifact_type="signal",
            description="Native Signal wrapping the baseline.",
        ),
    ),
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
    description="Compute a slope-sum baseline of the EMG envelope via ReSurfEMGAdapter.slopesum_baseline.",
    category="baseline",
    modality="emg",
    optional_packages=_RESURFEMG,
    alternatives=("emg.moving_baseline",),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="window_seconds",
            value_type="number",
            default=30.0,
            unit="s",
            minimum=0,
            description="Moving-percentile window length.",
        ),
        StepParameter(
            name="step_seconds",
            value_type="number",
            default=1.0,
            unit="s",
            minimum=0,
            description="Step between successive baseline windows.",
        ),
        StepParameter(
            name="percentile",
            value_type="number",
            default=33.0,
            minimum=0,
            maximum=100,
            description="Percentile used for the primary baseline.",
        ),
        StepParameter(
            name="augmented_percentile",
            value_type="number",
            default=25.0,
            minimum=0,
            maximum=100,
            description="Percentile used for the augmented (running mean/std) baseline.",
        ),
        StepParameter(
            name="moving_average_seconds",
            value_type="number",
            default=0.5,
            unit="s",
            minimum=0,
            description="Smoothing window for the running mean/std.",
        ),
        StepParameter(
            name="percentile_window_seconds",
            value_type="number",
            default=1.0,
            unit="s",
            minimum=0,
            description="Window used when recomputing the percentile series.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            description="Slope-sum baseline array, one value per envelope sample.",
        ),
        StepArtifact(
            name="slopesum_baseline_detail",
            artifact_type="diagnostic_summary",
            description="Running mean/std plus the intermediate slope-sum series.",
            public=False,
            compatibility_only=True,
        ),
        StepArtifact(
            name="baseline_signal",
            artifact_type="signal",
            description="Native Signal wrapping the baseline.",
        ),
        StepArtifact(
            name="slopesum_baseline_native_detail",
            artifact_type="mapping",
            description="Native running mean/std, without the raw upstream series.",
        ),
        StepArtifact(
            name="baseline_running_mean_signal",
            artifact_type="signal",
            description="Native Signal wrapping the running mean.",
        ),
        StepArtifact(
            name="baseline_running_std_signal",
            artifact_type="signal",
            description="Native Signal wrapping the running std.",
        ),
    ),
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
    description="Detect ECG (R-wave-like) peaks in a raw channel or a processed_emg key, via ReSurfEMGAdapter.detect_ecg_peaks. Prerequisite for emg.ecg_gating/emg.ecg_wavelet_denoising.",
    category="preprocessing",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle, used when 'ecg_channel' is unset.",
        ),
    ),
    parameters=(
        StepParameter(
            name="ecg_channel",
            value_type="integer",
            required=False,
            default=None,
            minimum=0,
            description="Raw channel-major index carrying ECG. Takes priority over 'source' when set.",
        ),
        StepParameter(
            name="source",
            value_type="string",
            default="raw_channel",
            description="Key into processed_emg to detect ECG on, when 'ecg_channel' is unset.",
        ),
        StepParameter(
            name="peak_fraction",
            value_type="number",
            default=0.4,
            minimum=0,
            maximum=1,
            description="Detection threshold as a fraction of signal amplitude.",
        ),
        StepParameter(
            name="peak_width_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Minimum peak width. Defaults to the detector's own choice when unset.",
        ),
        StepParameter(
            name="peak_distance_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Minimum distance between peaks. Defaults to the detector's own choice when unset.",
        ),
        StepParameter(
            name="bandpass_filter",
            value_type="boolean",
            default=True,
            description="Bandpass-filter the source before peak detection.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ecg_peak_indices",
            artifact_type="index_array",
            description="Detected ECG peak sample indices.",
        ),
        StepArtifact(
            name="ecg_peak_events",
            artifact_type="event_list",
            description="Native Event per detected ECG peak.",
        ),
        StepArtifact(
            name="ecg_peak_count_result",
            artifact_type="parameter_result",
            description="Native ParameterResult: count of detected ECG peaks.",
        ),
    ),
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
    "emg.ecg_estimated_subtraction",
    reads={"session": "session", "processed_emg": "processed_emg"},
    writes=(
        "ees_cleaned_emg",
        "processed_emg_after_ecg",
        "ees_cleaned_signal",
        "ees_estimated_ecg_signal",
        "ees_detection_signal",
        "ees_dynamic_threshold_signal",
        "ees_qrs_events",
        "ees_r_peak_indices",
        "ees_candidate_peaks_result",
        "ees_corrected_peaks_result",
        "ees_rejected_peaks_result",
        "ees_restored_peaks_result",
        "ees_qrs_indices_result",
        "ees_normalized_segments_result",
        "ees_template_result",
    ),
    summary="Estimate and subtract ECG artifacts using a QRS template.",
    description=(
        "Native Estimated ECG Subtraction (EES): detects QRS beats directly in "
        "'source', builds a template, and subtracts the estimated ECG artifact "
        "from the EMG channel. An alternative to emg.ecg_gating/"
        "emg.ecg_wavelet_denoising that does not consume emg.ecg_detect_peaks. "
        "The candidate/corrected/rejected/restored QRS-index and template "
        "outputs are compatibility-only diagnostics for reviewing detected "
        "beats, not part of the native public result."
    ),
    category="preprocessing",
    modality="emg",
    optional_packages=(),
    alternatives=("emg.ecg_gating", "emg.ecg_wavelet_denoising"),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying 'source' and 'fs'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="source",
            value_type="string",
            default="filtered",
            description="Key into processed_emg to clean.",
        ),
        StepParameter(
            name="detection_low_hz",
            value_type="number",
            default=4.0,
            unit="Hz",
            description="Lower edge of the QRS-detection bandpass filter.",
        ),
        StepParameter(
            name="detection_high_hz",
            value_type="number",
            default=50.0,
            unit="Hz",
            description="Upper edge of the QRS-detection bandpass filter.",
        ),
        StepParameter(
            name="filter_order",
            value_type="integer",
            default=4,
            minimum=1,
            description="Detection bandpass filter order.",
        ),
        StepParameter(
            name="detection_smoothing_seconds",
            value_type="number",
            default=0.0167,
            unit="s",
            description="Smoothing applied to the detection signal.",
        ),
        StepParameter(
            name="threshold_interval_seconds",
            value_type="number",
            default=0.5,
            unit="s",
            description="Interval over which the dynamic detection threshold is recomputed.",
        ),
        StepParameter(
            name="threshold_smoothing_seconds",
            value_type="number",
            default=0.0125,
            unit="s",
            description="Smoothing applied to the dynamic threshold.",
        ),
        StepParameter(
            name="qrs_window_seconds",
            value_type="number",
            default=0.3,
            unit="s",
            description="Window around each detected beat used to build the QRS template.",
        ),
        StepParameter(
            name="inter_qrs_tolerance",
            value_type="number",
            default=0.66,
            description="Fraction of the median inter-beat interval tolerated when validating beats.",
        ),
        StepParameter(
            name="minimum_template_beats",
            value_type="integer",
            default=3,
            minimum=1,
            description="Minimum number of beats required to build a stable template.",
        ),
        StepParameter(
            name="minimum_qrs_interval_seconds",
            value_type="number",
            required=False,
            default=0.25,
            unit="s",
            description="Shortest accepted inter-beat interval.",
        ),
        StepParameter(
            name="maximum_qrs_interval_seconds",
            value_type="number",
            required=False,
            default=2.0,
            unit="s",
            description="Longest accepted inter-beat interval.",
        ),
        StepParameter(
            name="envelope_window_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Envelope recomputation window on the cleaned signal. Defaults to the original preprocessing window.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ees_cleaned_emg",
            artifact_type="signal_array",
            description="Cleaned (ECG-removed) EMG array.",
        ),
        StepArtifact(
            name="processed_emg_after_ecg",
            artifact_type="emg_processed_bundle",
            description="Updated processed-EMG bundle with the cleaned signal as its 'filtered'/'envelope'.",
            public=False,
        ),
        StepArtifact(
            name="ees_cleaned_signal",
            artifact_type="signal",
            description="Native Signal wrapping the cleaned EMG.",
        ),
        StepArtifact(
            name="ees_estimated_ecg_signal",
            artifact_type="signal",
            description="Native Signal wrapping the estimated ECG artifact that was subtracted.",
        ),
        StepArtifact(
            name="ees_detection_signal",
            artifact_type="signal",
            description="Native Signal of the QRS-detection signal used to find beats.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_dynamic_threshold_signal",
            artifact_type="signal",
            description="Native Signal of the dynamic detection threshold over time.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_r_peak_indices",
            artifact_type="index_array",
            description="Detected R-peak sample indices.",
        ),
        StepArtifact(
            name="ees_qrs_events",
            artifact_type="event_list",
            description="Native Event per detected QRS beat.",
        ),
        StepArtifact(
            name="ees_candidate_peaks_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult: initially detected candidate peak indices.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_corrected_peaks_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult: candidate peaks after correction.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_rejected_peaks_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult: peaks rejected during correction.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_restored_peaks_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult: peaks restored by periodicity.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_qrs_indices_result",
            artifact_type="parameter_result",
            axes=("beat", "wave_q_r_s"),
            description="Native array-valued ParameterResult: Q/R/S sample index per detected beat.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_normalized_segments_result",
            artifact_type="parameter_result",
            axes=("beat", "template_sample"),
            description="Native array-valued ParameterResult: each beat's segment normalized onto the template timebase.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="ees_template_result",
            artifact_type="parameter_result",
            axes=("template_sample",),
            description="Native array-valued ParameterResult: the fitted QRS template.",
            compatibility_only=True,
        ),
    ),
)
def ecg_estimated_subtraction(
    session: M3Session,
    processed_emg: Any,
    *,
    source: str = "filtered",
    detection_low_hz: float = 4.0,
    detection_high_hz: float = 50.0,
    filter_order: int = 4,
    detection_smoothing_seconds: float = 0.0167,
    threshold_interval_seconds: float = 0.5,
    threshold_smoothing_seconds: float = 0.0125,
    qrs_window_seconds: float = 0.3,
    inter_qrs_tolerance: float = 0.66,
    minimum_template_beats: int = 3,
    minimum_qrs_interval_seconds: float | None = 0.25,
    maximum_qrs_interval_seconds: float | None = 2.0,
    envelope_window_seconds: float | None = None,
) -> dict[str, Any]:
    """Run the paper-based Estimated ECG Subtraction method.

    The step detects ECG directly in ``source``; it does not consume the
    output of ``emg.ecg_detect_peaks``. Diagnostic signals and arrays are
    retained so the detected beats and estimated artifact can be reviewed.
    """

    if source not in processed_emg:
        raise ValueError(
            f"emg.ecg_estimated_subtraction source {source!r} is not present in "
            f"processed_emg; available keys: {sorted(processed_emg.keys())}."
        )
    array = np.asarray(processed_emg[source], dtype=float)
    fs = float(processed_emg["fs"])
    result = _estimated_ecg_subtraction(
        array,
        sample_frequency=fs,
        detection_band_hz=(detection_low_hz, detection_high_hz),
        filter_order=filter_order,
        detection_smoothing_seconds=detection_smoothing_seconds,
        threshold_interval_seconds=threshold_interval_seconds,
        threshold_smoothing_seconds=threshold_smoothing_seconds,
        qrs_window_seconds=qrs_window_seconds,
        inter_qrs_tolerance=inter_qrs_tolerance,
        minimum_template_beats=minimum_template_beats,
        minimum_qrs_interval_seconds=minimum_qrs_interval_seconds,
        maximum_qrs_interval_seconds=maximum_qrs_interval_seconds,
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
        envelope = rolling_arv(
            result.cleaned,
            window_length=max(1, int(effective_envelope_window_seconds * fs)),
        )
    processed_emg_after_ecg = {
        **processed_emg,
        "filtered": result.cleaned,
        "envelope": envelope,
    }
    _update_session_after_ecg_removal(session, processed_emg_after_ecg)

    label, unit = _processed_channel_label_and_unit(processed_emg)
    time = np.arange(len(array), dtype=float) / fs
    method = "m3resp.estimated_ecg_subtraction"
    parameters = {
        "source": source,
        "detection_low_hz": detection_low_hz,
        "detection_high_hz": detection_high_hz,
        "filter_order": filter_order,
        "detection_smoothing_seconds": detection_smoothing_seconds,
        "threshold_interval_seconds": threshold_interval_seconds,
        "threshold_smoothing_seconds": threshold_smoothing_seconds,
        "qrs_window_seconds": qrs_window_seconds,
        "inter_qrs_tolerance": inter_qrs_tolerance,
        "minimum_template_beats": minimum_template_beats,
        "minimum_qrs_interval_seconds": minimum_qrs_interval_seconds,
        "maximum_qrs_interval_seconds": maximum_qrs_interval_seconds,
        "effective_envelope_window_seconds": effective_envelope_window_seconds,
    }
    signal_metadata = dict(parameters)
    ees_cleaned_signal = Signal(
        values=result.cleaned,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_ecg_ees_cleaned",
        modality="emg",
        channel=label,
        source="m3resp",
        processing_state="filtered",
        derived_from="processed",
        method=method,
        metadata=signal_metadata,
    )
    ees_estimated_ecg_signal = Signal(
        values=result.estimated_ecg,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_estimated_ecg",
        modality="emg",
        channel=label,
        source="m3resp",
        processing_state="derived",
        derived_from="processed",
        method=method,
        metadata=dict(parameters),
    )
    ees_detection_signal = Signal(
        values=result.detection_signal,
        time=time,
        sample_frequency=fs,
        unit=None,
        name=f"{label}_ees_detection",
        modality="emg",
        channel=label,
        source="m3resp",
        processing_state="derived",
        derived_from="processed",
        method=method,
        metadata=dict(parameters),
    )
    ees_dynamic_threshold_signal = Signal(
        values=result.dynamic_threshold,
        time=time,
        sample_frequency=fs,
        unit=None,
        name=f"{label}_ees_dynamic_threshold",
        modality="emg",
        channel=label,
        source="m3resp",
        processing_state="derived",
        derived_from="processed",
        method=method,
        metadata=dict(parameters),
    )
    for output_signal in (
        ees_cleaned_signal,
        ees_estimated_ecg_signal,
        ees_detection_signal,
        ees_dynamic_threshold_signal,
    ):
        session.signals.add(output_signal)

    restored = set(result.restored_peak_indices.tolist())
    ees_qrs_events = [
        Event(
            name="ees_qrs",
            modality="emg",
            time=float(r_index) / fs,
            sample_index=int(r_index),
            metadata={
                **parameters,
                "q_sample_index": int(q_index),
                "s_sample_index": int(s_index),
                "detection_peak_sample_index": int(detection_peak),
                "restored_by_periodicity": int(detection_peak) in restored,
            },
        )
        for detection_peak, (q_index, r_index, s_index) in zip(
            result.template_peak_indices, result.qrs_indices
        )
    ]
    session.add_events("ees_qrs", ees_qrs_events)

    result_specs = (
        ("ees_candidate_peaks", result.candidate_peak_indices, ["candidate"]),
        ("ees_corrected_peaks", result.corrected_peak_indices, ["corrected"]),
        ("ees_rejected_peaks", result.rejected_peak_indices, ["rejected"]),
        ("ees_restored_peaks", result.restored_peak_indices, ["restored"]),
        ("ees_qrs_indices", result.qrs_indices, ["beat", "wave_q_r_s"]),
        (
            "ees_normalized_segments",
            result.normalized_segments,
            ["beat", "template_sample"],
        ),
        ("ees_normalized_template", result.normalized_template, ["template_sample"]),
    )
    parameter_results: dict[str, ParameterResult] = {}
    for name, values, axes in result_specs:
        parameter_result = ParameterResult(
            name=name,
            value=values,
            modality="emg",
            channel=label,
            method=method,
            metadata={
                **parameters,
                "axes": axes,
                "template_sample_offsets": (
                    result.template_sample_offsets.tolist()
                    if "template" in name
                    else None
                ),
            },
        )
        session.parameter_results.add(parameter_result)
        parameter_results[name] = parameter_result

    _record_step(
        session,
        "emg.ecg_estimated_subtraction",
        metadata=_upstream_metadata(
            source_function="m3resp.processing.ecg.estimated_ecg_subtraction",
            operation="emg.ecg_estimated_subtraction",
            parameters=parameters,
            source_package="m3resp",
            implementation="m3resp.processing.ecg",
        ),
    )
    return {
        "ees_cleaned_emg": result.cleaned,
        "processed_emg_after_ecg": processed_emg_after_ecg,
        "ees_cleaned_signal": ees_cleaned_signal,
        "ees_estimated_ecg_signal": ees_estimated_ecg_signal,
        "ees_detection_signal": ees_detection_signal,
        "ees_dynamic_threshold_signal": ees_dynamic_threshold_signal,
        "ees_qrs_events": ees_qrs_events,
        "ees_r_peak_indices": result.qrs_indices[:, 1],
        "ees_candidate_peaks_result": parameter_results["ees_candidate_peaks"],
        "ees_corrected_peaks_result": parameter_results["ees_corrected_peaks"],
        "ees_rejected_peaks_result": parameter_results["ees_rejected_peaks"],
        "ees_restored_peaks_result": parameter_results["ees_restored_peaks"],
        "ees_qrs_indices_result": parameter_results["ees_qrs_indices"],
        "ees_normalized_segments_result": parameter_results["ees_normalized_segments"],
        "ees_template_result": parameter_results["ees_normalized_template"],
    }


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
    description="Remove ECG peaks from an EMG channel by gating each detected peak (zero/interpolate/replace), via ReSurfEMGAdapter.gate_ecg.",
    category="preprocessing",
    modality="emg",
    optional_packages=_RESURFEMG,
    alternatives=("emg.ecg_wavelet_denoising", "emg.ecg_estimated_subtraction"),
    mutually_exclusive_parameters=(("gate_width_seconds", "gate_width_samples"),),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying 'source' and 'fs'.",
        ),
        StepArtifact(
            name="ecg_peak_indices",
            artifact_type="index_array",
            description="ECG peak indices from 'emg.ecg_detect_peaks'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="source",
            value_type="string",
            default="filtered",
            description="Key into processed_emg to gate.",
        ),
        StepParameter(
            name="gate_width_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Gate width in seconds. Mutually exclusive with 'gate_width_samples'.",
        ),
        StepParameter(
            name="gate_width_samples",
            value_type="integer",
            required=False,
            default=None,
            minimum=1,
            description="Gate width in samples. Mutually exclusive with 'gate_width_seconds'. Defaults to 205 samples (resurfemg's own default) when both are unset.",
        ),
        StepParameter(
            name="fill_method",
            value_type="integer",
            default=1,
            choices=(0, 1, 2, 3),
            description="Gate fill strategy: 0 zeros, 1 interpolation, 2 mean of a neighboring segment, 3 running-RMS-based replacement.",
        ),
        StepParameter(
            name="envelope_window_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Envelope recomputation window on the gated signal. Defaults to the original preprocessing window.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ecg_gated_emg",
            artifact_type="signal_array",
            description="Gated EMG array.",
        ),
        StepArtifact(
            name="processed_emg_after_ecg",
            artifact_type="emg_processed_bundle",
            description="Updated processed-EMG bundle with the gated signal as its 'filtered'/'envelope'.",
            public=False,
        ),
        StepArtifact(
            name="ecg_gated_signal",
            artifact_type="signal",
            description="Native Signal wrapping the gated EMG.",
        ),
        StepArtifact(
            name="ecg_gate_mask_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult: boolean mask of gated samples.",
        ),
    ),
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
    description="Remove ECG peaks from an EMG channel via a-trous wavelet shrinkage around each detected peak, via ReSurfEMGAdapter.wavelet_denoise_ecg.",
    category="preprocessing",
    modality="emg",
    optional_packages=_RESURFEMG,
    alternatives=("emg.ecg_gating", "emg.ecg_estimated_subtraction"),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying 'source' and 'fs'.",
        ),
        StepArtifact(
            name="ecg_peak_indices",
            artifact_type="index_array",
            description="ECG peak indices from 'emg.ecg_detect_peaks'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="source",
            value_type="string",
            default="filtered",
            description="Key into processed_emg to denoise.",
        ),
        StepParameter(
            name="hard_thresholding",
            value_type="boolean",
            default=True,
            description="Use hard (vs. soft) wavelet-coefficient thresholding.",
        ),
        StepParameter(
            name="levels",
            value_type="integer",
            default=4,
            minimum=1,
            description="Number of a-trous wavelet decomposition levels.",
        ),
        StepParameter(
            name="wavelet_type",
            value_type="string",
            default="db2",
            description="Wavelet family (PyWavelets name, e.g. 'db2').",
        ),
        StepParameter(
            name="fixed_threshold",
            value_type="number",
            default=4.5,
            minimum=0,
            description="Fixed wavelet-coefficient shrinkage threshold.",
        ),
        StepParameter(
            name="envelope_window_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Envelope recomputation window on the cleaned signal. Defaults to the original preprocessing window.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ecg_wavelet_cleaned_emg",
            artifact_type="signal_array",
            description="Cleaned EMG array.",
        ),
        StepArtifact(
            name="processed_emg_after_ecg",
            artifact_type="emg_processed_bundle",
            description="Updated processed-EMG bundle with the cleaned signal as its 'filtered'/'envelope'.",
            public=False,
        ),
        StepArtifact(
            name="ecg_wavelet_cleaned_signal",
            artifact_type="signal",
            description="Native Signal wrapping the cleaned EMG.",
        ),
        StepArtifact(
            name="wavelet_decomposition_result",
            artifact_type="parameter_result",
            axes=("level", "sample"),
            description="Native array-valued ParameterResult: wavelet decomposition coefficients.",
        ),
        StepArtifact(
            name="wavelet_thresholds_result",
            artifact_type="parameter_result",
            axes=("level", "sample"),
            description="Native array-valued ParameterResult: per-level thresholds applied.",
        ),
        StepArtifact(
            name="wavelet_gate_mask_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult: boolean mask of denoised samples.",
        ),
    ),
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
    description="Detect ventilator breath peaks from the volume channel.",
    category="detection",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle from 'emg.ventilator_channels'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="breath_width_seconds",
            value_type="number",
            default=0.5,
            unit="s",
            minimum=0,
            description="Minimum breath width.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ventilator_breath_indices",
            artifact_type="index_array",
            description="Detected ventilator breath peak sample indices.",
        ),
    ),
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
    description="Detect occlusion (Pocc) manoeuvre peaks from the pressure channel.",
    category="detection",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle from 'emg.ventilator_channels'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="peep",
            value_type="number",
            required=False,
            default=None,
            unit="cmH2O",
            description="PEEP baseline. Defaults to the median pressure when unset.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pocc_indices",
            artifact_type="index_array",
            description="Detected Pocc manoeuvre peak sample indices.",
        ),
    ),
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
    description="Find Pocc manoeuvre start/end indices around each detected peak via baseline crossing, and record BreathEvents.",
    category="detection",
    modality="emg",
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle from 'emg.ventilator_channels'.",
        ),
        StepArtifact(
            name="pocc_indices",
            artifact_type="index_array",
            description="Pocc peak indices from 'emg.find_occluded_breaths'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="peep",
            value_type="number",
            required=False,
            default=None,
            unit="cmH2O",
            description="PEEP baseline. Should match the value used in 'emg.find_occluded_breaths'; defaults to the median pressure when unset.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pocc_start_indices",
            artifact_type="index_array",
            description="Pocc manoeuvre start sample indices.",
        ),
        StepArtifact(
            name="pocc_end_indices",
            artifact_type="index_array",
            description="Pocc manoeuvre end sample indices.",
        ),
        StepArtifact(
            name="pocc_interval_validity",
            artifact_type="boolean_array",
            description="Whether each manoeuvre's start/end crossing was found.",
        ),
        StepArtifact(
            name="pocc_events",
            artifact_type="breath_event_list",
            description="Native BreathEvent per Pocc manoeuvre.",
        ),
    ),
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
    description="Integrate pressure above the PEEP baseline over each Pocc manoeuvre's start/end window.",
    category="parameters",
    modality="emg",
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle from 'emg.ventilator_channels'.",
        ),
        StepArtifact(
            name="pocc_start_indices",
            artifact_type="index_array",
            description="Pocc manoeuvre start indices from 'emg.pocc_intervals'.",
        ),
        StepArtifact(
            name="pocc_end_indices",
            artifact_type="index_array",
            description="Pocc manoeuvre end indices from 'emg.pocc_intervals'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="peep",
            value_type="number",
            required=False,
            default=None,
            unit="cmH2O",
            description="PEEP baseline. Should match the value used in 'emg.pocc_intervals'; defaults to the median pressure when unset.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pocc_time_products",
            artifact_type="array",
            unit="cmH2O*s",
            description="Pressure-time product per Pocc manoeuvre.",
        ),
        StepArtifact(
            name="pocc_time_product_result",
            artifact_type="parameter_result",
            unit="cmH2O*s",
            description="Native array-valued ParameterResult of the same values.",
        ),
    ),
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
    description="Evaluate Pocc manoeuvre validity from the pressure upslope shape against three configurable thresholds (Warnaar et al. 2024), producing one QualityFlag and three criterion measurements per manoeuvre.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle supplying pressure.",
        ),
        StepArtifact(
            name="pocc_indices",
            artifact_type="index_array",
            description="Pocc peak indices from 'emg.find_occluded_breaths'.",
        ),
        StepArtifact(
            name="pocc_end_indices",
            artifact_type="index_array",
            description="Pocc end indices from 'emg.pocc_intervals'.",
        ),
        StepArtifact(
            name="pocc_time_products",
            artifact_type="array",
            description="Pocc time products from 'emg.pocc_time_product'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="dp_up_10_threshold",
            value_type="number",
            default=0.0,
            description="Minimum acceptable dP at 10% of the upslope.",
        ),
        StepParameter(
            name="dp_up_90_threshold",
            value_type="number",
            default=2.0,
            description="Minimum acceptable dP at 90% of the upslope.",
        ),
        StepParameter(
            name="dp_up_90_norm_threshold",
            value_type="number",
            default=0.8,
            description="Minimum acceptable normalized dP at 90% of the upslope.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pocc_quality",
            artifact_type="boolean_array",
            description="Overall pass/fail per manoeuvre.",
        ),
        StepArtifact(
            name="pocc_quality_criteria",
            artifact_type="array",
            axes=("criterion", "manoeuvre"),
            description="Raw upstream 3-by-N criteria matrix.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="pocc_quality_results",
            artifact_type="parameter_result_list",
            description="Native ParameterResult per manoeuvre per criterion (dp_up_10/dp_up_90/dp_up_90_norm).",
        ),
        StepArtifact(
            name="pocc_quality_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per manoeuvre.",
        ),
    ),
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
    description="Check that the median EMG-to-ECG interpeak distance ratio stays under a threshold, as a proxy for adequate ECG removal (Warnaar et al. 2024).",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="ecg_peak_indices",
            artifact_type="index_array",
            description="ECG peak indices from 'emg.ecg_detect_peaks'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="EMG breath peak indices.",
        ),
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying 'fs'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="threshold",
            value_type="number",
            default=1.1,
            minimum=0,
            description="Maximum acceptable EMG/ECG median interpeak distance ratio.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="interpeak_dist",
            artifact_type="boolean_metric",
            description="Whether the ratio passes the threshold.",
        ),
        StepArtifact(
            name="interpeak_dist_result",
            artifact_type="parameter_result_list",
            unit="s",
            description="Native ParameterResults: ECG median distance, EMG median distance, and their ratio.",
        ),
        StepArtifact(
            name="interpeak_dist_flag",
            artifact_type="quality_flag",
            description="Native QualityFlag for the overall check.",
        ),
    ),
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
    description="Find each breath's onset/offset sample indices where the envelope crosses the baseline.",
    category="detection",
    modality="emg",
    alternatives=("emg.onoffpeak_slope_extrapolation",),
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope.",
        ),
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            description="Baseline from 'emg.moving_baseline' or 'emg.slopesum_baseline'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices from 'emg.peak_indices'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="start_indices",
            artifact_type="index_array",
            description="Breath onset sample indices.",
        ),
        StepArtifact(
            name="end_indices",
            artifact_type="index_array",
            description="Breath offset sample indices.",
        ),
    ),
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
    description="Find each breath's onset/offset sample indices by extrapolating the steepest envelope slope near each peak.",
    category="detection",
    modality="emg",
    alternatives=("emg.onoffpeak_baseline_crossing",),
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices from 'emg.peak_indices'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="slope_window_seconds",
            value_type="number",
            default=0.5,
            unit="s",
            minimum=0,
            description="Window around each peak used to estimate the slope.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="onoffpeak_slope_result",
            artifact_type="onoff_result",
            description="Start/end indices plus slope-extrapolation diagnostics.",
        ),
    ),
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
    description="Compute the time from breath onset to peak for each detected breath.",
    category="features",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
        StepArtifact(
            name="start_indices",
            artifact_type="index_array",
            description="Breath onset indices.",
        ),
        StepArtifact(
            name="end_indices",
            artifact_type="index_array",
            description="Breath offset indices.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="time_to_peak",
            artifact_type="array",
            unit="s",
            description="Time-to-peak per breath.",
        ),
    ),
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
    description="Compute the pseudo-slope (rise rate) of the envelope for each detected breath.",
    category="features",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope.",
        ),
        StepArtifact(
            name="start_indices",
            artifact_type="index_array",
            description="Breath onset indices.",
        ),
        StepArtifact(
            name="end_indices",
            artifact_type="index_array",
            description="Breath offset indices.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pseudo_slope",
            artifact_type="array",
            description="Pseudo-slope per breath.",
        ),
    ),
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
    description="Compute the envelope amplitude above baseline at each breath peak.",
    category="features",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices.",
        ),
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            description="Baseline from 'emg.moving_baseline' or 'emg.slopesum_baseline'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="amplitude",
            artifact_type="array",
            description="Amplitude above baseline per breath.",
        ),
    ),
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
    description="Integrate the envelope above baseline over each breath's onset/offset window.",
    category="features",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
        StepArtifact(
            name="start_indices",
            artifact_type="index_array",
            description="Breath onset indices.",
        ),
        StepArtifact(
            name="end_indices",
            artifact_type="index_array",
            description="Breath offset indices.",
        ),
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            description="Baseline from 'emg.moving_baseline' or 'emg.slopesum_baseline'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="time_product",
            artifact_type="array",
            description="Time-product per breath.",
        ),
    ),
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
    description="Compute the area of the envelope that dips under baseline within a window around each breath peak.",
    category="features",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices.",
        ),
        StepArtifact(
            name="start_indices",
            artifact_type="index_array",
            description="Breath onset indices.",
        ),
        StepArtifact(
            name="end_indices",
            artifact_type="index_array",
            description="Breath offset indices.",
        ),
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            description="Baseline from 'emg.moving_baseline' or 'emg.slopesum_baseline'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="window_seconds",
            value_type="number",
            default=5.0,
            unit="s",
            minimum=0,
            description="Window around each peak searched for under-baseline area.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="area_under_baseline",
            artifact_type="array",
            description="Area-under-baseline result per breath, and supporting arrays.",
        ),
    ),
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
    description="Compute respiratory rate from the detected EMG breath peak indices.",
    category="parameters",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices.",
        ),
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying 'fs'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="respiratory_rate",
            artifact_type="scalar_metric",
            unit="breaths/min",
            description="EMG-derived respiratory rate.",
        ),
    ),
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
    description="Compute respiratory rate from the detected ventilator breath peak indices.",
    category="parameters",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="ventilator_breath_indices",
            artifact_type="index_array",
            description="Ventilator breath peak indices from 'emg.detect_ventilator_breath'.",
        ),
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle supplying 'fs'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ventilator_respiratory_rate",
            artifact_type="scalar_metric",
            unit="breaths/min",
            description="Ventilator-derived respiratory rate.",
        ),
    ),
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
    description="Compute a pseudo signal-to-noise ratio per breath. A measurement only becomes a pass/fail criterion when 'minimum_snr' is set; otherwise only the measurement is produced.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices.",
        ),
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            description="Baseline from 'emg.moving_baseline' or 'emg.slopesum_baseline'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="minimum_snr",
            value_type="number",
            required=False,
            default=None,
            description="Minimum acceptable SNR. When unset, only the measurement is produced, with no pass/fail flags.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="snr_pseudo",
            artifact_type="array",
            description="Pseudo-SNR per breath.",
        ),
        StepArtifact(
            name="snr_pseudo_results",
            artifact_type="parameter_result_list",
            description="Native ParameterResult per breath.",
        ),
        StepArtifact(
            name="snr_pseudo_flags",
            artifact_type="quality_flag_list",
            required=False,
            description="Native QualityFlag per breath, only when 'minimum_snr' is set.",
        ),
    ),
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
    description="Compute the percentage of each breath's window spent with the envelope under baseline, flagging breaths above 'aub_threshold'.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices.",
        ),
        StepArtifact(
            name="start_indices",
            artifact_type="index_array",
            description="Breath onset indices.",
        ),
        StepArtifact(
            name="end_indices",
            artifact_type="index_array",
            description="Breath offset indices.",
        ),
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            description="Baseline from 'emg.moving_baseline' or 'emg.slopesum_baseline'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="aub_window_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Window searched for under-baseline area. Defaults to the breath's own onset/offset window when unset.",
        ),
        StepParameter(
            name="aub_threshold",
            value_type="number",
            default=40.0,
            minimum=0,
            maximum=100,
            unit="%",
            description="Maximum acceptable percentage under baseline.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="percentage_under_baseline",
            artifact_type="array",
            description="Raw upstream (valid, percentages, reference_values) tuple.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="percentage_under_baseline_results",
            artifact_type="parameter_result_list",
            unit="%",
            description="Native ParameterResult per breath.",
        ),
        StepArtifact(
            name="percentage_under_baseline_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per breath.",
        ),
    ),
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
    description="Flag breaths whose area-under-baseline exceeds 'threshold_factor' times a local percentile of the recording's own area-under-baseline values.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="area_under_baseline",
            artifact_type="array",
            description="Area-under-baseline values from 'emg.area_under_baseline'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices.",
        ),
    ),
    parameters=(
        StepParameter(
            name="threshold_percentile",
            value_type="number",
            default=75.0,
            minimum=0,
            maximum=100,
            description="Percentile of area-under-baseline used as the local reference.",
        ),
        StepParameter(
            name="threshold_factor",
            value_type="number",
            default=4.0,
            minimum=0,
            description="Multiplier applied to the reference percentile to get the flagging threshold.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="detect_local_high_aub",
            artifact_type="boolean_array",
            description="Whether each breath is flagged.",
        ),
        StepArtifact(
            name="detect_local_high_aub_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per breath.",
        ),
        StepArtifact(
            name="detect_local_high_aub_threshold_result",
            artifact_type="parameter_result",
            description="Native ParameterResult: the effective threshold used.",
        ),
    ),
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
    description="Flag breaths whose time-product falls outside percentile-derived upper/lower bounds of the recording's own time-products.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="time_product",
            artifact_type="array",
            description="Time-products from 'emg.time_product'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices.",
        ),
    ),
    parameters=(
        StepParameter(
            name="upper_percentile",
            value_type="number",
            default=95.0,
            minimum=0,
            maximum=100,
            description="Percentile of time-products used as the upper reference.",
        ),
        StepParameter(
            name="upper_factor",
            value_type="number",
            default=10.0,
            minimum=0,
            description="Multiplier applied to the upper reference percentile.",
        ),
        StepParameter(
            name="lower_percentile",
            value_type="number",
            default=5.0,
            minimum=0,
            maximum=100,
            description="Percentile of time-products used as the lower reference.",
        ),
        StepParameter(
            name="lower_factor",
            value_type="number",
            default=0.1,
            minimum=0,
            description="Multiplier applied to the lower reference percentile.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="detect_extreme_time_products",
            artifact_type="boolean_array",
            description="Whether each breath is flagged.",
        ),
        StepArtifact(
            name="detect_extreme_time_products_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per breath.",
        ),
        StepArtifact(
            name="detect_extreme_time_products_bounds_result",
            artifact_type="parameter_result",
            description="Native ParameterResult: [lower_bound, upper_bound].",
        ),
    ),
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
    description="Flag Pocc manoeuvres that are not consecutive ventilator breaths, since a valid occlusion trial requires uninterrupted breaths.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="ventilator_breath_indices",
            artifact_type="index_array",
            description="Ventilator breath peak indices.",
        ),
        StepArtifact(
            name="pocc_indices",
            artifact_type="index_array",
            description="Pocc manoeuvre peak indices.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="detect_non_consecutive_manoeuvres",
            artifact_type="boolean_array",
            description="Whether each manoeuvre is flagged as non-consecutive.",
        ),
        StepArtifact(
            name="detect_non_consecutive_manoeuvres_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per manoeuvre.",
        ),
    ),
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
    description="Fit a bell curve to each breath's envelope window and score the fit error as a percentage of the time-product.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices.",
        ),
        StepArtifact(
            name="start_indices",
            artifact_type="index_array",
            description="Breath onset indices.",
        ),
        StepArtifact(
            name="end_indices",
            artifact_type="index_array",
            description="Breath offset indices.",
        ),
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
        StepArtifact(
            name="time_product",
            artifact_type="array",
            description="Time-products from 'emg.time_product', used to normalize the fit error.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="evaluate_bell_curve_error",
            artifact_type="array",
            description="Raw upstream (valid, percentage_error, error, y_min, fitted_parameters) tuple.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="evaluate_bell_curve_error_results",
            artifact_type="parameter_result_list",
            unit="%",
            description="Native ParameterResult per breath (percentage error) plus one array-valued ParameterResult per breath for the fitted bell-curve parameters.",
        ),
        StepArtifact(
            name="evaluate_bell_curve_error_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per breath.",
        ),
    ),
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
    description="Pair EMG and ventilator breaths index-by-index and score their timing agreement. Any unpaired breaths at the end (from unequal counts) are reported as a separate warning flag, not silently dropped.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="EMG breath peak indices.",
        ),
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying 'fs'.",
        ),
        StepArtifact(
            name="ventilator_breath_indices",
            artifact_type="index_array",
            description="Ventilator breath peak indices.",
        ),
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle supplying 'fs'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="evaluate_event_timing",
            artifact_type="array",
            description="Raw upstream (correct_timing, delta_time) tuple.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="evaluate_event_timing_results",
            artifact_type="parameter_result_list",
            unit="s",
            description="Native ParameterResult (timing delta) per paired breath.",
        ),
        StepArtifact(
            name="evaluate_event_timing_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per paired breath, plus one warning flag when breaths were unpaired.",
        ),
        StepArtifact(
            name="evaluate_event_timing_unmatched_count",
            artifact_type="count",
            description="Number of unpaired breaths, from unequal EMG/ventilator breath counts.",
        ),
    ),
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
    description="Check that the fraction of EMG breaths detected relative to the ventilator-derived respiratory rate exceeds 'minimum_fraction'.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="EMG breath peak indices.",
        ),
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying 'fs' and the envelope's duration.",
        ),
        StepArtifact(
            name="ventilator_respiratory_rate",
            artifact_type="scalar_metric",
            unit="breaths/min",
            description="Ventilator-derived respiratory rate from 'emg.ventilator_respiratory_rate'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="minimum_fraction",
            value_type="number",
            default=0.1,
            minimum=0,
            maximum=1,
            description="Minimum acceptable fraction of expected EMG breaths detected.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="evaluate_respiratory_rates",
            artifact_type="array",
            description="Raw upstream (detected_fraction, criterion_met) tuple.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="evaluate_respiratory_rates_result",
            artifact_type="parameter_result",
            description="Native ParameterResult: detected fraction.",
        ),
        StepArtifact(
            name="evaluate_respiratory_rates_flag",
            artifact_type="quality_flag",
            description="Native QualityFlag for the overall check.",
        ),
    ),
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
    description="Convert detected ventilator breath peak indices into native BreathEvents and store them on the session as 'ventilator_breaths'.",
    category="detection",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="ventilator_breath_indices",
            artifact_type="index_array",
            description="Ventilator breath peak indices from 'emg.detect_ventilator_breath'.",
        ),
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle supplying 'fs'.",
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="breath_width_seconds",
            value_type="number",
            default=0.5,
            unit="s",
            minimum=0,
            description="Assumed breath width used to derive start/end times around each peak.",
        ),
    ),
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
        normalize_ventilator_breath(
            detection, fs=fs, width_seconds=breath_width_seconds
        )
        for detection in iter_ventilator_detections(ventilator_breath_indices)
    ]
    session.add_events("ventilator_breaths", events)
    return {}
