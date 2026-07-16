"""Registered EIT pipeline steps.

Each step wraps a single ``eitprocessing`` operation. Upstream imports are
deferred to call time so the package installs without the optional
``eitprocessing`` dependency.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np

from m3resp.adapters.eitprocessing_adapter import (
    add_to_collection,
    continuous_data_to_signal,
)
from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, Signal
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step
from m3resp.workflows.utils import slice_signal_by_mode

#: Every eit.* step ultimately calls eitprocessing, directly or through
#: EITProcessingAdapter, so all of them declare the same optional dependency.
_EITPROCESSING = ("eitprocessing",)

#: Input/output for the session artifact shared by nearly every step.
_SESSION_ARTIFACT = StepArtifact(
    name="session",
    artifact_type="m3session",
    default_context_key="session",
    description="Backing M3Session the step reads from and/or records provenance onto.",
    public=False,
)


def _upstream_metadata(
    *, source_function: str, operation: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    """Build the Stage 2 EIT provenance metadata schema shared by native
    `Signal`/`ParameterResult` outputs (see `plan/stage2/
    1_eit_gap_migration_implementation_plan.md`, "Use one provenance
    schema")."""

    return {
        "source_package": "eitprocessing",
        "source_function": source_function,
        "implementation": "upstream_adapter",
        "parameters": parameters,
        "operation": operation,
    }


def _eitprocessing_version() -> str | None:
    """Installed `eitprocessing` version, read from package metadata without
    importing the package itself (so this stays optional-dependency-safe)."""

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("eitprocessing")
    except PackageNotFoundError:
        return None


def _record_step(
    session: M3Session, step_name: str, *, metadata: dict[str, Any]
) -> None:
    """Record per-step EIT provenance through the existing
    `M3Session._record()` seam (see `plan/stage2/
    1_eit_gap_migration_implementation_plan.md` Phase 5.2), reusing the
    step's declared reads/writes from the registry rather than a second
    EIT-only history mechanism."""

    from m3resp.workflows.registry import get_step

    definition = get_step(step_name)
    session._record(
        step_name,
        "eit",
        parameters={
            "step": step_name,
            "reads": sorted(definition.reads),
            "writes": list(definition.writes),
            "upstream_version": _eitprocessing_version(),
            **metadata,
        },
    )


def _object_array_to_float(values: Any) -> np.ndarray:
    """Convert an object-dtype array using `None` for missing entries into a
    float array with NaN in place of `None`, preserving its shape."""

    array = np.asarray(values, dtype=object)
    flat = [np.nan if value is None else float(value) for value in array.ravel()]
    return np.array(flat, dtype=float).reshape(array.shape)


def _sparse_data_to_array_parameter(
    obj: Any, *, modality: str, method: str, metadata: dict[str, Any]
) -> ParameterResult:
    """Convert an `eitprocessing.SparseData`-shaped object (one value per
    breath) into a single array-valued `ParameterResult`, preserving upstream
    time coordinates and unit rather than exploding it into one result per
    breath."""

    values = np.asarray(obj.values, dtype=float)
    metadata = dict(metadata)
    metadata["time"] = np.asarray(obj.time, dtype=float).tolist()
    metadata["shape"] = list(values.shape)
    metadata["dtype"] = str(values.dtype)
    name = getattr(obj, "label", None) or getattr(obj, "name", None) or "parameter"
    return ParameterResult(
        name=name,
        value=values,
        modality=modality,
        unit=getattr(obj, "unit", None),
        method=method,
        metadata=metadata,
    )


def _pixel_breaths_to_landmark_array(values: Any) -> np.ndarray:
    """Convert `PixelBreath`'s `(breath, row, column)` object array of
    `Breath | None` into a numeric `(breath, row, column, landmark)` array,
    where landmark is `[start_time, middle_time, end_time]`. Missing pixel
    breaths (including the always-unavailable first/last global breath)
    become NaN."""

    array = np.asarray(values, dtype=object)
    n_breaths, n_rows, n_cols = array.shape
    landmarks = np.full((n_breaths, n_rows, n_cols, 3), np.nan, dtype=float)
    for breath_index in range(n_breaths):
        for row in range(n_rows):
            for col in range(n_cols):
                breath = array[breath_index, row, col]
                if breath is not None:
                    landmarks[breath_index, row, col] = (
                        breath.start_time,
                        breath.middle_time,
                        breath.end_time,
                    )
    return landmarks


def _validate_unit_threshold(value: float, *, step: str, param: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{step} '{param}' must be between 0 and 1, got {value!r}.")


def _pixel_mask_to_parameter_result(
    mask: Any, *, name: str, method: str, metadata: dict[str, Any]
) -> ParameterResult:
    """Convert an `eitprocessing.roi.PixelMask` into an array-valued
    `ParameterResult`. Excluded pixels are already NaN in `mask.mask`, so
    this preserves that representation rather than cropping/flattening it."""

    value = np.asarray(mask.mask, dtype=float)
    included = ~np.isnan(value)
    metadata = dict(metadata)
    metadata.update(
        {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "axes": ["row", "column"],
            "included_pixel_count": int(included.sum()),
            "included_pixel_fraction": float(included.mean()) if value.size else 0.0,
        }
    )
    return ParameterResult(
        name=name,
        value=value,
        modality="eit",
        unit=None,
        method=method,
        metadata=metadata,
    )


@register_step(
    "eit.load",
    reads={"session": "session"},
    writes=(
        "raw_eit",
        "raw_global_impedance",
        "eit_sequence",
        "raw_global_impedance_signal",
    ),
    summary="Load an EIT recording into the session.",
    description="Load a vendor EIT recording file through EITProcessingAdapter and expose the raw pixel/global-impedance data.",
    category="loading",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="file",
            value_type="path",
            required=True,
            path_kind="file",
            description="EIT recording file to load.",
        ),
        StepParameter(
            name="vendor",
            value_type="choice",
            required=False,
            default=None,
            choices=("draeger", "sentec", "timpel"),
            description="Recording vendor. Required unless a custom loader was injected into the adapter.",
        ),
        StepParameter(
            name="loader_options",
            value_type="mapping",
            required=False,
            default=None,
            description="Extra keyword arguments forwarded to M3Session.load_eit().",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="raw_eit",
            artifact_type="eit_pixel_signal",
            description="Raw upstream EITData object (pixel impedance).",
            compatibility_only=True,
        ),
        StepArtifact(
            name="raw_global_impedance",
            artifact_type="eit_global_impedance",
            required=False,
            description="Raw upstream summed/global impedance signal object, when present.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="eit_sequence",
            artifact_type="eit_sequence",
            description="Upstream EIT Sequence container used by downstream adapter calls.",
            public=False,
            compatibility_only=True,
        ),
        StepArtifact(
            name="raw_global_impedance_signal",
            artifact_type="signal",
            required=False,
            description="Native Signal wrapping the raw global impedance, when present.",
        ),
    ),
)
def load(
    session: M3Session,
    *,
    file: str,
    vendor: str | None = None,
    loader_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if loader_options is not None and not isinstance(loader_options, Mapping):
        raise TypeError(
            "eit.load 'loader_options' must be a mapping of keyword arguments "
            f"for M3Session.load_eit(), got {type(loader_options).__name__}."
        )

    session.load_eit(file, vendor=vendor, **dict(loader_options or {}))
    recording = session.eit
    assert recording is not None

    raw_global_impedance_signal = None
    if recording.global_impedance is not None:
        raw_global_impedance_signal = continuous_data_to_signal(
            recording.global_impedance,
            modality="eit",
            channel="global_impedance",
            processing_state="raw",
            source="eitprocessing",
        )
        session.signals.add(raw_global_impedance_signal)

    _record_step(
        session,
        "eit.load",
        metadata=_upstream_metadata(
            source_function="eitprocessing.datahandling.loading.load_eit_data",
            operation="eit.load",
            parameters={"vendor": vendor, "loader_options": dict(loader_options or {})},
        ),
    )
    return {
        "raw_eit": recording.raw,
        "raw_global_impedance": recording.global_impedance,
        "eit_sequence": recording.data,
        "raw_global_impedance_signal": raw_global_impedance_signal,
    }


@register_step(
    "eit.slice",
    reads={"signal": "raw_eit"},
    writes=("result",),
    summary="Slice an EIT signal by sample index or time.",
    description="Slice any upstream EIT signal by sample index or time window, e.g. to select a detection window.",
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_pixel_signal",
            default_context_key="raw_eit",
            description="Upstream EIT signal to slice.",
            compatibility_only=True,
        ),
    ),
    parameters=(
        StepParameter(
            name="start",
            value_type="number",
            required=True,
            description="Slice start: a sample index (mode='index') or seconds (mode='time').",
        ),
        StepParameter(
            name="end",
            value_type="number",
            required=True,
            description="Slice end: a sample index (mode='index') or seconds (mode='time').",
        ),
        StepParameter(
            name="mode",
            value_type="choice",
            default="index",
            choices=("index", "time"),
            description="Whether 'start'/'end' are sample indices or seconds.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="result",
            artifact_type="eit_pixel_signal",
            description="Sliced signal, of the same type as the input.",
            compatibility_only=True,
        ),
    ),
)
def slice_signal(
    signal: Any, *, start: int | float, end: int | float, mode: str = "index"
) -> dict[str, Any]:
    return {
        "result": slice_signal_by_mode(signal, start=start, end=end, slicing_mode=mode)
    }


@register_step(
    "eit.detect_rates",
    reads={"signal": "selected_eit", "session": "session"},
    writes=(
        "respiratory_rate_hz",
        "heart_rate_hz",
        "rate_detector",
        "rate_captures",
        "respiratory_rate_result",
        "heart_rate_result",
    ),
    summary="Estimate respiratory and heart rate from an EIT signal.",
    description="Estimate respiratory and heart rate from an EIT signal via eitprocessing's RateDetection.",
    category="detection",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_pixel_signal",
            default_context_key="selected_eit",
            description="EIT signal (typically the detection-window slice) to estimate rates from.",
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="subject_type",
            value_type="choice",
            default="adult",
            choices=("adult", "neonate"),
            description="Subject type; changes the expected respiratory/heart rate search bands.",
        ),
        StepParameter(
            name="welch_window_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Welch PSD window length. Defaults to the detector's own choice when unset.",
        ),
        StepParameter(
            name="capture",
            value_type="boolean",
            default=False,
            description="Capture intermediate detector diagnostics into 'rate_captures'.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="respiratory_rate_hz",
            artifact_type="scalar_metric",
            unit="Hz",
            description="Estimated respiratory rate.",
        ),
        StepArtifact(
            name="heart_rate_hz",
            artifact_type="scalar_metric",
            unit="Hz",
            description="Estimated heart rate.",
        ),
        StepArtifact(
            name="rate_detector",
            artifact_type="eit_rate_detector",
            description="Configured upstream RateDetection instance, reusable by later steps.",
            public=False,
            compatibility_only=True,
        ),
        StepArtifact(
            name="rate_captures",
            artifact_type="diagnostic_summary",
            required=False,
            description="Intermediate detector diagnostics, when 'capture' is set.",
        ),
        StepArtifact(
            name="respiratory_rate_result",
            artifact_type="parameter_result",
            unit="Hz",
            description="Native ParameterResult for the respiratory rate.",
        ),
        StepArtifact(
            name="heart_rate_result",
            artifact_type="parameter_result",
            unit="Hz",
            description="Native ParameterResult for the heart rate.",
        ),
    ),
)
def detect_rates(
    signal: Any,
    session: M3Session,
    *,
    subject_type: Literal["adult", "neonate"] = "adult",
    welch_window_seconds: float | None = None,
    capture: bool = False,
) -> dict[str, Any]:
    rates = session.eit_adapter.detect_rates(
        signal,
        subject_type=subject_type,
        welch_window_seconds=welch_window_seconds,
        capture=capture,
    )
    respiratory_rate_hz = rates["respiratory_rate_hz"]
    heart_rate_hz = rates["heart_rate_hz"]
    for name, value in (
        ("respiratory_rate_hz", respiratory_rate_hz),
        ("heart_rate_hz", heart_rate_hz),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"eit.detect_rates produced a non-finite/negative {name}: {value!r}."
            )

    metadata = _upstream_metadata(
        source_function="eitprocessing.features.rate_detection.RateDetection.apply",
        operation="eit.detect_rates",
        parameters={
            "subject_type": subject_type,
            "welch_window_seconds": welch_window_seconds,
            "capture": capture,
        },
    )
    respiratory_rate_result = ParameterResult(
        name="respiratory_rate",
        value=respiratory_rate_hz,
        modality="eit",
        unit="Hz",
        method="eitprocessing.RateDetection",
        metadata=dict(metadata),
    )
    heart_rate_result = ParameterResult(
        name="heart_rate",
        value=heart_rate_hz,
        modality="eit",
        unit="Hz",
        method="eitprocessing.RateDetection",
        metadata=dict(metadata),
    )
    session.parameter_results.add(respiratory_rate_result)
    session.parameter_results.add(heart_rate_result)

    _record_step(session, "eit.detect_rates", metadata=metadata)
    return {
        "respiratory_rate_hz": respiratory_rate_hz,
        "heart_rate_hz": heart_rate_hz,
        "rate_detector": rates["rate_detector"],
        "rate_captures": rates["rate_captures"],
        "respiratory_rate_result": respiratory_rate_result,
        "heart_rate_result": heart_rate_result,
    }


@register_step(
    "eit.mdn_filter",
    reads={
        "signal": None,
        "respiratory_rate_hz": "respiratory_rate_hz",
        "heart_rate_hz": "heart_rate_hz",
        "eit_sequence": "eit_sequence",
        "session": "session",
    },
    writes=("filtered_eit", "filter_captures", "filtered_eit_signal"),
    summary="Apply an MDN heart-rate-removal filter to EIT data.",
    description="Apply eitprocessing's MDN filter to remove the cardiac (heart-rate) component from EIT pixel data.",
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_pixel_signal",
            description="Upstream EIT pixel signal to filter.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="respiratory_rate_hz",
            artifact_type="scalar_metric",
            unit="Hz",
            description="Respiratory rate used to separate the cardiac component.",
        ),
        StepArtifact(
            name="heart_rate_hz",
            artifact_type="scalar_metric",
            unit="Hz",
            description="Heart rate to remove.",
        ),
        StepArtifact(
            name="eit_sequence",
            artifact_type="eit_sequence",
            default_context_key="eit_sequence",
            description="Sequence the filtered signal is added onto.",
            public=False,
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="label",
            value_type="string",
            default="filtered",
            description="Label assigned to the filtered signal.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="filtered_eit",
            artifact_type="eit_pixel_signal",
            description="MDN-filtered upstream EIT signal.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="filter_captures",
            artifact_type="diagnostic_summary",
            description="Intermediate MDN filter diagnostics.",
        ),
        StepArtifact(
            name="filtered_eit_signal",
            artifact_type="signal",
            description="Native Signal wrapping the filtered pixel impedance.",
        ),
    ),
)
def mdn_filter(
    signal: Any,
    respiratory_rate_hz: float,
    heart_rate_hz: float,
    eit_sequence: Any,
    session: M3Session,
    *,
    label: str = "filtered",
) -> dict[str, Any]:
    result = session.eit_adapter.apply_mdn(
        signal,
        respiratory_rate_hz=respiratory_rate_hz,
        heart_rate_hz=heart_rate_hz,
        label=label,
    )
    filtered_eit = result["filtered_eit"]
    filter_captures = result["filter_captures"]
    add_to_collection(eit_sequence.eit_data, filtered_eit)

    metadata = _upstream_metadata(
        source_function="eitprocessing.filters.mdn.MDNFilter.apply",
        operation="eit.mdn_filter",
        parameters={
            "respiratory_rate_hz": respiratory_rate_hz,
            "heart_rate_hz": heart_rate_hz,
            "label": label,
        },
    )
    filtered_eit_signal = Signal(
        values=filtered_eit.pixel_impedance,
        time=filtered_eit.time,
        sample_frequency=getattr(filtered_eit, "sample_frequency", None),
        unit=getattr(filtered_eit, "unit", None),
        name=getattr(filtered_eit, "name", None)
        or getattr(filtered_eit, "label", None),
        modality="eit",
        channel="pixel_impedance",
        processing_state="filtered",
        source="eitprocessing",
        metadata=dict(metadata),
    )
    session.signals.add(filtered_eit_signal)

    _record_step(session, "eit.mdn_filter", metadata=metadata)
    return {
        "filtered_eit": filtered_eit,
        "filter_captures": filter_captures,
        "filtered_eit_signal": filtered_eit_signal,
    }


@register_step(
    "eit.butterworth_filter",
    reads={"signal": "raw_eit", "eit_sequence": "eit_sequence"},
    writes=("filtered_eit", "filter_captures"),
    summary="Apply a lowpass/bandpass Butterworth filter to EIT pixel data.",
    description="Apply a lowpass or bandpass Butterworth filter to EIT pixel impedance, as an alternative to eit.mdn_filter.",
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    alternatives=("eit.mdn_filter",),
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_pixel_signal",
            default_context_key="raw_eit",
            description="Upstream EIT pixel signal to filter.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="eit_sequence",
            artifact_type="eit_sequence",
            description="Sequence the filtered signal is added onto.",
            public=False,
            compatibility_only=True,
        ),
    ),
    parameters=(
        StepParameter(
            name="mode",
            value_type="choice",
            default="lowpass",
            choices=("lowpass", "bandpass"),
            description="Filter type. 'bandpass' uses ('highpass_hz', 'lowpass_hz') as the passband.",
        ),
        StepParameter(
            name="lowpass_hz",
            value_type="number",
            default=1.0,
            unit="Hz",
            minimum=0,
            description="Lowpass cutoff (also the bandpass upper edge).",
        ),
        StepParameter(
            name="highpass_hz",
            value_type="number",
            default=0.05,
            unit="Hz",
            minimum=0,
            description="Bandpass lower edge; unused when mode='lowpass'.",
        ),
        StepParameter(
            name="order",
            value_type="integer",
            default=4,
            minimum=1,
            description="Butterworth filter order.",
        ),
        StepParameter(
            name="label",
            value_type="string",
            default="filtered",
            description="Label assigned to the filtered signal.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="filtered_eit",
            artifact_type="eit_pixel_signal",
            description="Butterworth-filtered upstream EIT signal.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="filter_captures",
            artifact_type="diagnostic_summary",
            description="Intermediate Butterworth filter diagnostics.",
        ),
    ),
)
def butterworth_filter(
    signal: Any,
    eit_sequence: Any,
    *,
    mode: Literal["lowpass", "bandpass"] = "lowpass",
    lowpass_hz: float = 1.0,
    highpass_hz: float = 0.05,
    order: int = 4,
    label: str = "filtered",
) -> dict[str, Any]:
    import numpy as np
    from eitprocessing.filters.butterworth_filters import ButterworthFilter

    # Upstream annotates `cutoff_frequency` as `float | tuple[float]` (a
    # one-element tuple), but its bandpass/bandstop branch actually requires
    # and accepts a two-element tuple; the cast reflects the real contract.
    cutoff_frequency = cast(
        "float | tuple[float]",
        lowpass_hz if mode == "lowpass" else (highpass_hz, lowpass_hz),
    )
    captures: dict[str, Any] = {}
    filtered_pixels = ButterworthFilter(
        filter_type=mode,
        cutoff_frequency=cutoff_frequency,
        order=order,
        sample_frequency=signal.sample_frequency,
    ).apply(np.nan_to_num(signal.pixel_impedance), axis=0, captures=captures)
    filtered_eit = copy.deepcopy(signal)
    filtered_eit.label = label
    filtered_eit.name = f"{mode.title()}-filtered EIT data"
    filtered_eit.description = f"EIT data filtered with a {mode} Butterworth filter."
    filtered_eit.pixel_impedance = filtered_pixels
    add_to_collection(eit_sequence.eit_data, filtered_eit)
    return {"filtered_eit": filtered_eit, "filter_captures": captures}


@register_step(
    "eit.global_impedance",
    reads={"signal": "filtered_eit", "eit_sequence": "eit_sequence"},
    writes=("global_impedance",),
    summary="Compute and store the summed (global) impedance of an EIT signal.",
    description="Sum pixel impedance across the image to produce the single-channel global impedance waveform used for breath detection.",
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description="Filtered EIT pixel signal to sum.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="eit_sequence",
            artifact_type="eit_sequence",
            description="Sequence the summed signal is added onto.",
            public=False,
            compatibility_only=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="global_impedance",
            artifact_type="eit_global_impedance",
            description="Summed (global) impedance waveform.",
            compatibility_only=True,
        ),
    ),
)
def global_impedance(signal: Any, eit_sequence: Any) -> dict[str, Any]:
    summed = signal.get_summed_impedance()
    add_to_collection(eit_sequence.continuous_data, summed)
    return {"global_impedance": summed}


@register_step(
    "eit.detect_breaths",
    reads={"signal": "detection_signal"},
    writes=("breath_intervals", "breath_detector"),
    summary="Detect breaths on a continuous EIT impedance signal.",
    description="Detect breath start/end times on the global impedance waveform via eitprocessing's BreathDetection.",
    category="detection",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_global_impedance",
            default_context_key="detection_signal",
            description="Continuous impedance signal (typically the detection-window slice) to detect breaths on.",
            compatibility_only=True,
        ),
    ),
    parameters=(
        StepParameter(
            name="min_duration_s",
            value_type="number",
            default=2 / 3,
            unit="s",
            minimum=0,
            description="Minimum breath duration accepted by the detector.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="breath_intervals",
            artifact_type="interval_collection",
            description="Detected breath intervals.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="breath_detector",
            artifact_type="eit_breath_detector",
            description="Configured upstream BreathDetection instance, reusable by later steps.",
            public=False,
            compatibility_only=True,
        ),
    ),
)
def detect_breaths(signal: Any, *, min_duration_s: float = 2 / 3) -> dict[str, Any]:
    from eitprocessing.features.breath_detection import BreathDetection

    detector = BreathDetection(minimum_duration=min_duration_s)
    return {
        "breath_intervals": detector.find_breaths(signal),
        "breath_detector": detector,
    }


@register_step(
    "eit.normalize_breaths",
    reads={"breath_intervals": "breath_intervals", "session": "session"},
    writes=(),
    summary="Normalize detected EIT breath intervals into session events.",
    description="Convert detected EIT breath intervals into native BreathEvents and store them on the session as 'eit_breaths'.",
    category="detection",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="breath_intervals",
            artifact_type="interval_collection",
            default_context_key="breath_intervals",
            description="Breath intervals from 'eit.detect_breaths' to normalize.",
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
)
def normalize_breaths(breath_intervals: Any, session: M3Session) -> dict[str, Any]:
    events = session.eit_adapter.detect_breaths({"breath_intervals": breath_intervals})
    session.add_events("eit_breaths", events)
    return {}


@register_step(
    "eit.continuous_tiv",
    reads={
        "signal": "global_impedance",
        "eit_sequence": "eit_sequence",
        "breath_detector": "breath_detector",
    },
    writes=("continuous_tiv",),
    summary="Compute continuous tidal impedance variation (TIV).",
    description="Compute per-breath tidal impedance variation on the global impedance waveform via eitprocessing's TIV.",
    category="parameters",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_global_impedance",
            default_context_key="global_impedance",
            description="Global impedance waveform to compute TIV on.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="eit_sequence",
            artifact_type="eit_sequence",
            description="Sequence the result is added onto.",
            public=False,
            compatibility_only=True,
        ),
        StepArtifact(
            name="breath_detector",
            artifact_type="eit_breath_detector",
            description="Configured breath detector from 'eit.detect_breaths'.",
            public=False,
            compatibility_only=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="continuous_tiv",
            artifact_type="eit_sparse_data",
            description="Per-breath continuous TIV values (upstream SparseData).",
            compatibility_only=True,
        ),
    ),
)
def continuous_tiv(
    signal: Any, eit_sequence: Any, breath_detector: Any
) -> dict[str, Any]:
    from eitprocessing.parameters.tidal_impedance_variation import TIV

    result: Any = TIV(breath_detection=breath_detector).compute_parameter(
        signal, sequence=eit_sequence, store=False, result_label="continuous_tivs"
    )
    add_to_collection(eit_sequence.sparse_data, result)
    return {"continuous_tiv": result}


@register_step(
    "eit.eeli",
    reads={
        "signal": "global_impedance",
        "eit_sequence": "eit_sequence",
        "breath_detector": "breath_detector",
        "session": "session",
    },
    writes=("eeli", "eeli_result"),
    summary="Compute end-expiratory lung impedance (EELI).",
    description="Compute per-breath end-expiratory lung impedance (EELI) via eitprocessing's EELI parameter.",
    category="parameters",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_global_impedance",
            default_context_key="global_impedance",
            description="Global impedance waveform to compute EELI on.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="eit_sequence",
            artifact_type="eit_sequence",
            description="Sequence the result is added onto.",
            public=False,
            compatibility_only=True,
        ),
        StepArtifact(
            name="breath_detector",
            artifact_type="eit_breath_detector",
            description="Configured breath detector from 'eit.detect_breaths'.",
            public=False,
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="result_label",
            value_type="string",
            default="continuous_eelis",
            description="Label the upstream result is stored under.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="eeli",
            artifact_type="eit_sparse_data",
            description="Per-breath EELI values (upstream SparseData).",
            compatibility_only=True,
        ),
        StepArtifact(
            name="eeli_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult for EELI, one value per breath.",
        ),
    ),
)
def eeli(
    signal: Any,
    eit_sequence: Any,
    breath_detector: Any,
    session: M3Session,
    *,
    result_label: str = "continuous_eelis",
) -> dict[str, Any]:
    result = session.eit_adapter.compute_eeli(
        signal,
        sequence=eit_sequence,
        breath_detector=breath_detector,
        result_label=result_label,
    )
    metadata = _upstream_metadata(
        source_function="eitprocessing.parameters.eeli.EELI.compute_parameter",
        operation="eit.eeli",
        parameters={"result_label": result_label},
    )
    eeli_result = _sparse_data_to_array_parameter(
        result, modality="eit", method="eitprocessing.EELI", metadata=dict(metadata)
    )
    session.parameter_results.add(eeli_result)

    _record_step(session, "eit.eeli", metadata=metadata)
    return {"eeli": result, "eeli_result": eeli_result}


@register_step(
    "eit.pixel_tiv",
    reads={
        "filtered_eit": "filtered_eit",
        "signal": "global_impedance",
        "eit_sequence": "eit_sequence",
        "breath_detector": "breath_detector",
        "session": "session",
    },
    writes=("pixel_tiv", "pixel_tiv_result"),
    summary="Compute per-pixel tidal impedance variation (TIV).",
    description="Compute per-breath, per-pixel tidal impedance variation via eitprocessing's TIV.compute_pixel_parameter.",
    category="parameters",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="filtered_eit",
            artifact_type="eit_pixel_signal",
            description="Filtered EIT pixel signal to compute per-pixel TIV on.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="signal",
            artifact_type="eit_global_impedance",
            default_context_key="global_impedance",
            description="Global impedance waveform supplying breath timing.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="eit_sequence",
            artifact_type="eit_sequence",
            description="Sequence the result is added onto.",
            public=False,
            compatibility_only=True,
        ),
        StepArtifact(
            name="breath_detector",
            artifact_type="eit_breath_detector",
            description="Configured breath detector from 'eit.detect_breaths'.",
            public=False,
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="tiv_timing",
            value_type="choice",
            default="continuous",
            choices=("pixel", "continuous"),
            description="Whether breath timing is taken per-pixel or from the continuous (global) signal.",
        ),
        StepParameter(
            name="result_label",
            value_type="string",
            default="pixel_tivs",
            description="Label the upstream result is stored under.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pixel_tiv",
            artifact_type="eit_sparse_data",
            description="Per-breath, per-pixel TIV values (upstream SparseData).",
            compatibility_only=True,
        ),
        StepArtifact(
            name="pixel_tiv_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult, shape (breath, row, column).",
            axes=("breath", "row", "column"),
        ),
    ),
)
def pixel_tiv(
    filtered_eit: Any,
    signal: Any,
    eit_sequence: Any,
    breath_detector: Any,
    session: M3Session,
    *,
    tiv_timing: Literal["pixel", "continuous"] = "continuous",
    result_label: str = "pixel_tivs",
) -> dict[str, Any]:
    result = session.eit_adapter.compute_pixel_tiv(
        filtered_eit,
        signal,
        sequence=eit_sequence,
        breath_detector=breath_detector,
        tiv_timing=tiv_timing,
        result_label=result_label,
    )

    values = _object_array_to_float(result.values)
    time = _object_array_to_float(result.time)
    valid_breath_indices = [
        index for index in range(values.shape[0]) if not np.all(np.isnan(values[index]))
    ]

    metadata = _upstream_metadata(
        source_function=(
            "eitprocessing.parameters.tidal_impedance_variation."
            "TIV.compute_pixel_parameter"
        ),
        operation="eit.pixel_tiv",
        parameters={"tiv_timing": tiv_timing, "result_label": result_label},
    )
    parameter_metadata = dict(metadata)
    parameter_metadata.update(
        {
            "time": time.tolist(),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "axes": ["breath", "row", "column"],
            "tiv_timing": tiv_timing,
            "result_label": result_label,
            "valid_breath_indices": valid_breath_indices,
            "valid_breath_fraction": (
                len(valid_breath_indices) / values.shape[0] if values.shape[0] else 0.0
            ),
        }
    )
    pixel_tiv_result = ParameterResult(
        name=getattr(result, "label", None) or result_label,
        value=values,
        modality="eit",
        unit=getattr(result, "unit", None),
        method="eitprocessing.TIV",
        metadata=parameter_metadata,
    )
    session.parameter_results.add(pixel_tiv_result)

    _record_step(session, "eit.pixel_tiv", metadata=metadata)
    return {"pixel_tiv": result, "pixel_tiv_result": pixel_tiv_result}


_ALLOWED_PIXEL_BREATH_PHASE_MODES = {"negative amplitude", "phase shift", "none", None}


@register_step(
    "eit.pixel_breaths",
    reads={
        "eit_data": "filtered_eit",
        "timing_data": "global_impedance",
        "eit_sequence": "eit_sequence",
        "session": "session",
    },
    writes=("pixel_breaths", "pixel_breath_timing_result"),
    summary="Detect per-pixel breath timing (start/middle/end of in-/deflation).",
    description="Detect per-pixel breath start/middle/end timing via eitprocessing's PixelBreath.",
    category="detection",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="eit_data",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description="Filtered EIT pixel signal to detect pixel breaths on.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="timing_data",
            artifact_type="eit_global_impedance",
            default_context_key="global_impedance",
            description="Global impedance waveform supplying overall breath timing.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="eit_sequence",
            artifact_type="eit_sequence",
            description="Sequence the result is added onto.",
            public=False,
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="phase_correction_mode",
            value_type="choice",
            required=False,
            default="negative amplitude",
            choices=("negative amplitude", "phase shift", "none"),
            description="Per-pixel phase correction method. Null is also accepted, equivalent to 'none'.",
        ),
        StepParameter(
            name="minimum_duration_seconds",
            value_type="number",
            default=2 / 3,
            unit="s",
            minimum=0,
            description="Minimum per-pixel breath duration accepted.",
        ),
        StepParameter(
            name="result_label",
            value_type="string",
            default="pixel_breaths",
            description="Label the upstream result is stored under.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pixel_breaths",
            artifact_type="eit_sparse_data",
            description="Per-pixel breath objects (upstream SparseData of Breath | None).",
            compatibility_only=True,
        ),
        StepArtifact(
            name="pixel_breath_timing_result",
            artifact_type="parameter_result",
            unit="s",
            description="Native array-valued ParameterResult of [start, middle, end] landmark times.",
            axes=("breath", "row", "column", "landmark"),
        ),
    ),
)
def pixel_breaths(
    eit_data: Any,
    timing_data: Any,
    eit_sequence: Any,
    session: M3Session,
    *,
    phase_correction_mode: Literal["negative amplitude", "phase shift", "none"]
    | None = "negative amplitude",
    minimum_duration_seconds: float = 2 / 3,
    result_label: str = "pixel_breaths",
) -> dict[str, Any]:
    if phase_correction_mode not in _ALLOWED_PIXEL_BREATH_PHASE_MODES:
        raise ValueError(
            "eit.pixel_breaths 'phase_correction_mode' must be one of "
            "'negative amplitude', 'phase shift', 'none', or null; "
            f"got {phase_correction_mode!r}."
        )

    result = session.eit_adapter.find_pixel_breaths(
        eit_data,
        timing_data,
        sequence=eit_sequence,
        phase_correction_mode=phase_correction_mode,
        minimum_duration_seconds=minimum_duration_seconds,
        result_label=result_label,
    )

    landmarks = _pixel_breaths_to_landmark_array(result.values)
    valid = ~np.isnan(landmarks[..., 0])

    metadata = _upstream_metadata(
        source_function=(
            "eitprocessing.features.pixel_breath.PixelBreath.find_pixel_breaths"
        ),
        operation="eit.pixel_breaths",
        parameters={
            "phase_correction_mode": phase_correction_mode,
            "minimum_duration_seconds": minimum_duration_seconds,
            "result_label": result_label,
        },
    )
    metadata.update(
        {
            "shape": list(landmarks.shape),
            "dtype": str(landmarks.dtype),
            "axes": ["breath", "row", "column", "landmark"],
            "landmarks": ["start_time", "middle_time", "end_time"],
            "valid_pixel_breath_count": int(valid.sum()),
            "valid_pixel_breath_fraction": float(valid.mean()) if valid.size else 0.0,
        }
    )
    pixel_breath_timing_result = ParameterResult(
        name=result_label,
        value=landmarks,
        modality="eit",
        unit="s",
        method="eitprocessing.PixelBreath",
        metadata=metadata,
    )
    session.parameter_results.add(pixel_breath_timing_result)

    _record_step(session, "eit.pixel_breaths", metadata=metadata)
    return {
        "pixel_breaths": result,
        "pixel_breath_timing_result": pixel_breath_timing_result,
    }


@register_step(
    "eit.roi_tiv_lungspace",
    reads={
        "eit_data": "filtered_eit",
        "timing_data": "global_impedance",
        "session": "session",
    },
    writes=("tiv_lungspace_mask", "tiv_lungspace_captures", "tiv_lungspace_result"),
    summary="Threshold mean pixel TIV into a functional lung-space mask.",
    description="Threshold mean per-pixel TIV into a boolean/NaN-excluded functional lung-space mask via eitprocessing's TIVLungspace.",
    category="roi",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="eit_data",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description="Filtered EIT pixel signal to derive the mask from.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="timing_data",
            artifact_type="eit_global_impedance",
            default_context_key="global_impedance",
            description="Global impedance waveform supplying breath timing.",
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="threshold",
            value_type="number",
            default=0.15,
            minimum=0.0,
            maximum=1.0,
            description="Fraction of the maximum mean pixel TIV a pixel must reach to be included. Must be strictly between 0 and 1.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="tiv_lungspace_mask",
            artifact_type="roi_mask",
            description="Upstream PixelMask; excluded pixels are NaN.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="tiv_lungspace_captures",
            artifact_type="diagnostic_summary",
            description="Intermediate mask-derivation diagnostics.",
        ),
        StepArtifact(
            name="tiv_lungspace_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult (row, column), NaN for excluded pixels.",
            axes=("row", "column"),
        ),
    ),
)
def roi_tiv_lungspace(
    eit_data: Any,
    timing_data: Any,
    session: M3Session,
    *,
    threshold: float = 0.15,
) -> dict[str, Any]:
    _validate_unit_threshold(threshold, step="eit.roi_tiv_lungspace", param="threshold")

    result = session.eit_adapter.compute_tiv_lungspace(
        eit_data, timing_data=timing_data, threshold=threshold
    )
    mask = result["mask"]
    metadata = _upstream_metadata(
        source_function="eitprocessing.roi.tiv.TIVLungspace.apply",
        operation="eit.roi_tiv_lungspace",
        parameters={"threshold": threshold},
    )
    tiv_lungspace_result = _pixel_mask_to_parameter_result(
        mask,
        name="tiv_lungspace_mask",
        method="eitprocessing.TIVLungspace",
        metadata=dict(metadata),
    )
    session.parameter_results.add(tiv_lungspace_result)

    _record_step(session, "eit.roi_tiv_lungspace", metadata=metadata)
    return {
        "tiv_lungspace_mask": mask,
        "tiv_lungspace_captures": result["captures"],
        "tiv_lungspace_result": tiv_lungspace_result,
    }


@register_step(
    "eit.roi_amplitude_lungspace",
    reads={
        "eit_data": "filtered_eit",
        "timing_data": "global_impedance",
        "session": "session",
    },
    writes=(
        "amplitude_lungspace_mask",
        "amplitude_lungspace_captures",
        "amplitude_lungspace_result",
    ),
    summary=(
        "Threshold mean pixel amplitude into a lung-space mask (not "
        "recommended alone; supports eit.roi_watershed)."
    ),
    description=(
        "Threshold mean pixel amplitude into a lung-space mask via "
        "eitprocessing's AmplitudeLungspace. Not recommended as a "
        "general-purpose functional lung-space definition on its own "
        "(may include reconstruction artifacts); intended to support "
        "eit.roi_watershed."
    ),
    category="roi",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="eit_data",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description="Filtered EIT pixel signal to derive the mask from.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="timing_data",
            artifact_type="eit_global_impedance",
            default_context_key="global_impedance",
            description="Global impedance waveform supplying breath timing.",
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="threshold",
            value_type="number",
            default=0.15,
            minimum=0.0,
            maximum=1.0,
            description="Fraction of the maximum mean pixel amplitude a pixel must reach to be included. Must be strictly between 0 and 1.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="amplitude_lungspace_mask",
            artifact_type="roi_mask",
            description="Upstream PixelMask; excluded pixels are NaN.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="amplitude_lungspace_captures",
            artifact_type="diagnostic_summary",
            description="Intermediate mask-derivation diagnostics.",
        ),
        StepArtifact(
            name="amplitude_lungspace_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult (row, column), NaN for excluded pixels.",
            axes=("row", "column"),
        ),
    ),
)
def roi_amplitude_lungspace(
    eit_data: Any,
    timing_data: Any,
    session: M3Session,
    *,
    threshold: float = 0.15,
) -> dict[str, Any]:
    """Threshold mean pixel amplitude into a lung-space mask.

    Warning: upstream does not recommend amplitude alone as a general-purpose
    functional lung-space definition, as it potentially includes
    reconstruction artifacts. It is provided primarily for use by
    `eit.roi_watershed`.
    """

    _validate_unit_threshold(
        threshold, step="eit.roi_amplitude_lungspace", param="threshold"
    )

    result = session.eit_adapter.compute_amplitude_lungspace(
        eit_data, timing_data=timing_data, threshold=threshold
    )
    mask = result["mask"]
    metadata = _upstream_metadata(
        source_function="eitprocessing.roi.amplitude.AmplitudeLungspace.apply",
        operation="eit.roi_amplitude_lungspace",
        parameters={"threshold": threshold},
    )
    amplitude_lungspace_result = _pixel_mask_to_parameter_result(
        mask,
        name="amplitude_lungspace_mask",
        method="eitprocessing.AmplitudeLungspace",
        metadata=dict(metadata),
    )
    session.parameter_results.add(amplitude_lungspace_result)

    _record_step(session, "eit.roi_amplitude_lungspace", metadata=metadata)
    return {
        "amplitude_lungspace_mask": mask,
        "amplitude_lungspace_captures": result["captures"],
        "amplitude_lungspace_result": amplitude_lungspace_result,
    }


@register_step(
    "eit.roi_watershed",
    reads={
        "eit_data": "filtered_eit",
        "timing_data": "global_impedance",
        "session": "session",
    },
    writes=(
        "watershed_lungspace_mask",
        "watershed_captures",
        "watershed_lungspace_result",
    ),
    summary="Derive a lung-space mask with the watershed method (pendelluft-aware).",
    description="Derive a lung-space mask via eitprocessing's watershed method, which stays pendelluft-aware unlike a simple threshold.",
    category="roi",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="eit_data",
            artifact_type="eit_pixel_signal",
            default_context_key="filtered_eit",
            description="Filtered EIT pixel signal to derive the mask from.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="timing_data",
            artifact_type="eit_global_impedance",
            default_context_key="global_impedance",
            description="Global impedance waveform supplying breath timing.",
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="threshold_fraction",
            value_type="number",
            default=0.15,
            minimum=0.0,
            maximum=1.0,
            description="Watershed threshold fraction. Must be strictly between 0 and 1.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="watershed_lungspace_mask",
            artifact_type="roi_mask",
            description="Upstream PixelMask; excluded pixels are NaN.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="watershed_captures",
            artifact_type="diagnostic_summary",
            description="Intermediate mask-derivation diagnostics.",
        ),
        StepArtifact(
            name="watershed_lungspace_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult (row, column), NaN for excluded pixels.",
            axes=("row", "column"),
        ),
    ),
)
def roi_watershed(
    eit_data: Any,
    timing_data: Any,
    session: M3Session,
    *,
    threshold_fraction: float = 0.15,
) -> dict[str, Any]:
    _validate_unit_threshold(
        threshold_fraction, step="eit.roi_watershed", param="threshold_fraction"
    )

    result = session.eit_adapter.compute_watershed_lungspace(
        eit_data, timing_data=timing_data, threshold_fraction=threshold_fraction
    )
    mask = result["mask"]
    metadata = _upstream_metadata(
        source_function="eitprocessing.roi.watershed.WatershedLungspace.apply",
        operation="eit.roi_watershed",
        parameters={"threshold_fraction": threshold_fraction},
    )
    watershed_lungspace_result = _pixel_mask_to_parameter_result(
        mask,
        name="watershed_lungspace_mask",
        method="eitprocessing.WatershedLungspace",
        metadata=dict(metadata),
    )
    session.parameter_results.add(watershed_lungspace_result)

    _record_step(session, "eit.roi_watershed", metadata=metadata)
    return {
        "watershed_lungspace_mask": mask,
        "watershed_captures": result["captures"],
        "watershed_lungspace_result": watershed_lungspace_result,
    }


@register_step(
    "eit.roi_filter_by_size",
    reads={"mask": None, "session": "session"},
    writes=("size_filtered_roi_mask", "size_filtered_roi_result"),
    summary="Keep only connected mask regions at or above a minimum size.",
    description="Drop connected regions of a lung-space mask smaller than a minimum pixel count, via eitprocessing's FilterROIBySize.",
    category="roi",
    modality="eit",
    optional_packages=_EITPROCESSING,
    input_artifacts=(
        StepArtifact(
            name="mask",
            artifact_type="roi_mask",
            description="Lung-space mask to filter, e.g. from 'eit.roi_watershed'.",
            compatibility_only=True,
        ),
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="min_region_size",
            value_type="integer",
            default=10,
            minimum=1,
            unit="pixels",
            description="Minimum connected-region size kept.",
        ),
        StepParameter(
            name="connectivity",
            value_type="integer",
            default=1,
            choices=(1, 2),
            description="Pixel connectivity: 1 (4-connected) or 2 (8-connected).",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="size_filtered_roi_mask",
            artifact_type="roi_mask",
            description="Upstream PixelMask with small regions dropped.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="size_filtered_roi_result",
            artifact_type="parameter_result",
            description="Native array-valued ParameterResult (row, column), NaN for excluded pixels.",
            axes=("row", "column"),
        ),
    ),
)
def roi_filter_by_size(
    mask: Any,
    session: M3Session,
    *,
    min_region_size: int = 10,
    connectivity: Literal[1, 2] = 1,
) -> dict[str, Any]:
    if min_region_size <= 0:
        raise ValueError(
            "eit.roi_filter_by_size 'min_region_size' must be positive, "
            f"got {min_region_size!r}."
        )

    result = session.eit_adapter.filter_roi_by_size(
        mask, min_region_size=min_region_size, connectivity=connectivity
    )
    metadata = _upstream_metadata(
        source_function="eitprocessing.roi.filter_by_size.FilterROIBySize.apply",
        operation="eit.roi_filter_by_size",
        parameters={
            "min_region_size": min_region_size,
            "connectivity": connectivity,
        },
    )
    size_filtered_roi_result = _pixel_mask_to_parameter_result(
        result,
        name="size_filtered_roi_mask",
        method="eitprocessing.FilterROIBySize",
        metadata=dict(metadata),
    )
    session.parameter_results.add(size_filtered_roi_result)

    _record_step(session, "eit.roi_filter_by_size", metadata=metadata)
    return {
        "size_filtered_roi_mask": result,
        "size_filtered_roi_result": size_filtered_roi_result,
    }
