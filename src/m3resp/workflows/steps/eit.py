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
    _add_to_collection,
    _continuous_data_to_signal,
)
from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, Signal
from m3resp.workflows.registry import register_step
from m3resp.workflows.utils import slice_signal_by_mode


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
        raw_global_impedance_signal = _continuous_data_to_signal(
            recording.global_impedance,
            modality="eit",
            channel="global_impedance",
            processing_state="raw",
            source="eitprocessing",
        )
        session.signals.add(raw_global_impedance_signal)

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
    _add_to_collection(eit_sequence.eit_data, filtered_eit)

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
        metadata=_upstream_metadata(
            source_function="eitprocessing.filters.mdn.MDNFilter.apply",
            operation="eit.mdn_filter",
            parameters={
                "respiratory_rate_hz": respiratory_rate_hz,
                "heart_rate_hz": heart_rate_hz,
                "label": label,
            },
        ),
    )
    session.signals.add(filtered_eit_signal)

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
    _add_to_collection(eit_sequence.eit_data, filtered_eit)
    return {"filtered_eit": filtered_eit, "filter_captures": captures}


@register_step(
    "eit.global_impedance",
    reads={"signal": "filtered_eit", "eit_sequence": "eit_sequence"},
    writes=("global_impedance",),
    summary="Compute and store the summed (global) impedance of an EIT signal.",
)
def global_impedance(signal: Any, eit_sequence: Any) -> dict[str, Any]:
    summed = signal.get_summed_impedance()
    _add_to_collection(eit_sequence.continuous_data, summed)
    return {"global_impedance": summed}


@register_step(
    "eit.detect_breaths",
    reads={"signal": "detection_signal"},
    writes=("breath_intervals", "breath_detector"),
    summary="Detect breaths on a continuous EIT impedance signal.",
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
)
def continuous_tiv(
    signal: Any, eit_sequence: Any, breath_detector: Any
) -> dict[str, Any]:
    from eitprocessing.parameters.tidal_impedance_variation import TIV

    result: Any = TIV(breath_detection=breath_detector).compute_parameter(
        signal, sequence=eit_sequence, store=False, result_label="continuous_tivs"
    )
    _add_to_collection(eit_sequence.sparse_data, result)
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
    eeli_result = _sparse_data_to_array_parameter(
        result,
        modality="eit",
        method="eitprocessing.EELI",
        metadata=_upstream_metadata(
            source_function="eitprocessing.parameters.eeli.EELI.compute_parameter",
            operation="eit.eeli",
            parameters={"result_label": result_label},
        ),
    )
    session.parameter_results.add(eeli_result)
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
    metadata.update(
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
        metadata=metadata,
    )
    session.parameter_results.add(pixel_tiv_result)
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
    return {
        "pixel_breaths": result,
        "pixel_breath_timing_result": pixel_breath_timing_result,
    }
