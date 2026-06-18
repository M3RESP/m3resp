"""Registered EIT pipeline steps.

Each step wraps a single ``eitprocessing`` operation. Upstream imports are
deferred to call time so the package installs without the optional
``eitprocessing`` dependency.
"""

from __future__ import annotations

import copy
from typing import Any

from m3resp.core.session import M3Session
from m3resp.pipeline.registry import register_step
from m3resp.pipeline.utils import slice_signal_by_mode


def _add_to_collection(collection: Any, value: Any) -> None:
    """Add ``value`` to an eitprocessing collection, tolerating old signatures."""

    try:
        collection.add(value, overwrite=True)
    except TypeError:
        collection.add(value)


@register_step(
    "eit.load",
    reads={"session": "session"},
    writes=("raw_eit", "raw_global_impedance", "eit_sequence"),
    summary="Load an EIT recording into the session.",
)
def load(session: M3Session, *, file: str, vendor: str | None = None) -> dict[str, Any]:
    session.load_eit(file, vendor=vendor)
    recording = session.eit
    assert recording is not None
    return {
        "raw_eit": recording.raw,
        "raw_global_impedance": recording.global_impedance,
        "eit_sequence": recording.data,
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
    reads={"signal": "selected_eit"},
    writes=("respiratory_rate_hz", "heart_rate_hz", "rate_detector", "rate_captures"),
    summary="Estimate respiratory and heart rate from an EIT signal.",
)
def detect_rates(
    signal: Any,
    *,
    subject_type: str = "adult",
    welch_window_seconds: float | None = None,
    capture: bool = False,
) -> dict[str, Any]:
    from eitprocessing.features.rate_detection import RateDetection

    detector = (
        RateDetection(subject_type)
        if welch_window_seconds is None
        else RateDetection(subject_type, welch_window=welch_window_seconds)
    )
    captures: dict[str, Any] = {}
    if capture:
        respiratory_rate_hz, heart_rate_hz = detector.apply(
            signal,
            captures=captures,
            suppress_length_warnings=True,
            suppress_edge_case_warning=True,
        )
    else:
        respiratory_rate_hz, heart_rate_hz = detector.apply(signal)
    return {
        "respiratory_rate_hz": float(respiratory_rate_hz),
        "heart_rate_hz": float(heart_rate_hz),
        "rate_detector": detector,
        "rate_captures": captures,
    }


@register_step(
    "eit.mdn_filter",
    reads={
        "signal": None,
        "respiratory_rate_hz": "respiratory_rate_hz",
        "heart_rate_hz": "heart_rate_hz",
        "eit_sequence": "eit_sequence",
    },
    writes=("filtered_eit", "filter_captures"),
    summary="Apply an MDN heart-rate-removal filter to EIT data.",
)
def mdn_filter(
    signal: Any,
    respiratory_rate_hz: float,
    heart_rate_hz: float,
    eit_sequence: Any,
    *,
    label: str = "filtered",
) -> dict[str, Any]:
    from eitprocessing.filters.mdn import MDNFilter

    captures: dict[str, Any] = {}
    filtered_eit = MDNFilter(
        respiratory_rate=respiratory_rate_hz,
        heart_rate=heart_rate_hz,
    ).apply(signal, label=label, captures=captures)
    _add_to_collection(eit_sequence.eit_data, filtered_eit)
    return {"filtered_eit": filtered_eit, "filter_captures": captures}


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
    mode: str = "lowpass",
    lowpass_hz: float = 1.0,
    highpass_hz: float = 0.05,
    order: int = 4,
    label: str = "filtered",
) -> dict[str, Any]:
    import numpy as np
    from eitprocessing.filters.butterworth_filters import ButterworthFilter

    cutoff_frequency = lowpass_hz if mode == "lowpass" else (highpass_hz, lowpass_hz)
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
    },
    writes=("eeli",),
    summary="Compute end-expiratory lung impedance (EELI).",
)
def eeli(signal: Any, eit_sequence: Any, breath_detector: Any) -> dict[str, Any]:
    from eitprocessing.parameters.eeli import EELI

    result: Any = EELI(breath_detection=breath_detector).compute_parameter(
        signal, sequence=eit_sequence, store=False, result_label="continuous_eelis"
    )
    _add_to_collection(eit_sequence.sparse_data, result)
    return {"eeli": result}


@register_step(
    "eit.pixel_tiv",
    reads={
        "filtered_eit": "filtered_eit",
        "signal": "global_impedance",
        "eit_sequence": "eit_sequence",
        "breath_detector": "breath_detector",
    },
    writes=("pixel_tiv",),
    summary="Compute per-pixel tidal impedance variation (TIV).",
)
def pixel_tiv(
    filtered_eit: Any, signal: Any, eit_sequence: Any, breath_detector: Any
) -> dict[str, Any]:
    from eitprocessing.parameters.tidal_impedance_variation import TIV

    result: Any = TIV(breath_detection=breath_detector).compute_parameter(
        filtered_eit,
        signal,
        eit_sequence,
        tiv_timing="continuous",
        store=False,
        result_label="pixel_tivs",
    )
    _add_to_collection(eit_sequence.sparse_data, result)
    return {"pixel_tiv": result}
