"""Registered EIT rate-detection and filtering pipeline steps."""

from __future__ import annotations

import copy
import math
from typing import Any, Literal, cast

import numpy as np

from m3resp.adapters.eitprocessing_adapter import (
    add_to_collection,
)
from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, Signal
from m3resp.workflows.registry import (
    StepArtifact,
    StepParameter,
    register_step,
)

from ._shared import (
    _EITPROCESSING,
    _SESSION_ARTIFACT,
    _record_step,
    _upstream_metadata,
)


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
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"eit.detect_rates produced a non-finite/non-positive {name}: "
                f"{value!r}."
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
        name=f"MDN-filtered EIT data ({label})",
        description="EIT data filtered with MDN heart-rate noise removal.",
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
        category="impedance",
        channel="pixel_impedance",
        processing_state="intermediate",
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
