"""Registered EIT rate-detection pipeline steps."""

from __future__ import annotations

import math
from typing import Any, Literal

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult
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
    session_writes=("session.parameter_results",),
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
            minimum=10,
            description=(
                "Welch power-spectrum window length. Left unset, the detector "
                "uses its own default of 30 s; it rejects anything below 10 s."
            ),
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
    *,
    session: M3Session,
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
        # Checked against whichever rate detector `session.eit_adapter` wraps,
        # not just eitprocessing's. With eitprocessing's own detector only the
        # NaN case can occur - the rate is drawn from the search band and its
        # parabolic refinement shifts it by less than half a frequency bin, so
        # it cannot come back zero or negative, but a frequency bin in which no
        # pixel was measured leaves NaN in the averaged pixel power spectrum
        # and the refinement carries that through. A substituted or custom
        # detector is under no such constraint, so both cases are rejected.
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
