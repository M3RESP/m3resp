"""Registered EIT rate-detection workflow steps."""

from __future__ import annotations

import math
from typing import Any, Literal

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult
from m3resp.eit.processing import RateDetectionSettings
from m3resp.eit.processing import detect_rates as detect_rates_of
from m3resp.workflows.registry import (
    StepArtifact,
    StepParameter,
    register_step,
)

from ._shared import (
    _SESSION_ARTIFACT,
    _as_signal,
    _step_provenance,
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
    description="Estimate respiratory and heart rate from the frequency spectra of EIT images (m3resp.eit.processing.detect_rates).",
    category="detection",
    modality="eit",
    session_writes=("session.parameter_results",),
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_pixel_signal",
            default_context_key="selected_eit",
            description="EIT images (pixel impedance), typically the detection-window slice, to estimate rates from.",
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
                "Welch power-spectrum window length. Left unset, it is 30 s; "
                "the shortest allowed window is 10 s."
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
            description="The settings of the rate detection (RateDetectionSettings), for the rate figure.",
            public=False,
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
    record=True,
)
def detect_rates(
    signal: Any,
    *,
    session: M3Session,
    subject_type: Literal["adult", "neonate"] = "adult",
    welch_window_seconds: float | None = None,
    capture: bool = False,
) -> dict[str, Any]:
    """Find the breathing rate and the heart rate of EIT images.

    The rates come from ``m3resp.eit.processing.detect_rates``. Both rates go
    to ``session.parameter_results``.

    Args:
        signal (Any): EIT images as a ``Signal``, time in seconds.
        session (M3Session): Session that stores the rates.
        subject_type (Literal["adult", "neonate"]): Sets the rate ranges.
        welch_window_seconds (float | None): Length of one Welch window in
            seconds. None gives 30 s.
        capture (bool): If True, also return the spectra for the rate figure
            and give no warnings about short data or a rate at the edge of
            its range.

    Returns:
        dict[str, Any]: ``respiratory_rate_hz`` and ``heart_rate_hz`` in Hz,
            ``rate_detector`` (the ``RateDetectionSettings``),
            ``rate_captures`` (the spectra, empty when ``capture`` is False)
            and the two rates as ``ParameterResult``s.

    Raises:
        TypeError: If ``signal`` is not a ``Signal``.
        ValueError: If a setting is invalid, or if a rate is NaN.
    """

    settings = (
        RateDetectionSettings(subject_type)
        if welch_window_seconds is None
        else RateDetectionSettings(subject_type, welch_window=welch_window_seconds)
    )
    captures: dict[str, Any] = {}
    respiratory_rate_hz, heart_rate_hz = detect_rates_of(
        _as_signal(signal, name="signal", step="eit.detect_rates"),
        settings,
        captures=captures if capture else None,
        suppress_length_warnings=capture,
        suppress_edge_case_warning=capture,
    )
    for name, value in (
        ("respiratory_rate_hz", respiratory_rate_hz),
        ("heart_rate_hz", heart_rate_hz),
    ):
        # NaN means no rate could be estimated. This is the only failure that
        # can occur: the rate comes from the search band and the parabola
        # refinement moves it by less than half a frequency step, so it stays
        # above zero. A frequency step in which no pixel was measured leaves
        # NaN in the averaged pixel spectrum, and the refinement keeps it.
        if math.isnan(value):
            raise ValueError(
                f"eit.detect_rates could not estimate a {name}: the detector "
                "returned NaN, which happens when the power spectrum has no "
                "measured pixels in the frequency band being searched. Check "
                "that the signal covers enough breaths and that the subject "
                "type matches the expected rate range."
            )

    metadata = _step_provenance(
        source_function="m3resp.eit.processing.detect_rates",
        operation="eit.detect_rates",
        parameters={
            "subject_type": subject_type,
            "welch_window_seconds": welch_window_seconds,
            "capture": capture,
        },
        source_package="m3resp",
        implementation="m3resp.eit.processing.rate_detection",
    )
    respiratory_rate_result = ParameterResult(
        name="respiratory_rate",
        value=respiratory_rate_hz,
        modality="eit",
        unit="Hz",
        method="m3resp.eit.processing.detect_rates",
        metadata=dict(metadata),
    )
    heart_rate_result = ParameterResult(
        name="heart_rate",
        value=heart_rate_hz,
        modality="eit",
        unit="Hz",
        method="m3resp.eit.processing.detect_rates",
        metadata=dict(metadata),
    )
    session.parameter_results.add(respiratory_rate_result)
    session.parameter_results.add(heart_rate_result)

    return {
        "respiratory_rate_hz": respiratory_rate_hz,
        "heart_rate_hz": heart_rate_hz,
        "rate_detector": settings,
        "rate_captures": captures,
        "respiratory_rate_result": respiratory_rate_result,
        "heart_rate_result": heart_rate_result,
        "_provenance": metadata,
    }
