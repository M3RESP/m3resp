"""Registered EIT filtering pipeline steps.

Rate detection lives in `rates.py`; the rates it produces are inputs to the
MDN filter below, but estimating them is not a filtering operation.
"""

from __future__ import annotations

import copy
from typing import Any, Literal, cast

from m3resp.adapters.eitprocessing_adapter import (
    add_to_collection,
    filter_pixels_preserving_gaps,
)
from m3resp.core.session import M3Session
from m3resp.data import Signal
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
    session_writes=("session.signals",),
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
    *,
    respiratory_rate_hz: float,
    heart_rate_hz: float,
    eit_sequence: Any,
    session: M3Session,
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
    reads={"signal": None, "eit_sequence": "eit_sequence", "session": "session"},
    writes=("filtered_eit", "filter_captures", "filtered_eit_signal"),
    summary="Apply a Butterworth filter to EIT pixel data.",
    description="Apply a lowpass, highpass, bandpass or bandstop Butterworth filter to EIT pixel impedance, as an alternative to eit.mdn_filter.",
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    alternatives=("eit.mdn_filter",),
    session_writes=("session.signals",),
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_pixel_signal",
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
        _SESSION_ARTIFACT,
    ),
    parameters=(
        StepParameter(
            name="mode",
            value_type="choice",
            default="lowpass",
            choices=("lowpass", "highpass", "bandpass", "bandstop"),
            description=(
                "Filter type. 'lowpass' uses 'lowpass_hz', 'highpass' uses "
                "'highpass_hz', and 'bandpass'/'bandstop' use both as the "
                "band edges ('highpass_hz', 'lowpass_hz')."
            ),
        ),
        StepParameter(
            name="lowpass_hz",
            value_type="number",
            default=1.0,
            unit="Hz",
            minimum=0,
            description="Upper band edge; used by every mode except 'highpass'.",
        ),
        StepParameter(
            name="highpass_hz",
            value_type="number",
            default=0.05,
            unit="Hz",
            minimum=0,
            description="Lower band edge; used by every mode except 'lowpass'.",
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
        StepArtifact(
            name="filtered_eit_signal",
            artifact_type="signal",
            description="Native Signal wrapping the filtered pixel impedance.",
        ),
    ),
)
def butterworth_filter(
    signal: Any,
    *,
    eit_sequence: Any,
    session: M3Session,
    mode: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
    lowpass_hz: float = 1.0,
    highpass_hz: float = 0.05,
    order: int = 4,
    label: str = "filtered",
) -> dict[str, Any]:
    from eitprocessing.filters.butterworth_filters import ButterworthFilter

    # One edge for the single-sided modes, both edges for the two-sided ones.
    # Upstream annotates `cutoff_frequency` as `float | tuple[float]` (a
    # one-element tuple), but its bandpass/bandstop branch actually requires
    # and accepts a two-element tuple; the cast reflects the real contract.
    single_sided = {"lowpass": lowpass_hz, "highpass": highpass_hz}
    cutoff_frequency = cast(
        "float | tuple[float]",
        single_sided.get(mode, (highpass_hz, lowpass_hz)),
    )
    captures: dict[str, Any] = {}
    butterworth = ButterworthFilter(
        filter_type=mode,
        cutoff_frequency=cutoff_frequency,
        order=order,
        sample_frequency=signal.sample_frequency,
    )
    filtered_pixels = filter_pixels_preserving_gaps(
        signal.pixel_impedance,
        operation="eit.butterworth_filter",
        apply=lambda pixels: butterworth.apply(pixels, axis=0, captures=captures),
        captures=captures,
    )
    filtered_eit = copy.deepcopy(signal)
    filtered_eit.label = label
    filtered_eit.name = f"{mode.title()}-filtered EIT data"
    filtered_eit.description = f"EIT data filtered with a {mode} Butterworth filter."
    filtered_eit.pixel_impedance = filtered_pixels
    add_to_collection(eit_sequence.eit_data, filtered_eit)

    metadata = _upstream_metadata(
        source_function="eitprocessing.filters.butterworth_filters.ButterworthFilter.apply",
        operation="eit.butterworth_filter",
        parameters={
            "mode": mode,
            "lowpass_hz": lowpass_hz,
            "highpass_hz": highpass_hz,
            "order": order,
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

    _record_step(session, "eit.butterworth_filter", metadata=metadata)
    return {
        "filtered_eit": filtered_eit,
        "filter_captures": captures,
        "filtered_eit_signal": filtered_eit_signal,
    }
