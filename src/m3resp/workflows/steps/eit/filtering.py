"""Registered EIT filtering workflow steps.

Rate detection lives in `rates.py`; the rates it produces are inputs to the
MDN filter below. The MDN filter is ``m3resp.eit.processing.mdn_filter``.
"""

from __future__ import annotations

from typing import Any, get_args

from m3resp.adapters.eitprocessing_adapter import (
    MAX_EIT_BUTTERWORTH_ORDER,
    add_to_collection,
    as_signal,
    butterworth_filter_images,
    signal_to_eit_data,
)
from m3resp.core.session import M3Session
from m3resp.eit.processing.mdn_filter import (
    NOISE_FREQUENCY_LIMIT,
    NOTCH_DISTANCE,
    ORDER,
)
from m3resp.processing.filters import ButterworthFilterType
from m3resp.workflows.registry import (
    StepArtifact,
    StepParameter,
    register_step,
)

from ._shared import (
    _EITPROCESSING,
    _SESSION_ARTIFACT,
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
    summary="Remove the heart signal from EIT images with the MDN filter.",
    description=(
        "Remove the heart signal from EIT images with the MDN (multiple digital "
        "notch) filter: a band-stop filter around the heart rate and each of its "
        "harmonics up to the noise frequency limit, then a low-pass filter at "
        "that limit."
    ),
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    session_writes=("session.signals",),
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_pixel_signal",
            description="EIT images (pixel impedance) to filter.",
        ),
        StepArtifact(
            name="respiratory_rate_hz",
            artifact_type="scalar_metric",
            unit="Hz",
            description="Breathing rate; the first removed band starts at least halfway between it and the heart rate.",
        ),
        StepArtifact(
            name="heart_rate_hz",
            artifact_type="scalar_metric",
            unit="Hz",
            description="Heart rate to remove, with its harmonics.",
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
            name="noise_frequency_limit_hz",
            value_type="number",
            default=NOISE_FREQUENCY_LIMIT,
            unit="Hz",
            minimum=0,
            description=(
                "Low-pass edge (default 220 per minute). Harmonics of the heart "
                "rate are removed up to this frequency, and everything above it."
            ),
        ),
        StepParameter(
            name="notch_distance_hz",
            value_type="number",
            default=NOTCH_DISTANCE,
            unit="Hz",
            minimum=0,
            description="Half the width of each removed band (default 10 per minute).",
        ),
        StepParameter(
            name="order",
            value_type="integer",
            default=ORDER,
            minimum=1,
            maximum=10,
            description="Butterworth order of each band-stop and low-pass filter.",
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
            description="MDN-filtered EIT images as a Signal.",
        ),
        StepArtifact(
            name="filter_captures",
            artifact_type="diagnostic_summary",
            description=(
                "In-between results: low_pass_frequency, n_harmonics, "
                "frequency_bands (Hz), and the images before and after the "
                "filter as Signals (unfiltered_data, filtered_data)."
            ),
        ),
        StepArtifact(
            name="filtered_eit_signal",
            artifact_type="eit_pixel_signal",
            description="The same Signal as 'filtered_eit'.",
        ),
    ),
    record=True,
)
def mdn_filter(
    signal: Any,
    *,
    respiratory_rate_hz: float,
    heart_rate_hz: float,
    eit_sequence: Any,
    session: M3Session,
    noise_frequency_limit_hz: float = NOISE_FREQUENCY_LIMIT,
    notch_distance_hz: float = NOTCH_DISTANCE,
    order: int = ORDER,
    label: str = "filtered",
) -> dict[str, Any]:
    """Remove the heart signal from EIT images with the MDN filter.

    The filter is ``m3resp.eit.processing.mdn_filter``.

    Args:
        signal (Any): EIT images as a ``Signal`` (or an eitprocessing
            ``EITData``), time in seconds.
        respiratory_rate_hz (float): Breathing rate, in Hz.
        heart_rate_hz (float): Heart rate to remove, in Hz.
        eit_sequence (Any): Sequence that receives the filtered
            eitprocessing ``EITData``.
        session (M3Session): Session that receives the filtered Signal.
        noise_frequency_limit_hz (float): Low-pass edge in Hz. Defaults to
            220 per minute.
        notch_distance_hz (float): Half the width of each removed band, in
            Hz. Defaults to 10 per minute.
        order (int): Butterworth order of each filter, 1 to 10. Defaults to
            10.
        label (str): Label of the filtered data.

    Returns:
        dict[str, Any]: ``filtered_eit`` and ``filtered_eit_signal``, the
            same filtered ``Signal``, and ``filter_captures``, the in-between
            results of the filter.

    Raises:
        ValueError: If a rate is not a finite number above 0, if the
            breathing rate is not below the heart rate, or if a band edge is
            at or above half the frames per second.
    """

    result = session.eit_adapter.apply_mdn(
        signal,
        respiratory_rate_hz=respiratory_rate_hz,
        heart_rate_hz=heart_rate_hz,
        noise_frequency_limit_hz=noise_frequency_limit_hz,
        notch_distance_hz=notch_distance_hz,
        order=order,
        label=label,
        name=f"MDN-filtered EIT data ({label})",
        description="EIT data filtered with MDN heart-rate noise removal.",
    )
    filtered_eit_signal = result["filtered_eit"]
    add_to_collection(eit_sequence.eit_data, signal_to_eit_data(filtered_eit_signal))

    metadata = _upstream_metadata(
        source_package="m3resp",
        source_function="m3resp.eit.processing.mdn_filter",
        implementation="m3resp.eit.processing.mdn_filter",
        operation="eit.mdn_filter",
        parameters={
            "respiratory_rate_hz": respiratory_rate_hz,
            "heart_rate_hz": heart_rate_hz,
            "noise_frequency_limit_hz": noise_frequency_limit_hz,
            "notch_distance_hz": notch_distance_hz,
            "order": order,
            "label": label,
        },
    )
    filtered_eit_signal.metadata.update(metadata)
    session.signals.add(filtered_eit_signal)

    return {
        "filtered_eit": filtered_eit_signal,
        "filter_captures": result["filter_captures"],
        "filtered_eit_signal": filtered_eit_signal,
        "_provenance": metadata,
    }


@register_step(
    "eit.butterworth_filter",
    reads={"signal": None, "eit_sequence": "eit_sequence", "session": "session"},
    writes=("filtered_eit", "filter_captures", "filtered_eit_signal"),
    summary="Apply a Butterworth filter to EIT pixel data.",
    description="Apply a lowpass, highpass, bandpass or bandstop Butterworth filter to EIT pixel impedance (m3resp.processing.filters.butterworth_filter), as an alternative to eit.mdn_filter.",
    category="preprocessing",
    modality="eit",
    optional_packages=_EITPROCESSING,
    alternatives=("eit.mdn_filter",),
    session_writes=("session.signals",),
    input_artifacts=(
        StepArtifact(
            name="signal",
            artifact_type="eit_pixel_signal",
            description="EIT images (pixel impedance) to filter.",
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
            choices=get_args(ButterworthFilterType),
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
            maximum=MAX_EIT_BUTTERWORTH_ORDER,
            description="Butterworth filter order, 1 to 10.",
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
            description="Butterworth-filtered EIT images as a Signal.",
        ),
        StepArtifact(
            name="filter_captures",
            artifact_type="diagnostic_summary",
            description=(
                "In-between results: sample_frequency, the cutoffs "
                "(low_pass_frequency, high_pass_frequency or frequency_bands, "
                "in Hz), and the images before and after the filter as Signals "
                "(unfiltered_data, filtered_data)."
            ),
        ),
        StepArtifact(
            name="filtered_eit_signal",
            artifact_type="eit_pixel_signal",
            description="The same Signal as 'filtered_eit'.",
        ),
    ),
    record=True,
)
def butterworth_filter(
    signal: Any,
    *,
    eit_sequence: Any,
    session: M3Session,
    mode: ButterworthFilterType = "lowpass",
    lowpass_hz: float = 1.0,
    highpass_hz: float = 0.05,
    order: int = 4,
    label: str = "filtered",
) -> dict[str, Any]:
    """Apply a zero-phase Butterworth filter to EIT pixel signals.

    The filter is ``m3resp.processing.filters.butterworth_filter``, run on each
    pixel along time.

    Args:
        signal (Any): EIT images as a ``Signal`` (or an eitprocessing
            ``EITData``), with the frames per second in Hz and time along the
            first axis.
        eit_sequence (Any): Sequence receiving the filtered EIT data.
        session (M3Session): Session receiving the filtered Signal and provenance.
        mode (ButterworthFilterType): "lowpass", "highpass", "bandpass", or
            "bandstop". Defaults to "lowpass".
        lowpass_hz (float): Upper cutoff in Hz. Defaults to 1.0; used for lowpass,
            bandpass and bandstop modes.
        highpass_hz (float): Lower cutoff in Hz. Defaults to 0.05; used for
            highpass, bandpass and bandstop modes. Both band cutoffs must be
            positive, ordered and below half the sampling rate.
        order (int): Butterworth order, 1 to 10. Defaults to 4.
        label (str): Label for the filtered copy. Defaults to "filtered".

    Returns:
        dict[str, Any]: ``filtered_eit`` and ``filtered_eit_signal``, the same
            filtered ``Signal``, and ``filter_captures``, the filter
            diagnostics (the images before and after the filter as
            ``Signal``s in ``unfiltered_data`` and ``filtered_data``). Values
            keep their shape, units and time. Pixels with only missing
            samples stay missing.

    Raises:
        ValueError: If a pixel has gaps among otherwise present samples, if
            the order is outside 1 to 10, or if the other filter parameters
            are invalid.
    """

    # One edge for the single-sided modes, both edges (low, high) for the
    # two-sided ones.
    single_sided = {"lowpass": lowpass_hz, "highpass": highpass_hz}
    filtered_eit_signal, captures = butterworth_filter_images(
        as_signal(signal),
        mode=mode,
        cutoff_frequency=single_sided.get(mode, (highpass_hz, lowpass_hz)),
        order=order,
        label=label,
        operation="eit.butterworth_filter",
    )
    add_to_collection(eit_sequence.eit_data, signal_to_eit_data(filtered_eit_signal))

    metadata = _upstream_metadata(
        source_package="m3resp",
        source_function="m3resp.processing.filters.butterworth_filter",
        implementation="m3resp.processing.filters",
        operation="eit.butterworth_filter",
        parameters={
            "mode": mode,
            "lowpass_hz": lowpass_hz,
            "highpass_hz": highpass_hz,
            "order": order,
            "label": label,
        },
    )
    filtered_eit_signal.metadata.update(metadata)
    session.signals.add(filtered_eit_signal)

    return {
        "filtered_eit": filtered_eit_signal,
        "filter_captures": captures,
        "filtered_eit_signal": filtered_eit_signal,
        "_provenance": metadata,
    }
