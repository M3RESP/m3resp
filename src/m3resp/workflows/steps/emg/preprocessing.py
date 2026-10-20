"""Registered EMG preprocessing steps: channel, notch filter, band-pass filter, envelope.

Each step reads a processed EMG bundle (a dictionary holding the chosen
channel's samples, the sampling rate ``fs`` in Hz and the traces made so far)
and returns an updated copy. The usual order is

    emg.select_channel -> emg.notch_filter -> emg.bandpass_filter
    -> ECG removal -> emg.envelope

with the notch filter optional and placed before or after the band-pass
filter. ``emg.preprocess`` runs channel selection and both filters in one
step. The returned bundle also becomes the session's current processed EMG,
which breath detection reads.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from m3resp.adapters.resurfemg_adapter._defaults import (
    default_low_pass_hz,
    notch_max_frequency_hz,
)
from m3resp.core.session import M3Session
from m3resp.data import Signal
from m3resp.data.signals import ProcessingState
from m3resp.processing.windows import ENVELOPE_METHODS
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import (
    _RESURFEMG,
    _SESSION_ARTIFACT,
    _processed_channel_label_and_unit,
    _require_positive_seconds,
    _store_processed_emg,
    _upstream_metadata,
    resolve_emg_source,
)

_PROCESSED_EMG_INPUT = StepArtifact(
    name="processed_emg",
    artifact_type="emg_processed_bundle",
    description="Processed EMG bundle from an earlier preprocessing step.",
)


def _filter_input(processed_emg: Any, step_name: str) -> tuple[str, np.ndarray]:
    """The trace a filter step works on: the filtered one, else the raw channel.

    Filters run before ECG removal, so a bundle that already holds an
    ECG-cleaned trace is refused.
    """

    if not isinstance(processed_emg, Mapping):
        raise TypeError(f"{step_name} needs a processed EMG bundle.")
    if processed_emg.get("ecg_cleaned") is not None:
        raise ValueError(
            f"{step_name} filters the EMG before ECG removal, and this "
            "processed EMG has already been cleaned of ECG. Move the filter "
            "step before the ECG-removal step."
        )
    key = "filtered" if processed_emg.get("filtered") is not None else "raw_channel"
    if processed_emg.get(key) is None:
        raise ValueError(
            f"{step_name}: the processed EMG holds no samples; start it with "
            "'emg.select_channel' or 'emg.preprocess'."
        )
    return key, np.asarray(processed_emg[key], dtype=float)


def _trace_signal(
    processed_emg: Mapping[str, Any],
    values: np.ndarray,
    *,
    suffix: str,
    processing_state: ProcessingState,
    method: str,
    source: str,
    metadata: dict[str, Any],
) -> Signal:
    """A `Signal` for one trace of the bundle, on a time axis starting at 0 s."""

    label, unit = _processed_channel_label_and_unit(processed_emg)
    fs = float(processed_emg["fs"])
    return Signal(
        values=values,
        time=np.arange(len(values), dtype=float) / fs,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_{suffix}",
        modality="emg",
        category="electrical_potential",
        channel=label,
        source=source,
        processing_state=processing_state,
        derived_from="processed",
        method=method,
        metadata=metadata,
    )


@register_step(
    "emg.select_channel",
    reads={"session": "session"},
    writes=("processed_emg",),
    summary="Choose the EMG channel to analyse.",
    description="Start a processed EMG bundle from one channel of the loaded EMG recording. The filter steps and the envelope step work on this bundle.",
    category="preprocessing",
    modality="emg",
    session_reads=("session.raw.emg",),
    session_writes=("session.processed.emg",),
    alternatives=("emg.preprocess",),
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="channel",
            value_type="integer",
            required=False,
            default=None,
            minimum=0,
            description=(
                "Zero-based channel index. Left unset, the channel is chosen "
                "from the channel labels, leaving out ECG/EKG channels."
            ),
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle holding the chosen channel ('raw_channel'), 'fs' and 'channel'.",
        ),
    ),
    record=True,
)
def select_channel(session: M3Session, *, channel: int | None = None) -> dict[str, Any]:
    """Start a processed EMG bundle from one channel of the loaded recording.

    Args:
        session: Session holding the loaded EMG recording.
        channel: Zero-based channel index; None chooses it from the labels.

    Returns:
        dict[str, Any]: ``processed_emg``, the new bundle.

    Raises:
        MissingModalityDataError: If no EMG recording is loaded.
        UnresolvedChannelError: If channel is None and the labels do not
            point to one EMG channel.
    """

    recording = session._require_raw("emg")
    processed_emg = session.emg_adapter.select_emg_channel(
        recording.data, channel=channel
    )
    _store_processed_emg(session, processed_emg)
    return {
        "processed_emg": processed_emg,
        "_provenance": _upstream_metadata(
            source_function="m3resp.adapters.resurfemg_adapter.select_emg_channel",
            operation="emg.select_channel",
            parameters={
                "requested_channel": channel,
                "channel": processed_emg["channel"],
            },
            source_package="m3resp",
            implementation="m3resp",
        ),
    }


@register_step(
    "emg.notch_filter",
    reads={"session": "session", "processed_emg": "processed_emg"},
    writes=("processed_emg_after_notch", "notch_filtered_signal"),
    summary="Remove a frequency and its harmonics from the EMG.",
    description="Notch-filter the EMG at a base frequency and its harmonics, for mains hum (e.g. 50 Hz) or the comb an EIT device adds at its frame rate. Works on the filtered trace, or on the raw channel when no filter has run yet.",
    category="preprocessing",
    modality="emg",
    session_writes=("session.processed.emg", "session.signals"),
    input_artifacts=(_SESSION_ARTIFACT, _PROCESSED_EMG_INPUT),
    parameters=(
        StepParameter(
            name="base_frequency",
            value_type="number",
            required=True,
            unit="Hz",
            minimum=0,
            description="Lowest frequency to remove, e.g. 50 for mains hum.",
        ),
        StepParameter(
            name="max_frequency",
            value_type="number",
            required=False,
            default=None,
            unit="Hz",
            minimum=0,
            description="Highest harmonic to remove. Left unset, harmonics up to the Nyquist frequency are removed. A value above the Nyquist frequency is lowered to it, with a warning.",
        ),
        StepParameter(
            name="quality_factor",
            value_type="number",
            required=False,
            default=30.0,
            minimum=0,
            description="Quality factor of each notch; higher is narrower.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="processed_emg_after_notch",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle with the notch-filtered trace as 'filtered'.",
        ),
        StepArtifact(
            name="notch_filtered_signal",
            artifact_type="signal",
            description="Native Signal wrapping the notch-filtered EMG.",
        ),
    ),
    record=True,
)
def notch_filter(
    session: M3Session,
    processed_emg: Any,
    *,
    base_frequency: float,
    max_frequency: float | None = None,
    quality_factor: float = 30.0,
) -> dict[str, Any]:
    """Notch-filter the EMG at a base frequency and its harmonics.

    Args:
        session: Session whose current processed EMG is updated.
        processed_emg: Bundle from an earlier preprocessing step.
        base_frequency: Lowest frequency to remove, in Hz.
        max_frequency: Highest harmonic to remove, in Hz; None uses Nyquist.
            A value above Nyquist is lowered to it, with a warning.
        quality_factor: Quality factor of each notch.

    Returns:
        dict[str, Any]: ``processed_emg_after_notch`` (the bundle with the
            result as ``"filtered"``) and ``notch_filtered_signal``.

    Raises:
        ValueError: If a frequency is not positive, or the bundle already
            holds an ECG-cleaned trace.
    """

    _require_positive_seconds("base_frequency", base_frequency)
    if max_frequency is not None:
        _require_positive_seconds("max_frequency", max_frequency)
    _require_positive_seconds("quality_factor", quality_factor)
    source, values = _filter_input(processed_emg, "emg.notch_filter")
    fs = float(processed_emg["fs"])
    max_frequency = notch_max_frequency_hz(fs, max_frequency)
    filtered = session.emg_adapter.notch_filter_emg(
        values,
        sample_frequency=fs,
        base_frequency=base_frequency,
        max_frequency=max_frequency,
        quality_factor=quality_factor,
    )
    settings = {
        "notch_base_frequency": base_frequency,
        "notch_max_frequency": max_frequency,
        "notch_quality_factor": quality_factor,
        # True when the band-pass filter has not run yet.
        "notch_before_bandpass": "high_pass_hz"
        not in (processed_emg.get("filter") or {}),
    }
    processed_after = {
        **processed_emg,
        "filtered": filtered,
        "envelope": None,
        "envelope_source": None,
        "filter": {**(processed_emg.get("filter") or {}), **settings},
    }
    _store_processed_emg(session, processed_after)
    signal = _trace_signal(
        processed_after,
        filtered,
        suffix="notch_filtered",
        processing_state="intermediate",
        method="m3resp.harmonic_notch_filter",
        source="m3resp",
        metadata={"source": source, **settings},
    )
    session.signals.add(signal)
    return {
        "processed_emg_after_notch": processed_after,
        "notch_filtered_signal": signal,
        "_provenance": _upstream_metadata(
            source_function="m3resp.processing.filters.harmonic_notch_filter",
            operation="emg.notch_filter",
            parameters={"source": source, **settings},
            source_package="m3resp",
            implementation="m3resp.processing.filters",
        ),
    }


@register_step(
    "emg.bandpass_filter",
    reads={"session": "session", "processed_emg": "processed_emg"},
    writes=("processed_emg_after_bandpass", "bandpass_filtered_signal"),
    summary="Band-pass filter the EMG.",
    description="Band-pass filter the EMG with ReSurfEMG's Butterworth filter (20-500 Hz by default, the range used in respiratory-sEMG literature). Works on the filtered trace, or on the raw channel when no filter has run yet.",
    category="preprocessing",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_writes=("session.processed.emg", "session.signals"),
    input_artifacts=(_SESSION_ARTIFACT, _PROCESSED_EMG_INPUT),
    parameters=(
        StepParameter(
            name="high_pass_hz",
            value_type="number",
            required=False,
            default=20.0,
            unit="Hz",
            minimum=0,
            description="Lower cutoff frequency.",
        ),
        StepParameter(
            name="low_pass_hz",
            value_type="number",
            required=False,
            default=None,
            unit="Hz",
            minimum=0,
            description="Upper cutoff frequency. Left unset: 500 Hz, or 95 % of the Nyquist frequency if that is lower.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="processed_emg_after_bandpass",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle with the band-passed trace as 'filtered'.",
        ),
        StepArtifact(
            name="bandpass_filtered_signal",
            artifact_type="signal",
            description="Native Signal wrapping the band-passed EMG.",
        ),
    ),
    record=True,
)
def bandpass_filter(
    session: M3Session,
    processed_emg: Any,
    *,
    high_pass_hz: float = 20.0,
    low_pass_hz: float | None = None,
) -> dict[str, Any]:
    """Band-pass filter the EMG.

    Args:
        session: Session whose current processed EMG is updated.
        processed_emg: Bundle from an earlier preprocessing step.
        high_pass_hz: Lower cutoff in Hz.
        low_pass_hz: Upper cutoff in Hz; None uses the smaller of 500 Hz and
            95 % of the Nyquist frequency.

    Returns:
        dict[str, Any]: ``processed_emg_after_bandpass`` (the bundle with the
            result as ``"filtered"``) and ``bandpass_filtered_signal``.

    Raises:
        ValueError: If a cutoff is not positive, the lower cutoff is not below
            the upper one, or the bundle already holds an ECG-cleaned trace.
        OptionalDependencyError: If ReSurfEMG is unavailable.
    """

    source, values = _filter_input(processed_emg, "emg.bandpass_filter")
    fs = float(processed_emg["fs"])
    effective_low_pass_hz = (
        default_low_pass_hz(fs) if low_pass_hz is None else float(low_pass_hz)
    )
    _require_positive_seconds("high_pass_hz", high_pass_hz)
    _require_positive_seconds("low_pass_hz", effective_low_pass_hz)
    if high_pass_hz >= effective_low_pass_hz:
        raise ValueError(
            f"emg.bandpass_filter: high_pass_hz ({high_pass_hz}) must be below "
            f"low_pass_hz ({effective_low_pass_hz})."
        )
    filtered = session.emg_adapter.bandpass_filter_emg(
        values,
        sample_frequency=fs,
        high_pass_hz=high_pass_hz,
        low_pass_hz=effective_low_pass_hz,
    )
    settings = {"high_pass_hz": high_pass_hz, "low_pass_hz": effective_low_pass_hz}
    processed_after = {
        **processed_emg,
        "filtered": filtered,
        "envelope": None,
        "envelope_source": None,
        "filter": {**(processed_emg.get("filter") or {}), **settings},
    }
    _store_processed_emg(session, processed_after)
    signal = _trace_signal(
        processed_after,
        filtered,
        suffix="bandpass_filtered",
        processing_state="intermediate",
        method="resurfemg.emg_bandpass_butter",
        source="resurfemg",
        metadata={"source": source, **settings},
    )
    session.signals.add(signal)
    return {
        "processed_emg_after_bandpass": processed_after,
        "bandpass_filtered_signal": signal,
        "_provenance": _upstream_metadata(
            source_function="resurfemg.preprocessing.filtering.emg_bandpass_butter",
            operation="emg.bandpass_filter",
            parameters={"source": source, **settings},
        ),
    }


@register_step(
    "emg.envelope",
    reads={"session": "session", "processed_emg": "processed_emg"},
    writes=("processed_emg_with_envelope", "envelope_signal"),
    summary="Compute the EMG envelope.",
    description="Compute the EMG envelope over a moving window (RMS, ARV or median of the absolute signal). Works on the ECG-cleaned trace when ECG removal has run, else on the filtered trace. Baselines, breath detection and per-breath measures read this envelope.",
    category="preprocessing",
    modality="emg",
    session_writes=("session.processed.emg", "session.signals"),
    input_artifacts=(_SESSION_ARTIFACT, _PROCESSED_EMG_INPUT),
    parameters=(
        StepParameter(
            name="window_seconds",
            value_type="number",
            required=False,
            default=0.25,
            unit="s",
            minimum=0,
            description="Window length, converted to at least one sample (rounded down).",
        ),
        StepParameter(
            name="method",
            value_type="choice",
            required=False,
            default="rms",
            choices=ENVELOPE_METHODS,
            description=(
                "rms (root mean square, the usual sEMG envelope), arv (average "
                "rectified value) or median (median of the absolute signal, "
                "which ignores short spikes such as heartbeat leftovers)."
            ),
        ),
        StepParameter(
            name="source",
            value_type="choice",
            required=False,
            default=None,
            choices=("ecg_cleaned", "filtered", "raw_channel"),
            description="Trace to compute the envelope from. Left unset: the ECG-cleaned trace when present, else the filtered one.",
            advanced=True,
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="processed_emg_with_envelope",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle with the result as 'envelope'.",
        ),
        StepArtifact(
            name="envelope_signal",
            artifact_type="signal",
            description="Native Signal wrapping the envelope.",
        ),
    ),
    record=True,
)
def envelope(
    session: M3Session,
    processed_emg: Any,
    *,
    window_seconds: float = 0.25,
    method: str = "rms",
    source: str | None = None,
) -> dict[str, Any]:
    """Compute the EMG envelope.

    Args:
        session: Session whose current processed EMG is updated.
        processed_emg: Bundle from a filter or ECG-removal step.
        window_seconds: Window length in seconds.
        method: ``"rms"``, ``"arv"`` or ``"median"``.
        source: Trace to use; None takes the ECG-cleaned trace when present,
            else the filtered one.

    Returns:
        dict[str, Any]: ``processed_emg_with_envelope`` (the bundle with the
            result as ``"envelope"``, in the units of the trace, one value per
            sample) and ``envelope_signal``.

    Raises:
        ValueError: If the window is not positive, the method is unknown, or
            the bundle has no such trace.
    """

    _require_positive_seconds("window_seconds", window_seconds)
    if method not in ENVELOPE_METHODS:
        raise ValueError(
            f"emg.envelope: method must be one of {ENVELOPE_METHODS}; got {method!r}."
        )
    source = resolve_emg_source(processed_emg, source, "emg.envelope")
    fs = float(processed_emg["fs"])
    values = np.asarray(processed_emg[source], dtype=float)
    envelope_values = session.emg_adapter.emg_envelope(
        values, sample_frequency=fs, window_seconds=window_seconds, method=method
    )
    settings = {
        "envelope_window_seconds": window_seconds,
        "envelope_method": method,
    }
    processed_after = {
        **processed_emg,
        "envelope": envelope_values,
        "envelope_source": source,
        "filter": {**(processed_emg.get("filter") or {}), **settings},
    }
    _store_processed_emg(session, processed_after)
    parameters = {
        "source": source,
        **settings,
        "envelope_window_samples": max(1, int(window_seconds * fs)),
    }
    signal = _trace_signal(
        processed_after,
        envelope_values,
        suffix="envelope",
        processing_state="processed",
        method=f"m3resp.rolling_envelope.{method}",
        source="m3resp",
        metadata=dict(parameters),
    )
    session.signals.add(signal)
    return {
        "processed_emg_with_envelope": processed_after,
        "envelope_signal": signal,
        "_provenance": _upstream_metadata(
            source_function="m3resp.processing.windows.rolling_envelope",
            operation="emg.envelope",
            parameters=parameters,
            source_package="m3resp",
            implementation="m3resp.processing.windows",
        ),
    }


_ENVELOPE_SETTINGS = ("envelope_window_seconds", "envelope_method", "compute_envelope")


@register_step(
    "emg.preprocess",
    reads={"session": "session"},
    writes=("processed_emg",),
    summary="Choose the EMG channel and filter it.",
    description=(
        "Run emg.select_channel, emg.notch_filter (when a notch frequency is "
        "set) and emg.bandpass_filter in one step. The envelope is computed "
        "afterwards by emg.envelope, after ECG removal."
    ),
    category="preprocessing",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_writes=("session.processed.emg", "session.signals"),
    alternatives=("emg.select_channel",),
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="channel",
            value_type="integer",
            required=False,
            default=None,
            minimum=0,
            description=(
                "Zero-based channel index. Left unset, the channel is chosen "
                "from the channel labels, leaving out ECG/EKG channels."
            ),
        ),
        StepParameter(
            name="high_pass_hz",
            value_type="number",
            required=False,
            default=20.0,
            unit="Hz",
            minimum=0,
            description="Band-pass lower cutoff frequency.",
        ),
        StepParameter(
            name="low_pass_hz",
            value_type="number",
            required=False,
            default=None,
            unit="Hz",
            minimum=0,
            description="Band-pass upper cutoff frequency. Left unset: 500 Hz, or 95 % of the Nyquist frequency if that is lower.",
        ),
        StepParameter(
            name="notch_base_frequency",
            value_type="number",
            required=False,
            default=None,
            unit="Hz",
            minimum=0,
            description="Lowest frequency the notch filter removes, with its harmonics (e.g. 50 for mains hum). Left unset, no notch filter is applied.",
        ),
        StepParameter(
            name="notch_max_frequency",
            value_type="number",
            required=False,
            default=None,
            unit="Hz",
            minimum=0,
            description="Highest harmonic the notch filter removes. Left unset: the Nyquist frequency. A value above the Nyquist frequency is lowered to it, with a warning.",
            advanced=True,
        ),
        StepParameter(
            name="notch_quality_factor",
            value_type="number",
            required=False,
            default=30.0,
            minimum=0,
            description="Quality factor of each notch; higher is narrower.",
            advanced=True,
        ),
        StepParameter(
            name="notch_before_bandpass",
            value_type="boolean",
            required=False,
            default=False,
            description="Apply the notch filter before the band-pass filter; otherwise after it.",
            advanced=True,
        ),
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
            description="Mapping of 'raw_channel'/'filtered'/'fs'/'channel'/'filter'/'metadata'; 'envelope' is None until emg.envelope runs.",
        ),
    ),
)
def preprocess(
    session: M3Session,
    *,
    channel: int | None = None,
    high_pass_hz: float = 20.0,
    low_pass_hz: float | None = None,
    notch_base_frequency: float | None = None,
    notch_max_frequency: float | None = None,
    notch_quality_factor: float = 30.0,
    notch_before_bandpass: bool = False,
    variant: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Choose the EMG channel and filter it, through `M3Session.preprocess_emg`.

    Args:
        session: Session holding the loaded EMG recording.
        channel: Zero-based channel index; None chooses it from the labels.
        high_pass_hz: Band-pass lower cutoff in Hz.
        low_pass_hz: Band-pass upper cutoff in Hz; None uses the smaller of
            500 Hz and 95 % of the Nyquist frequency.
        notch_base_frequency: Notch frequency in Hz (with its harmonics);
            None applies no notch filter.
        notch_max_frequency: Highest harmonic notched, in Hz; None uses Nyquist.
        notch_quality_factor: Quality factor of each notch.
        notch_before_bandpass: Notch before the band-pass filter when True.
        variant: Name to store the result under; None is the default slot.
        **kwargs: Other keyword arguments for `M3Session.preprocess_emg`.

    Returns:
        dict[str, Any]: ``processed_emg``, the filtered bundle with
            ``"envelope"`` None.

    Raises:
        ValueError: If envelope settings are given; they belong to
            ``emg.envelope``.
    """

    envelope_settings = sorted(set(_ENVELOPE_SETTINGS) & kwargs.keys())
    if envelope_settings:
        raise ValueError(
            f"emg.preprocess: {', '.join(envelope_settings)} belong to the "
            "'emg.envelope' step, which computes the envelope after ECG "
            "removal. Set them there."
        )
    processed_emg = session.preprocess_emg(
        variant=variant,
        channel=channel,
        high_pass_hz=high_pass_hz,
        low_pass_hz=low_pass_hz,
        notch_base_frequency=notch_base_frequency,
        notch_max_frequency=notch_max_frequency,
        notch_quality_factor=notch_quality_factor,
        notch_before_bandpass=notch_before_bandpass,
        compute_envelope=False,
        **kwargs,
    )
    return {"processed_emg": processed_emg}
