"""Registered EMG loading/preprocessing workflow steps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from m3resp.adapters.resurfemg_adapter import peak_indices_from_events
from m3resp.core.session import M3Session
from m3resp.data import Signal
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import (
    _RESURFEMG,
    _SESSION_ARTIFACT,
    _upstream_metadata,
)

_LOAD_WRITES = ("emg_recording", "raw_emg_signals")

_LOAD_OUTPUTS = (
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
)


@register_step(
    "emg.load",
    reads={"session": "session"},
    writes=_LOAD_WRITES,
    summary="Load an EMG recording into the session.",
    description="Load an EMG recording file through ReSurfEMGAdapter and expose its raw channels as native Signals.",
    category="loading",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_writes=(
        "session.raw.emg",
        "session.signals",
        # A new recording is not synchronized: its start time is cleared.
        "session.start_times",
        "session.sync_methods",
    ),
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="file_path",
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
    output_artifacts=_LOAD_OUTPUTS,
    record=True,
)
def load(
    session: M3Session,
    *,
    file_path: str,
    loader_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if loader_options is not None and not isinstance(loader_options, Mapping):
        raise TypeError(
            "emg.load 'loader_options' must be a mapping of keyword arguments "
            f"for M3Session.load_emg(), got {type(loader_options).__name__}."
        )

    session.load_emg(file_path, verbose=False, **dict(loader_options or {}))
    return {
        **_loaded_recording_outputs(session),
        "_provenance": _upstream_metadata(
            source_function="resurfemg.data_connector.converter_functions.load_file",
            operation="emg.load",
            parameters={"loader_options": dict(loader_options or {})},
        ),
    }


def _loaded_recording_outputs(
    session: M3Session, *, replace_raw_signals: bool = False
) -> dict[str, Any]:
    """``emg_recording`` and ``raw_emg_signals`` from the session's recording.

    Adds one raw `Signal` per channel to `session.signals`. With
    `replace_raw_signals`, the raw EMG signals already in `session.signals`
    are removed first; the slicing step uses this so only the cut signals
    remain.
    """

    recording = session.emg
    assert recording is not None

    if replace_raw_signals:
        session.signals.items[:] = [
            existing
            for existing in session.signals.items
            if not (existing.modality == "emg" and existing.processing_state == "raw")
        ]

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
                category="electrical_potential",
                channel=label,
                source=str(recording.path),
                processing_state="raw",
                metadata={"channel_index": index, "file_metadata": metadata},
            )
            session.signals.add(signal)
            raw_emg_signals.append(signal)

    return {"emg_recording": recording, "raw_emg_signals": raw_emg_signals}


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
            description="Highest harmonic the notch filter removes. Left unset: the Nyquist frequency.",
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


@register_step(
    "emg.detect_breaths",
    reads={"session": "session", "baseline": "baseline"},
    optional_reads=("baseline",),
    writes=("emg_breath_events",),
    summary="Detect EMG breaths from the envelope.",
    description="Detect EMG breaths from the processed envelope via ReSurfEMGAdapter.detect_breaths. Accepts arbitrary adapter keyword arguments beyond 'variant'.",
    category="detection",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_reads=("session.processed.emg",),
    session_writes=("session.events.emg_breaths",),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            default_context_key="baseline",
            description="EMG baseline from 'emg.moving_baseline' or 'emg.slopesum_baseline'. Detection thresholds are set against the envelope above this level; without it they are set against zero.",
        ),
    ),
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
def detect_breaths(
    session: M3Session, baseline: Any = None, **kwargs: Any
) -> dict[str, Any]:
    events = session.detect_emg_breaths(baseline=baseline, **kwargs)
    return {"emg_breath_events": events}


@register_step(
    "emg.peak_indices",
    reads={"events": "emg_breath_events", "processed_emg": "processed_emg"},
    writes=("peak_indices",),
    summary="Derive EMG breath peak sample indices from detected breath events.",
    description="Convert detected EMG breath events into peak sample indices into the processed envelope.",
    category="detection",
    modality="emg",
    parameters_reviewed=True,
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
