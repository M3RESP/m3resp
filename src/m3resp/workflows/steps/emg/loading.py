"""Registered EMG loading/preprocessing pipeline steps."""

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
    _record_step,
    _upstream_metadata,
)


@register_step(
    "emg.load",
    reads={"session": "session"},
    writes=("emg_recording", "raw_emg_signals"),
    summary="Load an EMG recording into the session.",
    description="Load an EMG recording file through ReSurfEMGAdapter and expose its raw channels as native Signals.",
    category="loading",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_writes=("session.raw.emg", "session.signals"),
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
        StepParameter(
            name="file",
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
    output_artifacts=(
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
    ),
)
def load(
    session: M3Session,
    *,
    file: str,
    loader_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if loader_options is not None and not isinstance(loader_options, Mapping):
        raise TypeError(
            "emg.load 'loader_options' must be a mapping of keyword arguments "
            f"for M3Session.load_emg(), got {type(loader_options).__name__}."
        )

    session.load_emg(file, verbose=False, **dict(loader_options or {}))
    recording = session.emg
    assert recording is not None

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

    _record_step(
        session,
        "emg.load",
        metadata=_upstream_metadata(
            source_function="resurfemg.data_connector.converter_functions.load_file",
            operation="emg.load",
            parameters={"loader_options": dict(loader_options or {})},
        ),
    )
    return {"emg_recording": recording, "raw_emg_signals": raw_emg_signals}


@register_step(
    "emg.preprocess",
    reads={"session": "session"},
    writes=("processed_emg",),
    summary="Filter EMG and compute its envelope.",
    description="Filter the loaded EMG recording and compute its envelope via ReSurfEMGAdapter.preprocess. Accepts arbitrary adapter keyword arguments beyond 'variant'.",
    category="preprocessing",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_writes=("session.processed.emg", "session.signals"),
    input_artifacts=(_SESSION_ARTIFACT,),
    parameters=(
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
            description="Mapping of 'filtered'/'envelope'/'fs'/'channel'/'metadata'.",
        ),
    ),
)
def preprocess(session: M3Session, **kwargs: Any) -> dict[str, Any]:
    return {"processed_emg": session.preprocess_emg(**kwargs)}


@register_step(
    "emg.detect_breaths",
    reads={"session": "session"},
    writes=("emg_breath_events",),
    summary="Detect EMG breaths from the envelope.",
    description="Detect EMG breaths from the processed envelope via ReSurfEMGAdapter.detect_breaths. Accepts arbitrary adapter keyword arguments beyond 'variant'.",
    category="detection",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_reads=("session.processed.emg",),
    session_writes=("session.events.emg_breaths",),
    input_artifacts=(_SESSION_ARTIFACT,),
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
def detect_breaths(session: M3Session, **kwargs: Any) -> dict[str, Any]:
    events = session.detect_emg_breaths(**kwargs)
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
