"""Registered ECG-peak-detection pipeline step (for downstream ECG removal steps)."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.core.events import Event
from m3resp.data import ParameterResult
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import (
    _RESURFEMG,
    _SESSION_ARTIFACT,
    _record_step,
    _upstream_metadata,
)


def _select_ecg_source(
    session: M3Session,
    processed_emg: Any,
    *,
    ecg_channel: int | None,
    source: str,
) -> tuple[np.ndarray, float, str]:
    """Return `(array, sample_frequency, source_label)` for ECG detection.

    `ecg_channel` (a raw channel-major index) takes priority over `source`
    (a key into `processed_emg`, e.g. "raw_channel"/"filtered"/"envelope").
    """

    if ecg_channel is not None:
        recording = session.emg
        if recording is None or recording.raw is None:
            raise ValueError("emg.ecg_detect_peaks needs a loaded EMG recording.")
        raw = np.asarray(recording.raw)
        if not (0 <= ecg_channel < raw.shape[0]):
            raise ValueError(
                f"ecg_channel {ecg_channel!r} is out of range; the loaded "
                f"recording has channels 0..{raw.shape[0] - 1}."
            )
        raw_fs = (recording.metadata or {}).get("fs")
        if raw_fs is None:
            raise ValueError("emg.ecg_detect_peaks needs recording.metadata['fs'].")
        return raw[ecg_channel], float(raw_fs), f"raw_channel[{ecg_channel}]"

    if source not in processed_emg:
        available = sorted(
            key
            for key, value in processed_emg.items()
            if isinstance(value, np.ndarray) or hasattr(value, "__len__")
        )
        raise ValueError(
            f"emg.ecg_detect_peaks source {source!r} is not present in "
            f"processed_emg; available keys: {available}."
        )
    array = np.asarray(processed_emg[source], dtype=float)
    fs = float(processed_emg["fs"])
    return array, fs, source


@register_step(
    "emg.ecg_detect_peaks",
    reads={"session": "session", "processed_emg": "processed_emg"},
    writes=("ecg_peak_indices", "ecg_peak_events", "ecg_peak_count_result"),
    summary="Detect ECG peak sample indices in an EMG/ECG channel.",
    description="Detect ECG (R-wave-like) peaks in a raw channel or a processed_emg key, via ReSurfEMGAdapter.detect_ecg_peaks. Prerequisite for emg.ecg_gating/emg.ecg_wavelet_denoising.",
    category="preprocessing",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle, used when 'ecg_channel' is unset.",
        ),
    ),
    parameters=(
        StepParameter(
            name="ecg_channel",
            value_type="integer",
            required=False,
            default=None,
            minimum=0,
            description="Raw channel-major index carrying ECG. Takes priority over 'source' when set.",
        ),
        StepParameter(
            name="source",
            value_type="string",
            default="raw_channel",
            description="Key into processed_emg to detect ECG on, when 'ecg_channel' is unset.",
        ),
        StepParameter(
            name="peak_fraction",
            value_type="number",
            default=0.4,
            minimum=0,
            maximum=1,
            description="Detection threshold as a fraction of signal amplitude.",
        ),
        StepParameter(
            name="peak_width_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Minimum peak width. Defaults to the detector's own choice when unset.",
        ),
        StepParameter(
            name="peak_distance_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Minimum distance between peaks. Defaults to the detector's own choice when unset.",
        ),
        StepParameter(
            name="bandpass_filter",
            value_type="boolean",
            default=True,
            description="Bandpass-filter the source before peak detection.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ecg_peak_indices",
            artifact_type="index_array",
            description="Detected ECG peak sample indices.",
        ),
        StepArtifact(
            name="ecg_peak_events",
            artifact_type="event_list",
            description="Native Event per detected ECG peak.",
        ),
        StepArtifact(
            name="ecg_peak_count_result",
            artifact_type="parameter_result",
            description="Native ParameterResult: count of detected ECG peaks.",
        ),
    ),
)
def ecg_detect_peaks(
    session: M3Session,
    processed_emg: Any,
    *,
    ecg_channel: int | None = None,
    source: str = "raw_channel",
    peak_fraction: float = 0.4,
    peak_width_seconds: float | None = None,
    peak_distance_seconds: float | None = None,
    bandpass_filter: bool = True,
) -> dict[str, Any]:
    array, fs, source_label = _select_ecg_source(
        session, processed_emg, ecg_channel=ecg_channel, source=source
    )
    peak_width_samples = (
        max(1, int(peak_width_seconds * fs)) if peak_width_seconds is not None else None
    )
    peak_distance_samples = (
        max(1, int(peak_distance_seconds * fs))
        if peak_distance_seconds is not None
        else None
    )

    indices = session.emg_adapter.detect_ecg_peaks(
        array,
        sample_frequency=fs,
        peak_fraction=peak_fraction,
        peak_width_samples=peak_width_samples,
        peak_distance_samples=peak_distance_samples,
        bandpass_filter=bandpass_filter,
    )

    detection_parameters = {
        "source": source_label,
        "peak_fraction": peak_fraction,
        "requested_peak_width_seconds": peak_width_seconds,
        "requested_peak_distance_seconds": peak_distance_seconds,
        "effective_peak_width_samples": peak_width_samples,
        "effective_peak_distance_samples": peak_distance_samples,
        "bandpass_filter": bandpass_filter,
    }
    events = [
        Event(
            name="ecg_peak",
            modality="emg",
            time=float(index) / fs,
            sample_index=int(index),
            metadata=dict(detection_parameters),
        )
        for index in indices
    ]
    session.add_events("ecg_peaks", events)

    count_result = ParameterResult(
        name="ecg_peak_count",
        value=float(len(indices)),
        modality="emg",
        method="resurfemg.detect_ecg_peaks",
        metadata=detection_parameters,
    )

    _record_step(
        session,
        "emg.ecg_detect_peaks",
        metadata=_upstream_metadata(
            source_function="resurfemg.preprocessing.ecg_removal.detect_ecg_peaks",
            operation="emg.ecg_detect_peaks",
            parameters=detection_parameters,
        ),
    )
    return {
        "ecg_peak_indices": indices,
        "ecg_peak_events": events,
        "ecg_peak_count_result": count_result,
    }
