"""Registered on/off-peak interval pipeline steps."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, QualityFlag
from m3resp.processing.intervals import (
    onoff_from_baseline_crossings,
    onoff_from_slope,
)
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import (
    _RESURFEMG,
    _SESSION_ARTIFACT,
    _record_step,
    _upstream_metadata,
)


@register_step(
    "emg.interpeak_dist",
    reads={
        "session": "session",
        "ecg_peak_indices": "ecg_peak_indices",
        "peak_indices": "peak_indices",
        "processed_emg": "processed_emg",
    },
    writes=("interpeak_dist", "interpeak_dist_result", "interpeak_dist_flag"),
    summary="Check the ECG-to-EMG median interpeak distance ratio (Warnaar et al. 2024).",
    description="Check that the median EMG-to-ECG interpeak distance ratio stays under a threshold, as a proxy for adequate ECG removal (Warnaar et al. 2024).",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_writes=("session.quality", "session.parameter_results"),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="ecg_peak_indices",
            artifact_type="index_array",
            description="ECG peak indices from 'emg.ecg_detect_peaks'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="EMG breath peak indices.",
        ),
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying 'fs'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="threshold",
            value_type="number",
            default=1.1,
            minimum=0,
            description="Maximum acceptable EMG/ECG median interpeak distance ratio.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="interpeak_dist",
            artifact_type="boolean_metric",
            description="Whether the ratio passes the threshold.",
        ),
        StepArtifact(
            name="interpeak_dist_result",
            artifact_type="parameter_result_list",
            unit="s",
            description="Native ParameterResults: ECG median distance, EMG median distance, and their ratio.",
        ),
        StepArtifact(
            name="interpeak_dist_flag",
            artifact_type="quality_flag",
            description="Native QualityFlag for the overall check.",
        ),
    ),
)
def interpeak_dist(
    session: M3Session,
    ecg_peak_indices: Any,
    peak_indices: Any,
    processed_emg: Any,
    *,
    threshold: float = 1.1,
) -> dict[str, Any]:
    # Upstream compares distances in raw samples, which is only meaningful
    # if both peak sets share a time base - both are indices into the same
    # processed_emg channel here, so that holds without conversion.
    fs = float(processed_emg["fs"])
    valid = session.emg_adapter.interpeak_distance(
        ecg_peak_indices, peak_indices, threshold=threshold
    )

    ecg_peaks = np.asarray(ecg_peak_indices, dtype=int)
    emg_peaks = np.asarray(peak_indices, dtype=int)
    ecg_median_samples = float(np.median(np.diff(ecg_peaks)))
    emg_median_samples = float(np.median(np.diff(emg_peaks)))
    # Degenerate peak sets (e.g. duplicate indices) give a zero median
    # interval; match upstream's own behavior (a RuntimeWarning plus an inf/
    # nan ratio, not a raised error) via NumPy division instead of Python's
    # float division, which would raise ZeroDivisionError here.
    ratio = float(np.divide(emg_median_samples, ecg_median_samples))

    shared_metadata = {
        "threshold": threshold,
        "time_base": "shared (both peak sets index the same EMG channel)",
        "sample_frequency": fs,
    }
    results = [
        ParameterResult(
            name="interpeak_dist_ecg_median",
            value=ecg_median_samples / fs,
            modality="emg",
            unit="s",
            method="resurfemg.interpeak_dist",
            metadata=dict(shared_metadata),
        ),
        ParameterResult(
            name="interpeak_dist_emg_median",
            value=emg_median_samples / fs,
            modality="emg",
            unit="s",
            method="resurfemg.interpeak_dist",
            metadata=dict(shared_metadata),
        ),
        ParameterResult(
            name="interpeak_dist_ratio",
            value=ratio,
            modality="emg",
            method="resurfemg.interpeak_dist",
            metadata=dict(shared_metadata),
        ),
    ]
    flag = QualityFlag(
        name="interpeak_dist",
        passed=bool(valid),
        severity="info",
        modality="emg",
        value=ratio,
        threshold=threshold,
        metadata=dict(shared_metadata),
    )

    for result in results:
        session.parameter_results.add(result)
    session.quality.add(flag)

    _record_step(
        session,
        "emg.interpeak_dist",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.interpeak_dist",
            operation="emg.interpeak_dist",
            parameters={"threshold": threshold},
        ),
    )
    return {
        "interpeak_dist": valid,
        "interpeak_dist_result": results,
        "interpeak_dist_flag": flag,
    }


@register_step(
    "emg.onoffpeak_baseline_crossing",
    reads={
        "processed_emg": "processed_emg",
        "baseline": "baseline",
        "peak_indices": "peak_indices",
    },
    writes=("start_indices", "end_indices", "start_end_validity"),
    summary="Find EMG breath on/offset indices by baseline crossing.",
    description="Find each breath's onset/offset sample indices where the envelope crosses the baseline.",
    category="detection",
    modality="emg",
    alternatives=("emg.onoffpeak_slope_extrapolation",),
    parameters_reviewed=True,
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope.",
        ),
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            description="Baseline from 'emg.moving_baseline' or 'emg.slopesum_baseline'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices from 'emg.peak_indices'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="start_indices",
            artifact_type="index_array",
            description="Breath onset sample indices.",
        ),
        StepArtifact(
            name="end_indices",
            artifact_type="index_array",
            description="Breath offset sample indices.",
        ),
        StepArtifact(
            name="start_end_validity",
            artifact_type="boolean_array",
            description="Whether each breath's onset/offset window is valid (found, non-overlapping).",
        ),
    ),
)
def onoffpeak_baseline_crossing(
    processed_emg: Any, baseline: Any, peak_indices: Any
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    start_indices, end_indices, _valid_starts, _valid_ends, valid_peaks = (
        onoff_from_baseline_crossings(envelope, baseline, peak_indices)
    )
    return {
        "start_indices": start_indices,
        "end_indices": end_indices,
        "start_end_validity": np.asarray(valid_peaks, dtype=bool),
    }


@register_step(
    "emg.onoffpeak_slope_extrapolation",
    reads={"processed_emg": "processed_emg", "peak_indices": "peak_indices"},
    writes=("onoffpeak_slope_result",),
    summary="Find EMG breath on/offset indices by slope extrapolation.",
    description="Find each breath's onset/offset sample indices by extrapolating the steepest envelope slope near each peak.",
    category="detection",
    modality="emg",
    alternatives=("emg.onoffpeak_baseline_crossing",),
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices from 'emg.peak_indices'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="slope_window_seconds",
            value_type="number",
            default=0.5,
            unit="s",
            minimum=0,
            description="Window around each peak used to estimate the slope.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="onoffpeak_slope_result",
            artifact_type="onoff_result",
            description="Start/end indices plus slope-extrapolation diagnostics.",
        ),
    ),
)
def onoffpeak_slope_extrapolation(
    processed_emg: Any, peak_indices: Any, *, slope_window_seconds: float = 0.5
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    slope_window_samples = max(1, int(slope_window_seconds * fs))
    result = onoff_from_slope(
        envelope,
        sample_frequency=fs,
        peak_indices=peak_indices,
        slope_window=slope_window_samples,
    )
    return {"onoffpeak_slope_result": result}
