"""Registered EMG per-breath feature pipeline steps."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.processing.metrics import (
    amplitude_at_peaks,
    respiratory_rate_from_indices,
    window_integral,
)
from m3resp.processing.metrics import (
    area_under_baseline as _area_under_baseline,
)
from m3resp.processing.metrics import (
    pseudo_slope as _pseudo_slope,
)
from m3resp.processing.metrics import (
    time_to_peak as _time_to_peak,
)
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import _mask_invalid


@register_step(
    "emg.time_to_peak",
    reads={
        "processed_emg": "processed_emg",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "start_end_validity": "start_end_validity",
    },
    writes=("time_to_peak",),
    summary="Compute EMG breath time-to-peak.",
    description="Compute the time from breath onset to peak for each detected breath.",
    category="features",
    modality="emg",
    parameters_reviewed=True,
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
        StepArtifact(
            name="start_indices",
            artifact_type="index_array",
            description="Breath onset indices.",
        ),
        StepArtifact(
            name="end_indices",
            artifact_type="index_array",
            description="Breath offset indices.",
        ),
        StepArtifact(
            name="start_end_validity",
            artifact_type="boolean_array",
            description="Per-breath validity from 'emg.onoffpeak_baseline_crossing'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="time_to_peak",
            artifact_type="array",
            unit="s",
            description="Time-to-peak per breath (NaN where the onset/offset window is invalid).",
        ),
    ),
)
def time_to_peak(
    processed_emg: Any,
    start_indices: Any,
    end_indices: Any,
    *,
    start_end_validity: Any = None,
) -> dict[str, Any]:

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    absolute_times, percent_times = _time_to_peak(envelope, start_indices, end_indices)
    if start_end_validity is None:
        return {"time_to_peak": (absolute_times, percent_times)}
    return {
        "time_to_peak": (
            _mask_invalid(absolute_times, start_end_validity),
            _mask_invalid(percent_times, start_end_validity),
        )
    }


@register_step(
    "emg.pseudo_slope",
    reads={
        "processed_emg": "processed_emg",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "start_end_validity": "start_end_validity",
    },
    writes=("pseudo_slope",),
    summary="Compute EMG breath pseudo-slope.",
    description="Compute the pseudo-slope (rise rate) of the envelope for each detected breath.",
    category="features",
    modality="emg",
    parameters_reviewed=True,
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope.",
        ),
        StepArtifact(
            name="start_indices",
            artifact_type="index_array",
            description="Breath onset indices.",
        ),
        StepArtifact(
            name="end_indices",
            artifact_type="index_array",
            description="Breath offset indices.",
        ),
        StepArtifact(
            name="start_end_validity",
            artifact_type="boolean_array",
            description="Per-breath validity from 'emg.onoffpeak_baseline_crossing'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pseudo_slope",
            artifact_type="array",
            description="Pseudo-slope per breath (NaN where the onset/offset window is invalid).",
        ),
    ),
)
def pseudo_slope(
    processed_emg: Any,
    start_indices: Any,
    end_indices: Any,
    *,
    start_end_validity: Any = None,
) -> dict[str, Any]:

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    result = _pseudo_slope(envelope, start_indices, end_indices)
    if start_end_validity is None:
        return {"pseudo_slope": result}
    return {"pseudo_slope": _mask_invalid(result, start_end_validity)}


@register_step(
    "emg.amplitude",
    reads={
        "processed_emg": "processed_emg",
        "peak_indices": "peak_indices",
        "baseline": "baseline",
    },
    writes=("amplitude",),
    summary="Compute EMG breath amplitude above baseline.",
    description="Compute the envelope amplitude above baseline at each breath peak.",
    category="features",
    modality="emg",
    parameters_reviewed=True,
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices.",
        ),
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            description="Baseline from 'emg.moving_baseline' or 'emg.slopesum_baseline'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="amplitude",
            artifact_type="array",
            description="Amplitude above baseline per breath.",
        ),
    ),
)
def amplitude(processed_emg: Any, peak_indices: Any, baseline: Any) -> dict[str, Any]:

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    return {"amplitude": amplitude_at_peaks(envelope, peak_indices, baseline)}


@register_step(
    "emg.time_product",
    reads={
        "processed_emg": "processed_emg",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "baseline": "baseline",
        "start_end_validity": "start_end_validity",
    },
    writes=("time_product",),
    summary="Compute EMG breath time-product (area above baseline).",
    description="Integrate the envelope above baseline over each breath's onset/offset window.",
    category="features",
    modality="emg",
    parameters_reviewed=True,
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
        StepArtifact(
            name="start_indices",
            artifact_type="index_array",
            description="Breath onset indices.",
        ),
        StepArtifact(
            name="end_indices",
            artifact_type="index_array",
            description="Breath offset indices.",
        ),
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            description="Baseline from 'emg.moving_baseline' or 'emg.slopesum_baseline'.",
        ),
        StepArtifact(
            name="start_end_validity",
            artifact_type="boolean_array",
            description="Per-breath validity from 'emg.onoffpeak_baseline_crossing'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="time_product",
            artifact_type="array",
            description="Time-product per breath (NaN where the onset/offset window is invalid).",
        ),
    ),
)
def time_product(
    processed_emg: Any,
    start_indices: Any,
    end_indices: Any,
    baseline: Any,
    *,
    start_end_validity: Any = None,
) -> dict[str, Any]:

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    result = window_integral(envelope, fs, start_indices, end_indices, baseline)
    if start_end_validity is None:
        return {"time_product": result}
    return {"time_product": _mask_invalid(result, start_end_validity)}


@register_step(
    "emg.area_under_baseline",
    reads={
        "processed_emg": "processed_emg",
        "peak_indices": "peak_indices",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "baseline": "baseline",
        "start_end_validity": "start_end_validity",
    },
    writes=("area_under_baseline",),
    summary="Compute EMG area under baseline around each breath peak.",
    description="Compute the area of the envelope that dips under baseline within a window around each breath peak.",
    category="features",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices.",
        ),
        StepArtifact(
            name="start_indices",
            artifact_type="index_array",
            description="Breath onset indices.",
        ),
        StepArtifact(
            name="end_indices",
            artifact_type="index_array",
            description="Breath offset indices.",
        ),
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            description="Baseline from 'emg.moving_baseline' or 'emg.slopesum_baseline'.",
        ),
        StepArtifact(
            name="start_end_validity",
            artifact_type="boolean_array",
            description="Per-breath validity from 'emg.onoffpeak_baseline_crossing'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="window_seconds",
            value_type="number",
            default=5.0,
            unit="s",
            minimum=0,
            description="Window around each peak searched for under-baseline area.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="area_under_baseline",
            artifact_type="array",
            description="Area-under-baseline result per breath, and supporting arrays (NaN where the onset/offset window is invalid).",
        ),
    ),
)
def area_under_baseline(
    processed_emg: Any,
    peak_indices: Any,
    start_indices: Any,
    end_indices: Any,
    baseline: Any,
    *,
    window_seconds: float = 5.0,
    start_end_validity: Any = None,
) -> dict[str, Any]:

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    window_samples = max(1, int(window_seconds * fs))
    areas, references = _area_under_baseline(
        envelope,
        fs,
        peak_indices,
        start_indices,
        end_indices,
        window_samples,
        baseline,
    )
    if start_end_validity is None:
        return {"area_under_baseline": (areas, references)}
    return {
        "area_under_baseline": (
            _mask_invalid(areas, start_end_validity),
            _mask_invalid(references, start_end_validity),
        )
    }


@register_step(
    "emg.respiratory_rate",
    reads={"peak_indices": "peak_indices", "processed_emg": "processed_emg"},
    writes=("respiratory_rate",),
    summary="Compute respiratory rate from detected EMG breath peaks.",
    description="Compute respiratory rate from the detected EMG breath peak indices.",
    category="parameters",
    modality="emg",
    parameters_reviewed=True,
    input_artifacts=(
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices.",
        ),
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying 'fs'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="respiratory_rate",
            artifact_type="scalar_metric",
            unit="breaths/min",
            description="EMG-derived respiratory rate.",
        ),
    ),
)
def respiratory_rate(peak_indices: Any, processed_emg: Any) -> dict[str, Any]:
    fs = float(processed_emg["fs"])
    return {"respiratory_rate": respiratory_rate_from_indices(peak_indices, fs)}
