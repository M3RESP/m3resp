"""Registered EMG baseline-estimation pipeline steps."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.data import Signal
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import (
    _RESURFEMG,
    _SESSION_ARTIFACT,
    _processed_channel_label_and_unit,
    _record_step,
    _require_percentile,
    _require_positive_seconds,
    _upstream_metadata,
)


@register_step(
    "emg.moving_baseline",
    reads={"session": "session", "processed_emg": "processed_emg"},
    writes=("baseline", "baseline_signal"),
    summary="Compute a moving-percentile EMG baseline.",
    description="Compute a moving-percentile baseline of the EMG envelope via ReSurfEMGAdapter.moving_baseline.",
    category="baseline",
    modality="emg",
    optional_packages=_RESURFEMG,
    alternatives=("emg.slopesum_baseline",),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="window_seconds",
            value_type="number",
            default=30.0,
            unit="s",
            minimum=0,
            description="Moving-percentile window length.",
        ),
        StepParameter(
            name="step_seconds",
            value_type="number",
            default=1.0,
            unit="s",
            minimum=0,
            description="Step between successive baseline windows.",
        ),
        StepParameter(
            name="percentile",
            value_type="number",
            default=33.0,
            minimum=0,
            maximum=100,
            description="Percentile of the envelope computed within each window.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            unit=None,
            description="Moving-percentile baseline array, one value per envelope sample.",
        ),
        StepArtifact(
            name="baseline_signal",
            artifact_type="signal",
            description="Native Signal wrapping the baseline.",
        ),
    ),
)
def moving_baseline(
    session: M3Session,
    processed_emg: Any,
    *,
    window_seconds: float = 30.0,
    step_seconds: float = 1.0,
    percentile: float = 33.0,
) -> dict[str, Any]:
    _require_positive_seconds("window_seconds", window_seconds)
    _require_positive_seconds("step_seconds", step_seconds)
    _require_percentile("percentile", percentile)

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    window_samples = max(1, int(window_seconds * fs))
    step_samples = max(1, int(step_seconds * fs))
    baseline = session.emg_adapter.moving_baseline(
        envelope,
        window_samples=window_samples,
        step_samples=step_samples,
        percentile=percentile,
    )

    label, unit = _processed_channel_label_and_unit(processed_emg)
    time = np.arange(len(envelope), dtype=float) / fs
    baseline_signal = Signal(
        values=baseline,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_moving_baseline",
        modality="emg",
        channel=label,
        source="resurfemg",
        processing_state="derived",
        derived_from="processed",
        method="resurfemg.moving_baseline",
        metadata={
            "requested_window_seconds": window_seconds,
            "requested_step_seconds": step_seconds,
            "effective_window_samples": window_samples,
            "effective_step_samples": step_samples,
            "percentile": percentile,
        },
    )
    session.signals.add(baseline_signal)
    _record_step(
        session,
        "emg.moving_baseline",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.baseline.moving_baseline",
            operation="emg.moving_baseline",
            parameters={
                "window_seconds": window_seconds,
                "step_seconds": step_seconds,
                "percentile": percentile,
                "effective_window_samples": window_samples,
                "effective_step_samples": step_samples,
            },
        ),
    )
    return {"baseline": baseline, "baseline_signal": baseline_signal}


@register_step(
    "emg.slopesum_baseline",
    reads={"session": "session", "processed_emg": "processed_emg"},
    writes=(
        "baseline",
        "slopesum_baseline_detail",
        "baseline_signal",
        "slopesum_baseline_native_detail",
        "baseline_running_mean_signal",
        "baseline_running_std_signal",
    ),
    summary="Compute a slope-sum EMG baseline.",
    description="Compute a slope-sum baseline of the EMG envelope via ReSurfEMGAdapter.slopesum_baseline.",
    category="baseline",
    modality="emg",
    optional_packages=_RESURFEMG,
    alternatives=("emg.moving_baseline",),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="window_seconds",
            value_type="number",
            default=30.0,
            unit="s",
            minimum=0,
            description="Moving-percentile window length.",
        ),
        StepParameter(
            name="step_seconds",
            value_type="number",
            default=1.0,
            unit="s",
            minimum=0,
            description="Step between successive baseline windows.",
        ),
        StepParameter(
            name="percentile",
            value_type="number",
            default=33.0,
            minimum=0,
            maximum=100,
            description="Percentile used for the primary baseline.",
        ),
        StepParameter(
            name="augmented_percentile",
            value_type="number",
            default=25.0,
            minimum=0,
            maximum=100,
            description="Percentile used for the augmented (running mean/std) baseline.",
        ),
        StepParameter(
            name="moving_average_seconds",
            value_type="number",
            default=0.5,
            unit="s",
            minimum=0,
            description="Smoothing window for the running mean/std.",
        ),
        StepParameter(
            name="percentile_window_seconds",
            value_type="number",
            default=1.0,
            unit="s",
            minimum=0,
            description="Window used when recomputing the percentile series.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="baseline",
            artifact_type="signal_array",
            description="Slope-sum baseline array, one value per envelope sample.",
        ),
        StepArtifact(
            name="slopesum_baseline_detail",
            artifact_type="diagnostic_summary",
            description="Running mean/std plus the intermediate slope-sum series.",
            public=False,
            compatibility_only=True,
        ),
        StepArtifact(
            name="baseline_signal",
            artifact_type="signal",
            description="Native Signal wrapping the baseline.",
        ),
        StepArtifact(
            name="slopesum_baseline_native_detail",
            artifact_type="mapping",
            description="Native running mean/std, without the raw upstream series.",
        ),
        StepArtifact(
            name="baseline_running_mean_signal",
            artifact_type="signal",
            description="Native Signal wrapping the running mean.",
        ),
        StepArtifact(
            name="baseline_running_std_signal",
            artifact_type="signal",
            description="Native Signal wrapping the running std.",
        ),
    ),
)
def slopesum_baseline(
    session: M3Session,
    processed_emg: Any,
    *,
    window_seconds: float = 30.0,
    step_seconds: float = 1.0,
    percentile: float = 33.0,
    augmented_percentile: float = 25.0,
    moving_average_seconds: float = 0.5,
    percentile_window_seconds: float = 1.0,
) -> dict[str, Any]:
    _require_positive_seconds("window_seconds", window_seconds)
    _require_positive_seconds("step_seconds", step_seconds)
    _require_positive_seconds("moving_average_seconds", moving_average_seconds)
    _require_positive_seconds("percentile_window_seconds", percentile_window_seconds)
    _require_percentile("percentile", percentile)
    _require_percentile("augmented_percentile", augmented_percentile)

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    window_samples = max(1, int(window_seconds * fs))
    step_samples = max(1, int(step_seconds * fs))
    moving_average_samples = max(1, int(moving_average_seconds * fs))
    percentile_window_samples = max(1, int(percentile_window_seconds * fs))
    baseline, running_mean, running_std, series = session.emg_adapter.slopesum_baseline(
        envelope,
        window_samples=window_samples,
        step_samples=step_samples,
        sample_frequency=fs,
        percentile=percentile,
        augmented_percentile=augmented_percentile,
        moving_average_samples=moving_average_samples,
        percentile_window_samples=percentile_window_samples,
    )

    label, unit = _processed_channel_label_and_unit(processed_emg)
    time = np.arange(len(envelope), dtype=float) / fs
    effective_samples = {
        "requested_window_seconds": window_seconds,
        "requested_step_seconds": step_seconds,
        "requested_moving_average_seconds": moving_average_seconds,
        "requested_percentile_window_seconds": percentile_window_seconds,
        "effective_window_samples": window_samples,
        "effective_step_samples": step_samples,
        "effective_moving_average_samples": moving_average_samples,
        "effective_percentile_window_samples": percentile_window_samples,
        "percentile": percentile,
        "augmented_percentile": augmented_percentile,
    }
    baseline_signal = Signal(
        values=baseline,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_slopesum_baseline",
        modality="emg",
        channel=label,
        source="resurfemg",
        processing_state="derived",
        derived_from="processed",
        method="resurfemg.slopesum_baseline",
        metadata=dict(effective_samples),
    )
    running_mean_signal = Signal(
        values=running_mean,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_slopesum_baseline_running_mean",
        modality="emg",
        channel=label,
        source="resurfemg",
        processing_state="derived",
        derived_from="processed",
        method="resurfemg.slopesum_baseline",
    )
    running_std_signal = Signal(
        values=running_std,
        time=time,
        sample_frequency=fs,
        unit=unit,
        name=f"{label}_slopesum_baseline_running_std",
        modality="emg",
        channel=label,
        source="resurfemg",
        processing_state="derived",
        derived_from="processed",
        method="resurfemg.slopesum_baseline",
    )
    session.signals.add(baseline_signal)
    session.signals.add(running_mean_signal)
    session.signals.add(running_std_signal)

    _record_step(
        session,
        "emg.slopesum_baseline",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.baseline.slopesum_baseline",
            operation="emg.slopesum_baseline",
            parameters=effective_samples,
        ),
    )
    return {
        "baseline": baseline,
        "slopesum_baseline_detail": {
            "running_mean": running_mean,
            "running_std": running_std,
            "series": series,
        },
        "baseline_signal": baseline_signal,
        "slopesum_baseline_native_detail": {
            "running_mean": running_mean,
            "running_std": running_std,
        },
        "baseline_running_mean_signal": running_mean_signal,
        "baseline_running_std_signal": running_std_signal,
    }
