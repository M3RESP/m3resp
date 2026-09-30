"""Registered EMG baseline-estimation workflow steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from m3resp.data import Signal
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import (
    _RESURFEMG,
    _SESSION_ARTIFACT,
    _processed_channel_label_and_unit,
    _require_percentile,
    _require_positive_seconds,
    _upstream_metadata,
    require_envelope,
)

if TYPE_CHECKING:
    from m3resp.core.session import M3Session


@register_step(
    "emg.moving_baseline",
    reads={"session": "session", "processed_emg": "processed_emg"},
    writes=("baseline", "baseline_signal"),
    summary="Compute a moving-percentile EMG baseline.",
    description="Compute a moving-percentile baseline of the EMG envelope via ReSurfEMGAdapter.moving_baseline.",
    category="baseline",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_writes=("session.signals",),
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
            default=7.5,
            unit="s",
            minimum=0,
            description="Moving-percentile window length.",
        ),
        StepParameter(
            name="step_seconds",
            value_type="number",
            default=0.2,
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
    record=True,
)
def moving_baseline(
    session: M3Session,
    processed_emg: Any,
    *,
    window_seconds: float = 7.5,
    step_seconds: float = 0.2,
    percentile: float = 33.0,
) -> dict[str, Any]:
    baseline, baseline_signal, parameters = compute_moving_baseline(
        session,
        processed_emg,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        percentile=percentile,
    )
    session.signals.add(baseline_signal)
    return {
        "baseline": baseline,
        "baseline_signal": baseline_signal,
        "_provenance": _upstream_metadata(
            source_function="resurfemg.postprocessing.baseline.moving_baseline",
            operation="emg.moving_baseline",
            parameters=parameters,
        ),
    }


def compute_moving_baseline(
    session: M3Session,
    processed_emg: Any,
    *,
    window_seconds: float = 7.5,
    step_seconds: float = 0.2,
    percentile: float = 33.0,
    variant: str | None = None,
) -> tuple[np.ndarray, Signal, dict[str, Any]]:
    """The moving baseline of the envelope in `processed_emg`, as an array and a `Signal`.

    Used by the ``emg.moving_baseline`` step and by
    `M3Session.emg_moving_baseline`. The `Signal` is returned and is added to
    the session by the caller.

    Args:
        session (M3Session): Session that runs the ReSurfEMG calculation.
        processed_emg (Mapping): Processed EMG bundle with the envelope and
            the sampling rate ``"fs"`` in Hz.
        window_seconds (float): Window length in seconds.
        step_seconds (float): Time between baseline updates, in seconds.
        percentile (float): Percentile of the envelope in each window,
            0 to 100.
        variant (str | None): Name of the preprocessing variant the envelope
            came from. It is added to the signal name
            (``<channel>_moving_baseline:<variant>``) and metadata; None adds
            nothing.

    Returns:
        tuple: the baseline (one value per envelope sample, in envelope
            units), its `Signal`, and the settings used (requested seconds
            and the window and step in samples).

    Raises:
        ValueError: If a setting is out of range or the bundle has no envelope.
    """

    _require_positive_seconds("window_seconds", window_seconds)
    _require_positive_seconds("step_seconds", step_seconds)
    _require_percentile("percentile", percentile)

    envelope = require_envelope(processed_emg, "emg.moving_baseline")
    fs = float(processed_emg["fs"])
    window_samples = max(1, int(window_seconds * fs))
    step_samples = max(1, int(step_seconds * fs))
    baseline = session.emg_adapter.moving_baseline(
        envelope,
        window_samples=window_samples,
        step_samples=step_samples,
        percentile=percentile,
    )
    parameters = {
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "percentile": percentile,
        "effective_window_samples": window_samples,
        "effective_step_samples": step_samples,
    }

    label, unit = _processed_channel_label_and_unit(processed_emg)
    name = f"{label}_moving_baseline"
    metadata: dict[str, Any] = {
        "requested_window_seconds": window_seconds,
        "requested_step_seconds": step_seconds,
        "effective_window_samples": window_samples,
        "effective_step_samples": step_samples,
        "percentile": percentile,
    }
    if variant is not None:
        name = f"{name}:{variant}"
        metadata["variant"] = variant
    baseline_signal = Signal(
        values=baseline,
        time=np.arange(len(envelope), dtype=float) / fs,
        sample_frequency=fs,
        unit=unit,
        name=name,
        modality="emg",
        category="electrical_potential",
        channel=label,
        source="resurfemg",
        processing_state="derived",
        derived_from="processed",
        method="resurfemg.moving_baseline",
        metadata=metadata,
    )
    return baseline, baseline_signal, parameters


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
    session_writes=("session.signals",),
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
            default=7.5,
            unit="s",
            minimum=0,
            description="Moving-percentile window length.",
        ),
        StepParameter(
            name="step_seconds",
            value_type="number",
            default=0.2,
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
            description="Percentile of the envelope plus its slope in each window; 1.2 times this value is the slope-sum baseline.",
        ),
        StepParameter(
            name="moving_average_seconds",
            value_type="number",
            default=0.5,
            unit="s",
            minimum=0,
            description="Smoothing window of the envelope slope.",
        ),
        StepParameter(
            name="percentile_window_seconds",
            value_type="number",
            default=1.0,
            unit="s",
            minimum=0,
            description="Time between updates of the slope-sum baseline.",
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
            description="Running mean and standard deviation of the moving baseline, plus the moving baseline as a pandas Series.",
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
            description="Native Signal wrapping the running mean of the moving baseline.",
        ),
        StepArtifact(
            name="baseline_running_std_signal",
            artifact_type="signal",
            description="Native Signal wrapping the running standard deviation of the moving baseline.",
        ),
    ),
    record=True,
)
def slopesum_baseline(
    session: M3Session,
    processed_emg: Any,
    *,
    window_seconds: float = 7.5,
    step_seconds: float = 0.2,
    percentile: float = 33.0,
    augmented_percentile: float = 25.0,
    moving_average_seconds: float = 0.5,
    percentile_window_seconds: float = 1.0,
) -> dict[str, Any]:
    baseline, signals, detail, effective_samples = compute_slopesum_baseline(
        session,
        processed_emg,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        percentile=percentile,
        augmented_percentile=augmented_percentile,
        moving_average_seconds=moving_average_seconds,
        percentile_window_seconds=percentile_window_seconds,
    )
    baseline_signal, running_mean_signal, running_std_signal = signals
    for signal in signals:
        session.signals.add(signal)

    return {
        "baseline": baseline,
        "slopesum_baseline_detail": detail,
        "baseline_signal": baseline_signal,
        "slopesum_baseline_native_detail": {
            "running_mean": detail["running_mean"],
            "running_std": detail["running_std"],
        },
        "baseline_running_mean_signal": running_mean_signal,
        "baseline_running_std_signal": running_std_signal,
        "_provenance": _upstream_metadata(
            source_function="resurfemg.postprocessing.baseline.slopesum_baseline",
            operation="emg.slopesum_baseline",
            parameters=effective_samples,
        ),
    }


def compute_slopesum_baseline(
    session: M3Session,
    processed_emg: Any,
    *,
    window_seconds: float = 7.5,
    step_seconds: float = 0.2,
    percentile: float = 33.0,
    augmented_percentile: float = 25.0,
    moving_average_seconds: float = 0.5,
    percentile_window_seconds: float = 1.0,
    variant: str | None = None,
) -> tuple[np.ndarray, tuple[Signal, Signal, Signal], dict[str, Any], dict[str, Any]]:
    """The slope-sum baseline of the envelope in `processed_emg`, with its `Signal`s.

    In each window, the slope-sum baseline is 1.2 times a percentile of the
    envelope plus its absolute, smoothed slope. ReSurfEMG also returns the
    running mean and running standard deviation of the moving baseline.

    Used by the ``emg.slopesum_baseline`` step and by
    `M3Session.emg_slopesum_baseline`. The three `Signal`s (slope-sum
    baseline, and running mean and running standard deviation of the moving
    baseline) are returned and are added to the session by the caller.

    Args:
        session (M3Session): Session that runs the ReSurfEMG calculation.
        processed_emg (Mapping): Processed EMG bundle with the envelope and
            the sampling rate ``"fs"`` in Hz.
        window_seconds (float): Window length in seconds.
        step_seconds (float): Time between updates of the moving baseline,
            in seconds.
        percentile (float): Percentile of the envelope in each window for
            the moving baseline, 0 to 100.
        augmented_percentile (float): Percentile of the envelope plus its
            slope in each window, 0 to 100.
        moving_average_seconds (float): Smoothing window of the envelope
            slope, in seconds.
        percentile_window_seconds (float): Time between updates of the
            slope-sum baseline, in seconds.
        variant (str | None): Name of the preprocessing variant the envelope
            came from. It is added to each signal name
            (``<channel>_slopesum_baseline:<variant>``) and metadata; None
            adds nothing.

    Returns:
        tuple: the slope-sum baseline (one value per envelope sample, in
            envelope units); its three `Signal`s; the running mean, running
            standard deviation and moving baseline from ReSurfEMG
            (``"running_mean"``, ``"running_std"``, ``"series"``); and the
            settings used (requested seconds and the windows in samples).

    Raises:
        ValueError: If a setting is out of range or the bundle has no envelope.
    """

    _require_positive_seconds("window_seconds", window_seconds)
    _require_positive_seconds("step_seconds", step_seconds)
    _require_positive_seconds("moving_average_seconds", moving_average_seconds)
    _require_positive_seconds("percentile_window_seconds", percentile_window_seconds)
    _require_percentile("percentile", percentile)
    _require_percentile("augmented_percentile", augmented_percentile)

    envelope = require_envelope(processed_emg, "emg.slopesum_baseline")
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
    suffix = "" if variant is None else f":{variant}"
    variant_metadata = {} if variant is None else {"variant": variant}

    def baseline_part(values: Any, name: str, metadata: dict[str, Any]) -> Signal:
        return Signal(
            values=values,
            time=time,
            sample_frequency=fs,
            unit=unit,
            name=f"{label}_{name}{suffix}",
            modality="emg",
            category="electrical_potential",
            channel=label,
            source="resurfemg",
            processing_state="derived",
            derived_from="processed",
            method="resurfemg.slopesum_baseline",
            metadata={**metadata, **variant_metadata},
        )

    signals = (
        baseline_part(baseline, "slopesum_baseline", dict(effective_samples)),
        baseline_part(running_mean, "slopesum_baseline_running_mean", {}),
        baseline_part(running_std, "slopesum_baseline_running_std", {}),
    )
    detail = {
        "running_mean": running_mean,
        "running_std": running_std,
        "series": series,
    }
    return baseline, signals, detail, effective_samples
