"""Registered EMG signal-quality pipeline steps (SNR/AUB/time-product based)."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import (
    _RESURFEMG,
    _SESSION_ARTIFACT,
    _per_breath_flags,
    _per_breath_results,
    _record_step,
    _upstream_metadata,
)


@register_step(
    "emg.snr_pseudo",
    reads={
        "session": "session",
        "processed_emg": "processed_emg",
        "peak_indices": "peak_indices",
        "baseline": "baseline",
    },
    writes=("snr_pseudo", "snr_pseudo_results", "snr_pseudo_flags"),
    summary="Compute a pseudo signal-to-noise ratio for detected EMG breaths.",
    description="Compute a pseudo signal-to-noise ratio per breath. A measurement only becomes a pass/fail criterion when 'minimum_snr' is set; otherwise only the measurement is produced.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_writes=("session.quality", "session.parameter_results"),
    input_artifacts=(
        _SESSION_ARTIFACT,
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
            name="baseline",
            artifact_type="signal_array",
            description="Baseline from 'emg.moving_baseline' or 'emg.slopesum_baseline'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="minimum_snr",
            value_type="number",
            required=False,
            default=None,
            description="Minimum acceptable SNR. When unset, only the measurement is produced, with no pass/fail flags.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="snr_pseudo",
            artifact_type="array",
            description="Pseudo-SNR per breath.",
        ),
        StepArtifact(
            name="snr_pseudo_results",
            artifact_type="parameter_result_list",
            description="Native ParameterResult per breath.",
        ),
        StepArtifact(
            name="snr_pseudo_flags",
            artifact_type="quality_flag_list",
            required=False,
            description="Native QualityFlag per breath, only when 'minimum_snr' is set.",
        ),
    ),
)
def snr_pseudo(
    session: M3Session,
    processed_emg: Any,
    peak_indices: Any,
    baseline: Any,
    *,
    minimum_snr: float | None = None,
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    result = session.emg_adapter.snr_pseudo(
        envelope, peak_indices, baseline, sample_frequency=fs
    )

    results = _per_breath_results(
        "snr_pseudo",
        result,
        modality="emg",
        peak_indices=peak_indices,
        method="resurfemg.snr_pseudo",
        fs=fs,
    )
    # A measurement only becomes a criterion when a threshold is actually
    # configured - no invented pass/fail otherwise (see
    # m3resp.processing.quality.quality_flag_from_result's docstring).
    flags = (
        _per_breath_flags(
            "snr_pseudo",
            result >= minimum_snr,
            modality="emg",
            peak_indices=peak_indices,
            fs=fs,
            threshold=minimum_snr,
        )
        if minimum_snr is not None
        else []
    )

    for parameter_result in results:
        session.parameter_results.add(parameter_result)
    for flag in flags:
        session.quality.add(flag)

    _record_step(
        session,
        "emg.snr_pseudo",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.snr_pseudo",
            operation="emg.snr_pseudo",
            parameters={"minimum_snr": minimum_snr},
        ),
    )
    return {
        "snr_pseudo": result,
        "snr_pseudo_results": results,
        "snr_pseudo_flags": flags,
    }


@register_step(
    "emg.percentage_under_baseline",
    reads={
        "session": "session",
        "processed_emg": "processed_emg",
        "peak_indices": "peak_indices",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "baseline": "baseline",
    },
    writes=(
        "percentage_under_baseline",
        "percentage_under_baseline_results",
        "percentage_under_baseline_flags",
    ),
    summary="Compute the percentage of each EMG breath spent under baseline.",
    description="Compute the percentage of each breath's window spent with the envelope under baseline, flagging breaths above 'aub_threshold'.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_writes=("session.quality", "session.parameter_results"),
    input_artifacts=(
        _SESSION_ARTIFACT,
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
    ),
    parameters=(
        StepParameter(
            name="aub_window_seconds",
            value_type="number",
            required=False,
            default=None,
            unit="s",
            description="Window searched for under-baseline area. Defaults to the breath's own onset/offset window when unset.",
        ),
        StepParameter(
            name="aub_threshold",
            value_type="number",
            default=40.0,
            minimum=0,
            maximum=100,
            unit="%",
            description="Maximum acceptable percentage under baseline.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="percentage_under_baseline",
            artifact_type="array",
            description="Raw upstream (valid, percentages, reference_values) tuple.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="percentage_under_baseline_results",
            artifact_type="parameter_result_list",
            unit="%",
            description="Native ParameterResult per breath.",
        ),
        StepArtifact(
            name="percentage_under_baseline_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per breath.",
        ),
    ),
)
def percentage_under_baseline(
    session: M3Session,
    processed_emg: Any,
    peak_indices: Any,
    start_indices: Any,
    end_indices: Any,
    baseline: Any,
    *,
    aub_window_seconds: float | None = None,
    aub_threshold: float = 40.0,
) -> dict[str, Any]:
    import numpy as np

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    aub_window_samples = (
        max(1, int(aub_window_seconds * fs)) if aub_window_seconds is not None else None
    )
    result = session.emg_adapter.percentage_under_baseline(
        envelope,
        peak_indices,
        start_indices,
        end_indices,
        baseline,
        sample_frequency=fs,
        aub_window_samples=aub_window_samples,
        aub_threshold=aub_threshold,
    )
    valid, percentages, reference_values = result

    results = _per_breath_results(
        "percentage_under_baseline",
        percentages,
        modality="emg",
        peak_indices=peak_indices,
        unit="%",
        method="resurfemg.percentage_under_baseline",
        fs=fs,
        extra_metadata_per_item=[
            {"reference_value": float(reference)} for reference in reference_values
        ],
    )
    flags = _per_breath_flags(
        "percentage_under_baseline",
        valid,
        modality="emg",
        peak_indices=peak_indices,
        fs=fs,
        threshold=aub_threshold,
    )

    for parameter_result in results:
        session.parameter_results.add(parameter_result)
    for flag in flags:
        session.quality.add(flag)

    _record_step(
        session,
        "emg.percentage_under_baseline",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.percentage_under_baseline",
            operation="emg.percentage_under_baseline",
            parameters={
                "aub_window_seconds": aub_window_seconds,
                "aub_threshold": aub_threshold,
                "effective_aub_window_samples": aub_window_samples,
            },
        ),
    )
    return {
        "percentage_under_baseline": result,
        "percentage_under_baseline_results": results,
        "percentage_under_baseline_flags": flags,
    }


@register_step(
    "emg.detect_local_high_aub",
    reads={
        "session": "session",
        "area_under_baseline": "area_under_baseline",
        "peak_indices": "peak_indices",
    },
    writes=(
        "detect_local_high_aub",
        "detect_local_high_aub_flags",
        "detect_local_high_aub_threshold_result",
    ),
    summary="Flag EMG breaths with locally elevated area-under-baseline.",
    description="Flag breaths whose area-under-baseline exceeds 'threshold_factor' times a local percentile of the recording's own area-under-baseline values.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_writes=("session.quality", "session.parameter_results"),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="area_under_baseline",
            artifact_type="array",
            description="Area-under-baseline values from 'emg.area_under_baseline'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices.",
        ),
    ),
    parameters=(
        StepParameter(
            name="threshold_percentile",
            value_type="number",
            default=75.0,
            minimum=0,
            maximum=100,
            description="Percentile of area-under-baseline used as the local reference.",
        ),
        StepParameter(
            name="threshold_factor",
            value_type="number",
            default=4.0,
            minimum=0,
            description="Multiplier applied to the reference percentile to get the flagging threshold.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="detect_local_high_aub",
            artifact_type="boolean_array",
            description="Whether each breath is flagged.",
        ),
        StepArtifact(
            name="detect_local_high_aub_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per breath.",
        ),
        StepArtifact(
            name="detect_local_high_aub_threshold_result",
            artifact_type="parameter_result",
            description="Native ParameterResult: the effective threshold used.",
        ),
    ),
)
def detect_local_high_aub(
    session: M3Session,
    area_under_baseline: Any,
    peak_indices: Any,
    *,
    threshold_percentile: float = 75.0,
    threshold_factor: float = 4.0,
) -> dict[str, Any]:
    aubs = area_under_baseline[0]
    result = session.emg_adapter.detect_local_high_aub(
        aubs,
        threshold_percentile=threshold_percentile,
        threshold_factor=threshold_factor,
    )
    # Upstream's own formula (resurfemg.postprocessing.quality_assessment.
    # detect_local_high_aub) - recomputed here since it only returns the
    # boolean array, not the threshold it compared against.
    effective_threshold = float(
        threshold_factor
        * np.percentile(np.asarray(aubs, dtype=float), threshold_percentile)
    )

    flags = _per_breath_flags(
        "detect_local_high_aub",
        result,
        modality="emg",
        peak_indices=peak_indices,
        threshold=effective_threshold,
    )
    threshold_result = ParameterResult(
        name="detect_local_high_aub_threshold",
        value=effective_threshold,
        modality="emg",
        method="resurfemg.detect_local_high_aub",
        metadata={
            "threshold_percentile": threshold_percentile,
            "threshold_factor": threshold_factor,
        },
    )

    for flag in flags:
        session.quality.add(flag)
    session.parameter_results.add(threshold_result)

    _record_step(
        session,
        "emg.detect_local_high_aub",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.detect_local_high_aub",
            operation="emg.detect_local_high_aub",
            parameters={
                "threshold_percentile": threshold_percentile,
                "threshold_factor": threshold_factor,
            },
        ),
    )
    return {
        "detect_local_high_aub": result,
        "detect_local_high_aub_flags": flags,
        "detect_local_high_aub_threshold_result": threshold_result,
    }


@register_step(
    "emg.detect_extreme_time_products",
    reads={
        "session": "session",
        "time_product": "time_product",
        "peak_indices": "peak_indices",
    },
    writes=(
        "detect_extreme_time_products",
        "detect_extreme_time_products_flags",
        "detect_extreme_time_products_bounds_result",
    ),
    summary="Flag EMG breaths with extreme time-products.",
    description="Flag breaths whose time-product falls outside percentile-derived upper/lower bounds of the recording's own time-products.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    session_writes=("session.quality", "session.parameter_results"),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="time_product",
            artifact_type="array",
            description="Time-products from 'emg.time_product'.",
        ),
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="Breath peak indices.",
        ),
    ),
    parameters=(
        StepParameter(
            name="upper_percentile",
            value_type="number",
            default=95.0,
            minimum=0,
            maximum=100,
            description="Percentile of time-products used as the upper reference.",
        ),
        StepParameter(
            name="upper_factor",
            value_type="number",
            default=10.0,
            minimum=0,
            description="Multiplier applied to the upper reference percentile.",
        ),
        StepParameter(
            name="lower_percentile",
            value_type="number",
            default=5.0,
            minimum=0,
            maximum=100,
            description="Percentile of time-products used as the lower reference.",
        ),
        StepParameter(
            name="lower_factor",
            value_type="number",
            default=0.1,
            minimum=0,
            description="Multiplier applied to the lower reference percentile.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="detect_extreme_time_products",
            artifact_type="boolean_array",
            description="Whether each breath is flagged.",
        ),
        StepArtifact(
            name="detect_extreme_time_products_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per breath.",
        ),
        StepArtifact(
            name="detect_extreme_time_products_bounds_result",
            artifact_type="parameter_result",
            description="Native ParameterResult: [lower_bound, upper_bound].",
        ),
    ),
)
def detect_extreme_time_products(
    session: M3Session,
    time_product: Any,
    peak_indices: Any,
    *,
    upper_percentile: float = 95.0,
    upper_factor: float = 10.0,
    lower_percentile: float = 5.0,
    lower_factor: float = 0.1,
) -> dict[str, Any]:
    result = session.emg_adapter.detect_extreme_time_products(
        time_product,
        upper_percentile=upper_percentile,
        upper_factor=upper_factor,
        lower_percentile=lower_percentile,
        lower_factor=lower_factor,
    )
    values = np.asarray(time_product, dtype=float)
    upper_bound = float(upper_factor * np.percentile(values, upper_percentile))
    lower_bound = float(lower_factor * np.percentile(values, lower_percentile))

    flags = _per_breath_flags(
        "detect_extreme_time_products",
        result,
        modality="emg",
        peak_indices=peak_indices,
    )
    bounds_result = ParameterResult(
        name="detect_extreme_time_products_bounds",
        value=np.array([lower_bound, upper_bound]),
        modality="emg",
        method="resurfemg.detect_extreme_time_products",
        metadata={
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "upper_percentile": upper_percentile,
            "upper_factor": upper_factor,
            "lower_percentile": lower_percentile,
            "lower_factor": lower_factor,
        },
    )

    for flag in flags:
        session.quality.add(flag)
    session.parameter_results.add(bounds_result)

    _record_step(
        session,
        "emg.detect_extreme_time_products",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.detect_extreme_time_products",
            operation="emg.detect_extreme_time_products",
            parameters={
                "upper_percentile": upper_percentile,
                "upper_factor": upper_factor,
                "lower_percentile": lower_percentile,
                "lower_factor": lower_factor,
            },
        ),
    )
    return {
        "detect_extreme_time_products": result,
        "detect_extreme_time_products_flags": flags,
        "detect_extreme_time_products_bounds_result": bounds_result,
    }
