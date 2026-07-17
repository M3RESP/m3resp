"""Registered EMG event-quality pipeline steps (manoeuvre/timing/rate checks)."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, QualityFlag
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
    "emg.detect_non_consecutive_manoeuvres",
    reads={
        "session": "session",
        "ventilator_breath_indices": "ventilator_breath_indices",
        "pocc_indices": "pocc_indices",
    },
    writes=(
        "detect_non_consecutive_manoeuvres",
        "detect_non_consecutive_manoeuvres_flags",
    ),
    summary="Flag non-consecutive occlusion manoeuvres against ventilator breaths.",
    description="Flag Pocc manoeuvres that are not consecutive ventilator breaths, since a valid occlusion trial requires uninterrupted breaths.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="ventilator_breath_indices",
            artifact_type="index_array",
            description="Ventilator breath peak indices.",
        ),
        StepArtifact(
            name="pocc_indices",
            artifact_type="index_array",
            description="Pocc manoeuvre peak indices.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="detect_non_consecutive_manoeuvres",
            artifact_type="boolean_array",
            description="Whether each manoeuvre is flagged as non-consecutive.",
        ),
        StepArtifact(
            name="detect_non_consecutive_manoeuvres_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per manoeuvre.",
        ),
    ),
)
def detect_non_consecutive_manoeuvres(
    session: M3Session, ventilator_breath_indices: Any, pocc_indices: Any
) -> dict[str, Any]:
    result = session.emg_adapter.detect_non_consecutive_manoeuvres(
        ventilator_breath_indices, pocc_indices
    )
    flags = _per_breath_flags(
        "detect_non_consecutive_manoeuvres",
        result,
        modality="pressure",
        peak_indices=pocc_indices,
    )
    for flag in flags:
        session.quality.add(flag)
    _record_step(
        session,
        "emg.detect_non_consecutive_manoeuvres",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.detect_non_consecutive_manoeuvres",
            operation="emg.detect_non_consecutive_manoeuvres",
            parameters={},
        ),
    )
    return {
        "detect_non_consecutive_manoeuvres": result,
        "detect_non_consecutive_manoeuvres_flags": flags,
    }


@register_step(
    "emg.evaluate_bell_curve_error",
    reads={
        "session": "session",
        "peak_indices": "peak_indices",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "processed_emg": "processed_emg",
        "time_product": "time_product",
    },
    writes=(
        "evaluate_bell_curve_error",
        "evaluate_bell_curve_error_results",
        "evaluate_bell_curve_error_flags",
    ),
    summary="Score how well each EMG breath matches a bell-curve shape.",
    description="Fit a bell curve to each breath's envelope window and score the fit error as a percentage of the time-product.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
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
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying the envelope and 'fs'.",
        ),
        StepArtifact(
            name="time_product",
            artifact_type="array",
            description="Time-products from 'emg.time_product', used to normalize the fit error.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="evaluate_bell_curve_error",
            artifact_type="array",
            description="Raw upstream (valid, percentage_error, error, y_min, fitted_parameters) tuple.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="evaluate_bell_curve_error_results",
            artifact_type="parameter_result_list",
            unit="%",
            description="Native ParameterResult per breath (percentage error) plus one array-valued ParameterResult per breath for the fitted bell-curve parameters.",
        ),
        StepArtifact(
            name="evaluate_bell_curve_error_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per breath.",
        ),
    ),
)
def evaluate_bell_curve_error(
    session: M3Session,
    peak_indices: Any,
    start_indices: Any,
    end_indices: Any,
    processed_emg: Any,
    time_product: Any,
) -> dict[str, Any]:

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    result = session.emg_adapter.evaluate_bell_curve_error(
        peak_indices,
        start_indices,
        end_indices,
        envelope,
        time_product,
        sample_frequency=fs,
    )
    valid_peak, percentage_bell_error, bell_error, y_min, fitted_parameters = result

    results = _per_breath_results(
        "evaluate_bell_curve_error",
        percentage_bell_error,
        modality="emg",
        peak_indices=peak_indices,
        unit="%",
        method="resurfemg.evaluate_bell_curve_error",
        fs=fs,
        extra_metadata_per_item=[
            {"bell_error": float(bell_error[index]), "y_min": float(y_min[index])}
            for index in range(len(percentage_bell_error))
        ],
    )
    # Array-valued (one fitted bell-curve parameter vector per breath), so
    # this is its own ParameterResult rather than buried in metadata - it
    # then reuses the shared parameter_result_arrays.npz exporter (plan
    # Phase 6.3) instead of a competing EMG-specific array format.
    results.extend(
        _per_breath_results(
            "evaluate_bell_curve_error_fitted_parameters",
            list(np.asarray(fitted_parameters)),
            modality="emg",
            peak_indices=peak_indices,
            method="resurfemg.evaluate_bell_curve_error",
            fs=fs,
        )
    )
    flags = _per_breath_flags(
        "evaluate_bell_curve_error",
        valid_peak,
        modality="emg",
        peak_indices=peak_indices,
        fs=fs,
    )

    for parameter_result in results:
        session.parameter_results.add(parameter_result)
    for flag in flags:
        session.quality.add(flag)

    _record_step(
        session,
        "emg.evaluate_bell_curve_error",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.evaluate_bell_curve_error",
            operation="emg.evaluate_bell_curve_error",
            parameters={},
        ),
    )
    return {
        "evaluate_bell_curve_error": result,
        "evaluate_bell_curve_error_results": results,
        "evaluate_bell_curve_error_flags": flags,
    }


@register_step(
    "emg.evaluate_event_timing",
    reads={
        "session": "session",
        "peak_indices": "peak_indices",
        "processed_emg": "processed_emg",
        "ventilator_breath_indices": "ventilator_breath_indices",
        "ventilator_signals": "ventilator_signals",
    },
    writes=(
        "evaluate_event_timing",
        "evaluate_event_timing_results",
        "evaluate_event_timing_flags",
        "evaluate_event_timing_unmatched_count",
    ),
    summary="Score the timing agreement between EMG and ventilator breaths.",
    description="Pair EMG and ventilator breaths index-by-index and score their timing agreement. Any unpaired breaths at the end (from unequal counts) are reported as a separate warning flag, not silently dropped.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
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
        StepArtifact(
            name="ventilator_breath_indices",
            artifact_type="index_array",
            description="Ventilator breath peak indices.",
        ),
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle supplying 'fs'.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="evaluate_event_timing",
            artifact_type="array",
            description="Raw upstream (correct_timing, delta_time) tuple.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="evaluate_event_timing_results",
            artifact_type="parameter_result_list",
            unit="s",
            description="Native ParameterResult (timing delta) per paired breath.",
        ),
        StepArtifact(
            name="evaluate_event_timing_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per paired breath, plus one warning flag when breaths were unpaired.",
        ),
        StepArtifact(
            name="evaluate_event_timing_unmatched_count",
            artifact_type="count",
            description="Number of unpaired breaths, from unequal EMG/ventilator breath counts.",
        ),
    ),
)
def evaluate_event_timing(
    session: M3Session,
    peak_indices: Any,
    processed_emg: Any,
    ventilator_breath_indices: Any,
    ventilator_signals: Any,
) -> dict[str, Any]:
    fs = float(processed_emg["fs"])
    vent_fs = float(ventilator_signals["fs"])
    # Keep the raw output's existing truncation behavior (Phase 5.1: "existing
    # pipeline consumers do not break"), but report the truncation instead of
    # silently dropping the unmatched events (Phase 5.4).
    paired_count = min(len(peak_indices), len(ventilator_breath_indices))
    unmatched_count = abs(len(peak_indices) - len(ventilator_breath_indices))
    paired_emg_peaks = peak_indices[:paired_count]
    paired_vent_peaks = ventilator_breath_indices[:paired_count]
    result = session.emg_adapter.evaluate_event_timing(
        paired_emg_peaks / fs,
        paired_vent_peaks / vent_fs,
    )
    correct_timing, delta_time = result

    results = _per_breath_results(
        "evaluate_event_timing_delta",
        delta_time,
        modality="emg",
        peak_indices=paired_emg_peaks,
        unit="s",
        method="resurfemg.evaluate_event_timing",
        fs=fs,
        extra_metadata_per_item=[
            {
                "emg_sample_index": int(paired_emg_peaks[index]),
                "ventilator_sample_index": int(paired_vent_peaks[index]),
                "emg_sample_frequency": fs,
                "ventilator_sample_frequency": vent_fs,
            }
            for index in range(paired_count)
        ],
    )
    flags = _per_breath_flags(
        "evaluate_event_timing",
        correct_timing,
        modality="emg",
        peak_indices=paired_emg_peaks,
        fs=fs,
    )
    if unmatched_count:
        flags.append(
            QualityFlag(
                name="evaluate_event_timing_unmatched",
                passed=False,
                severity="warning",
                modality="emg",
                message=(
                    f"{unmatched_count} event(s) had no paired counterpart and "
                    "were not assessed."
                ),
                metadata={
                    "unmatched_count": unmatched_count,
                    "emg_event_count": len(peak_indices),
                    "ventilator_event_count": len(ventilator_breath_indices),
                },
            )
        )

    for parameter_result in results:
        session.parameter_results.add(parameter_result)
    for flag in flags:
        session.quality.add(flag)

    _record_step(
        session,
        "emg.evaluate_event_timing",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.evaluate_event_timing",
            operation="emg.evaluate_event_timing",
            parameters={"unmatched_count": unmatched_count},
        ),
    )
    return {
        "evaluate_event_timing": result,
        "evaluate_event_timing_results": results,
        "evaluate_event_timing_flags": flags,
        "evaluate_event_timing_unmatched_count": unmatched_count,
    }


@register_step(
    "emg.evaluate_respiratory_rates",
    reads={
        "session": "session",
        "peak_indices": "peak_indices",
        "processed_emg": "processed_emg",
        "ventilator_respiratory_rate": "ventilator_respiratory_rate",
    },
    writes=(
        "evaluate_respiratory_rates",
        "evaluate_respiratory_rates_result",
        "evaluate_respiratory_rates_flag",
    ),
    summary="Score agreement between EMG-derived and ventilator-derived respiratory rate.",
    description="Check that the fraction of EMG breaths detected relative to the ventilator-derived respiratory rate exceeds 'minimum_fraction'.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="peak_indices",
            artifact_type="index_array",
            description="EMG breath peak indices.",
        ),
        StepArtifact(
            name="processed_emg",
            artifact_type="emg_processed_bundle",
            description="Processed EMG bundle supplying 'fs' and the envelope's duration.",
        ),
        StepArtifact(
            name="ventilator_respiratory_rate",
            artifact_type="scalar_metric",
            unit="breaths/min",
            description="Ventilator-derived respiratory rate from 'emg.ventilator_respiratory_rate'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="minimum_fraction",
            value_type="number",
            default=0.1,
            minimum=0,
            maximum=1,
            description="Minimum acceptable fraction of expected EMG breaths detected.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="evaluate_respiratory_rates",
            artifact_type="array",
            description="Raw upstream (detected_fraction, criterion_met) tuple.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="evaluate_respiratory_rates_result",
            artifact_type="parameter_result",
            description="Native ParameterResult: detected fraction.",
        ),
        StepArtifact(
            name="evaluate_respiratory_rates_flag",
            artifact_type="quality_flag",
            description="Native QualityFlag for the overall check.",
        ),
    ),
)
def evaluate_respiratory_rates(
    session: M3Session,
    peak_indices: Any,
    processed_emg: Any,
    ventilator_respiratory_rate: Any,
    *,
    minimum_fraction: float = 0.1,
) -> dict[str, Any]:
    fs = float(processed_emg["fs"])
    envelope = processed_emg["envelope"]
    rr_vent = ventilator_respiratory_rate[0]
    result = session.emg_adapter.evaluate_respiratory_rates(
        peak_indices, len(envelope) / fs, rr_vent, minimum_fraction=minimum_fraction
    )
    detected_fraction, criterion_met = result

    parameter_result = ParameterResult(
        name="evaluate_respiratory_rates_detected_fraction",
        value=detected_fraction,
        modality="emg",
        method="resurfemg.evaluate_respiratory_rates",
        metadata={
            "minimum_fraction": minimum_fraction,
            "ventilator_rr": float(rr_vent),
        },
    )
    flag = QualityFlag(
        name="evaluate_respiratory_rates",
        passed=bool(criterion_met),
        severity="info",
        modality="emg",
        value=detected_fraction,
        threshold=minimum_fraction,
        metadata={"ventilator_rr": float(rr_vent)},
    )

    session.parameter_results.add(parameter_result)
    session.quality.add(flag)

    _record_step(
        session,
        "emg.evaluate_respiratory_rates",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.evaluate_respiratory_rates",
            operation="emg.evaluate_respiratory_rates",
            parameters={"minimum_fraction": minimum_fraction},
        ),
    )
    return {
        "evaluate_respiratory_rates": result,
        "evaluate_respiratory_rates_result": parameter_result,
        "evaluate_respiratory_rates_flag": flag,
    }
