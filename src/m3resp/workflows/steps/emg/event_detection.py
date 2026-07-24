"""Registered EMG/ventilator breath and Pocc event-detection pipeline steps."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.events import BreathEvent
from m3resp.core.session import M3Session
from m3resp.data import ParameterResult
from m3resp.processing.intervals import (
    onoff_from_baseline_crossings,
)
from m3resp.processing.metrics import (
    window_integral,
)
from m3resp.processing.peaks import (
    detect_occluded_breath_peaks,
    detect_ventilator_breath_peaks,
)
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
    "emg.detect_ventilator_breath",
    reads={"ventilator_signals": "ventilator_signals"},
    writes=("ventilator_breath_indices",),
    summary="Detect ventilator breaths from the ventilator volume channel.",
    description="Detect ventilator breath peaks from the volume channel.",
    category="detection",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle from 'emg.ventilator_channels'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="breath_width_seconds",
            value_type="number",
            default=0.5,
            unit="s",
            minimum=0,
            description="Minimum breath width.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ventilator_breath_indices",
            artifact_type="index_array",
            description="Detected ventilator breath peak sample indices.",
        ),
    ),
)
def detect_ventilator_breath(
    ventilator_signals: Any, *, breath_width_seconds: float = 0.5
) -> dict[str, Any]:
    import numpy as np

    volume = ventilator_signals["volume"]
    fs = float(ventilator_signals["fs"])
    width_samples = max(1, int(breath_width_seconds * fs))
    indices = detect_ventilator_breath_peaks(
        volume,
        start_index=0,
        end_index=len(volume) - 1,
        width_samples=width_samples,
    )
    return {"ventilator_breath_indices": np.asarray(indices, dtype=int)}


@register_step(
    "emg.find_occluded_breaths",
    reads={"ventilator_signals": "ventilator_signals"},
    writes=("pocc_indices",),
    summary="Detect occluded (Pocc) breaths from the ventilator pressure channel.",
    description="Detect occlusion (Pocc) manoeuvre peaks from the pressure channel.",
    category="detection",
    modality="emg",
    input_artifacts=(
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle from 'emg.ventilator_channels'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="peep",
            value_type="number",
            required=False,
            default=None,
            unit="cmH2O",
            description="PEEP baseline. Defaults to the median pressure when unset.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pocc_indices",
            artifact_type="index_array",
            description="Detected Pocc manoeuvre peak sample indices.",
        ),
    ),
)
def find_occluded_breaths(
    ventilator_signals: Any, *, peep: float | None = None
) -> dict[str, Any]:
    import numpy as np

    pressure = ventilator_signals["pressure"]
    fs = float(ventilator_signals["fs"])
    if peep is None:
        peep = float(np.nanmedian(pressure))
    indices = detect_occluded_breath_peaks(
        pressure,
        sample_frequency=fs,
        peep=peep,
    )
    return {"pocc_indices": np.asarray(indices, dtype=int)}


@register_step(
    "emg.pocc_intervals",
    reads={
        "session": "session",
        "ventilator_signals": "ventilator_signals",
        "pocc_indices": "pocc_indices",
    },
    writes=(
        "pocc_start_indices",
        "pocc_end_indices",
        "pocc_interval_validity",
        "pocc_events",
    ),
    summary="Find Pocc manoeuvre start/end indices from the pressure channel.",
    description="Find Pocc manoeuvre start/end indices around each detected peak via baseline crossing, and record BreathEvents.",
    category="detection",
    modality="emg",
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle from 'emg.ventilator_channels'.",
        ),
        StepArtifact(
            name="pocc_indices",
            artifact_type="index_array",
            description="Pocc peak indices from 'emg.find_occluded_breaths'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="peep",
            value_type="number",
            required=False,
            default=None,
            unit="cmH2O",
            description="PEEP baseline. Should match the value used in 'emg.find_occluded_breaths'; defaults to the median pressure when unset.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pocc_start_indices",
            artifact_type="index_array",
            description="Pocc manoeuvre start sample indices.",
        ),
        StepArtifact(
            name="pocc_end_indices",
            artifact_type="index_array",
            description="Pocc manoeuvre end sample indices.",
        ),
        StepArtifact(
            name="pocc_interval_validity",
            artifact_type="boolean_array",
            description="Whether each manoeuvre's start/end crossing was found.",
        ),
        StepArtifact(
            name="pocc_events",
            artifact_type="breath_event_list",
            description="Native BreathEvent per Pocc manoeuvre.",
        ),
    ),
)
def pocc_intervals(
    session: M3Session,
    ventilator_signals: Any,
    pocc_indices: Any,
    *,
    peep: float | None = None,
) -> dict[str, Any]:
    pressure = np.asarray(ventilator_signals["pressure"], dtype=float)
    fs = float(ventilator_signals["fs"])
    peaks = np.asarray(pocc_indices, dtype=int)

    # Same PEEP rule as emg.find_occluded_breaths, so pocc_indices (detected
    # against this same baseline) and these intervals stay consistent.
    effective_peep = peep if peep is not None else float(np.nanmedian(pressure))
    baseline = np.full(pressure.shape, effective_peep)

    starts, ends, valid_starts, valid_ends, valid_peaks = onoff_from_baseline_crossings(
        pressure, baseline, peaks
    )

    events: list[BreathEvent] = []
    for index, peak in enumerate(peaks):
        events.append(
            BreathEvent(
                modality="pressure",
                start_time=float(starts[index]) / fs,
                end_time=float(ends[index]) / fs,
                peak_time=float(peak) / fs,
                start_index=int(starts[index]),
                peak_index=int(peak),
                end_index=int(ends[index]),
                sample_frequency=fs,
                signal_name="pressure",
                source="m3resp.processing.intervals.onoff_from_baseline_crossings",
                metadata={
                    "event_type": "pocc",
                    "peep": effective_peep,
                    "valid": bool(valid_peaks[index]),
                    "valid_start": bool(valid_starts[index]),
                    "valid_end": bool(valid_ends[index]),
                },
            )
        )
    session.add_events("pocc_breaths", events)

    _record_step(
        session,
        "emg.pocc_intervals",
        metadata=_upstream_metadata(
            source_function="m3resp.processing.intervals.onoff_from_baseline_crossings",
            operation="emg.pocc_intervals",
            parameters={"peep": effective_peep, "requested_peep": peep},
            source_package="m3resp",
            implementation="m3resp.processing.intervals",
        ),
    )
    return {
        "pocc_start_indices": starts,
        "pocc_end_indices": ends,
        "pocc_interval_validity": np.asarray(valid_peaks, dtype=bool),
        "pocc_events": events,
    }


@register_step(
    "emg.pocc_time_product",
    reads={
        "session": "session",
        "ventilator_signals": "ventilator_signals",
        "pocc_start_indices": "pocc_start_indices",
        "pocc_end_indices": "pocc_end_indices",
    },
    writes=("pocc_time_products", "pocc_time_product_result"),
    summary="Compute the pressure-time product for each Pocc manoeuvre.",
    description="Integrate pressure above the PEEP baseline over each Pocc manoeuvre's start/end window.",
    category="parameters",
    modality="emg",
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle from 'emg.ventilator_channels'.",
        ),
        StepArtifact(
            name="pocc_start_indices",
            artifact_type="index_array",
            description="Pocc manoeuvre start indices from 'emg.pocc_intervals'.",
        ),
        StepArtifact(
            name="pocc_end_indices",
            artifact_type="index_array",
            description="Pocc manoeuvre end indices from 'emg.pocc_intervals'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="peep",
            value_type="number",
            required=False,
            default=None,
            unit="cmH2O",
            description="PEEP baseline. Should match the value used in 'emg.pocc_intervals'; defaults to the median pressure when unset.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pocc_time_products",
            artifact_type="array",
            unit="cmH2O*s",
            description="Pressure-time product per Pocc manoeuvre.",
        ),
        StepArtifact(
            name="pocc_time_product_result",
            artifact_type="parameter_result",
            unit="cmH2O*s",
            description="Native array-valued ParameterResult of the same values.",
        ),
    ),
)
def pocc_time_product(
    session: M3Session,
    ventilator_signals: Any,
    pocc_start_indices: Any,
    pocc_end_indices: Any,
    *,
    peep: float | None = None,
) -> dict[str, Any]:
    pressure = np.asarray(ventilator_signals["pressure"], dtype=float)
    fs = float(ventilator_signals["fs"])
    effective_peep = peep if peep is not None else float(np.nanmedian(pressure))
    baseline = np.full(pressure.shape, effective_peep)

    time_products = window_integral(
        pressure, fs, pocc_start_indices, pocc_end_indices, baseline
    )

    pressure_unit = ventilator_signals.get("unit") or "cmH2O"
    parameters = {"peep": effective_peep, "requested_peep": peep}
    result = ParameterResult(
        name="pocc_time_product",
        value=time_products,
        modality="pressure",
        unit=f"{pressure_unit}*s",
        method="m3resp.processing.metrics.window_integral",
        metadata={
            **parameters,
            "start_indices": np.asarray(pocc_start_indices, dtype=int).tolist(),
            "end_indices": np.asarray(pocc_end_indices, dtype=int).tolist(),
        },
    )

    _record_step(
        session,
        "emg.pocc_time_product",
        metadata=_upstream_metadata(
            source_function="m3resp.processing.metrics.window_integral",
            operation="emg.pocc_time_product",
            parameters=parameters,
            source_package="m3resp",
            implementation="m3resp.processing.metrics",
        ),
    )
    return {"pocc_time_products": time_products, "pocc_time_product_result": result}


_POCC_CRITERIA_ROW_NAMES = ("dp_up_10", "dp_up_90", "dp_up_90_norm")


@register_step(
    "emg.pocc_quality",
    reads={
        "session": "session",
        "ventilator_signals": "ventilator_signals",
        "pocc_indices": "pocc_indices",
        "pocc_end_indices": "pocc_end_indices",
        "pocc_time_products": "pocc_time_products",
    },
    writes=(
        "pocc_quality",
        "pocc_quality_criteria",
        "pocc_quality_results",
        "pocc_quality_flags",
    ),
    summary="Evaluate Pocc manoeuvre quality from the pressure upslope (Warnaar et al. 2024).",
    description="Evaluate Pocc manoeuvre validity from the pressure upslope shape against three configurable thresholds (Warnaar et al. 2024), producing one QualityFlag and three criterion measurements per manoeuvre.",
    category="quality",
    modality="emg",
    optional_packages=_RESURFEMG,
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle supplying pressure.",
        ),
        StepArtifact(
            name="pocc_indices",
            artifact_type="index_array",
            description="Pocc peak indices from 'emg.find_occluded_breaths'.",
        ),
        StepArtifact(
            name="pocc_end_indices",
            artifact_type="index_array",
            description="Pocc end indices from 'emg.pocc_intervals'.",
        ),
        StepArtifact(
            name="pocc_time_products",
            artifact_type="array",
            description="Pocc time products from 'emg.pocc_time_product'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="dp_up_10_threshold",
            value_type="number",
            default=0.0,
            description="Minimum acceptable dP at 10% of the upslope.",
        ),
        StepParameter(
            name="dp_up_90_threshold",
            value_type="number",
            default=2.0,
            description="Minimum acceptable dP at 90% of the upslope.",
        ),
        StepParameter(
            name="dp_up_90_norm_threshold",
            value_type="number",
            default=0.8,
            description="Minimum acceptable normalized dP at 90% of the upslope.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pocc_quality",
            artifact_type="boolean_array",
            description="Overall pass/fail per manoeuvre.",
        ),
        StepArtifact(
            name="pocc_quality_criteria",
            artifact_type="array",
            axes=("criterion", "manoeuvre"),
            description="Raw upstream 3-by-N criteria matrix.",
            compatibility_only=True,
        ),
        StepArtifact(
            name="pocc_quality_results",
            artifact_type="parameter_result_list",
            description="Native ParameterResult per manoeuvre per criterion (dp_up_10/dp_up_90/dp_up_90_norm).",
        ),
        StepArtifact(
            name="pocc_quality_flags",
            artifact_type="quality_flag_list",
            description="Native QualityFlag per manoeuvre.",
        ),
    ),
)
def pocc_quality(
    session: M3Session,
    ventilator_signals: Any,
    pocc_indices: Any,
    pocc_end_indices: Any,
    pocc_time_products: Any,
    *,
    dp_up_10_threshold: float = 0.0,
    dp_up_90_threshold: float = 2.0,
    dp_up_90_norm_threshold: float = 0.8,
) -> dict[str, Any]:
    pressure = np.asarray(ventilator_signals["pressure"], dtype=float)
    pressure_unit = ventilator_signals.get("unit") or "cmH2O"

    valid, criteria = session.emg_adapter.pocc_quality(
        pressure,
        pocc_indices,
        pocc_end_indices,
        pocc_time_products,
        dp_up_10_threshold=dp_up_10_threshold,
        dp_up_90_threshold=dp_up_90_threshold,
        dp_up_90_norm_threshold=dp_up_90_norm_threshold,
    )

    thresholds_by_row = {
        "dp_up_10": dp_up_10_threshold,
        "dp_up_90": dp_up_90_threshold,
        "dp_up_90_norm": dp_up_90_norm_threshold,
    }
    flags = _per_breath_flags(
        "pocc_quality",
        valid,
        modality="pressure",
        peak_indices=pocc_indices,
        extra_metadata={"pressure_sample_index_end": None},
    )
    # Link each flag to its Pocc end index too, not just its peak.
    for flag, end_index in zip(flags, pocc_end_indices):
        flag.metadata["pressure_sample_index_end"] = int(end_index)

    results: list[ParameterResult] = []
    for row_name, row_values in zip(_POCC_CRITERIA_ROW_NAMES, criteria):
        results.extend(
            _per_breath_results(
                f"pocc_quality_{row_name}",
                row_values,
                modality="pressure",
                peak_indices=pocc_indices,
                unit=pressure_unit,
                method="resurfemg.pocc_quality",
                extra_metadata_per_item=[
                    {"threshold": thresholds_by_row[row_name], "criterion": row_name}
                    for _ in row_values
                ],
            )
        )

    for result in results:
        session.parameter_results.add(result)
    for flag in flags:
        session.quality.add(flag)

    _record_step(
        session,
        "emg.pocc_quality",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.pocc_quality",
            operation="emg.pocc_quality",
            parameters={
                "dp_up_10_threshold": dp_up_10_threshold,
                "dp_up_90_threshold": dp_up_90_threshold,
                "dp_up_90_norm_threshold": dp_up_90_norm_threshold,
            },
        ),
    )
    return {
        "pocc_quality": valid,
        "pocc_quality_criteria": criteria,
        "pocc_quality_results": results,
        "pocc_quality_flags": flags,
    }
