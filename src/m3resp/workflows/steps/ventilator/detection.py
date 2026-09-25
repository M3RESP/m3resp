"""Registered ventilator breath and Pocc event-detection pipeline steps."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.events import BreathEvent
from m3resp.core.exceptions import MissingModalityDataError
from m3resp.core.session import M3Session
from m3resp.data import ParameterResult
from m3resp.processing.intervals import (
    onoff_from_baseline_crossings,
)
from m3resp.processing.metrics import (
    area_under_baseline,
    window_integral,
)
from m3resp.processing.peaks import (
    detect_occluded_breath_peaks,
    detect_pressure_dip_breaths,
    detect_ventilator_breath_peaks,
)
from m3resp.processing.ventilator import estimate_peep
from m3resp.workflows.registry import StepArtifact, StepParameter, register_step

from ._shared import (
    _RESURFEMG,
    _SESSION_ARTIFACT,
    _per_breath_flags,
    _per_breath_results,
    _record_step,
    _upstream_metadata,
)


def _resolve_peep(ventilator_signals: Any, pressure: Any, peep: float | None) -> float:
    """PEEP to measure against: the caller's value, else the end-expiratory
    estimate (Warnaar et al. 2024). Never falls back to a whole-trace
    statistic, which would sit above PEEP and bias every derived threshold."""

    if peep is not None:
        return float(peep)
    volume = (
        ventilator_signals.get("volume")
        if isinstance(ventilator_signals, dict)
        else None
    )
    if volume is None:
        raise MissingModalityDataError(
            "PEEP is estimated from the end-expiratory minima of the "
            "ventilator volume signal, which is not present here. Provide "
            "the volume channel or pass an explicit `peep` value."
        )
    return estimate_peep(pressure, volume)


def _pressure_baseline(
    session: M3Session,
    pressure: np.ndarray,
    fs: float,
    *,
    window_seconds: float,
    step_seconds: float,
    percentile: float,
) -> np.ndarray:
    """Moving baseline of the airway pressure, as ReSurfEMG takes it for Pocc
    on/offset and time products: the same moving percentile as the EMG
    baseline, run on the raw pressure. It follows the measured
    end-expiratory level where the set PEEP would stay flat."""

    return session.emg_adapter.moving_baseline(
        pressure,
        window_samples=max(1, int(window_seconds * fs)),
        step_samples=max(1, int(step_seconds * fs)),
        percentile=percentile,
    )


@register_step(
    "ventilator.detect_breaths",
    aliases=("emg.detect_ventilator_breath",),
    reads={"ventilator_signals": "ventilator_signals"},
    writes=("ventilator_breath_indices",),
    summary="Detect ventilator breaths from the ventilator volume channel.",
    description="Detect ventilator breath peaks from the volume channel.",
    category="detection",
    modality="ventilator",
    input_artifacts=(
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle from 'ventilator.channels'.",
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
def detect_breaths(
    ventilator_signals: Any, *, breath_width_seconds: float = 0.25
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
    "ventilator.detect_pressure_breaths",
    reads={"ventilator_signals": "ventilator_signals"},
    writes=("ventilator_breath_indices",),
    summary="Detect spontaneous breaths from dips in the airway pressure.",
    description="Detect spontaneous breaths as dips in the airway pressure (breathing in through a mouthpiece or mask lowers it), for recordings without a ventilator volume channel. Not for mechanically ventilated breaths.",
    category="detection",
    modality="ventilator",
    input_artifacts=(
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle from 'ventilator.channels' with a 'pressure' channel.",
        ),
    ),
    parameters=(
        StepParameter(
            name="smoothing_seconds",
            value_type="number",
            default=0.2,
            unit="s",
            minimum=0,
            description="Moving-average window that removes fast noise before dips are found.",
        ),
        StepParameter(
            name="min_depth",
            value_type="number",
            default=0.15,
            minimum=0,
            description="Smallest dip that counts as a breath, in the pressure's own unit (e.g. cmH2O).",
        ),
        StepParameter(
            name="min_interval_seconds",
            value_type="number",
            default=2.0,
            unit="s",
            minimum=0,
            description="Shortest time between two breaths (2 s allows up to 30 breaths/min).",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="ventilator_breath_indices",
            artifact_type="index_array",
            description="Sample index of the lowest point of each pressure dip.",
        ),
    ),
)
def detect_pressure_breaths(
    ventilator_signals: Any,
    *,
    smoothing_seconds: float = 0.2,
    min_depth: float = 0.15,
    min_interval_seconds: float = 2.0,
) -> dict[str, Any]:
    pressure = ventilator_signals.get("pressure")
    if pressure is None:
        raise MissingModalityDataError(
            "ventilator.detect_pressure_breaths needs a 'pressure' channel; ask "
            "ventilator.channels for it (e.g. pressure_channel=0)."
        )
    indices = detect_pressure_dip_breaths(
        pressure,
        sample_frequency=float(ventilator_signals["fs"]),
        smoothing_seconds=smoothing_seconds,
        min_depth=min_depth,
        min_interval_seconds=min_interval_seconds,
    )
    return {"ventilator_breath_indices": np.asarray(indices, dtype=int)}


@register_step(
    "ventilator.find_occluded_breaths",
    aliases=("emg.find_occluded_breaths",),
    reads={"ventilator_signals": "ventilator_signals"},
    writes=("pocc_indices",),
    summary="Detect occluded (Pocc) breaths from the ventilator pressure channel.",
    description="Detect occlusion (Pocc) manoeuvre peaks from the pressure channel.",
    category="detection",
    modality="ventilator",
    input_artifacts=(
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle from 'ventilator.channels'.",
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
    peep = _resolve_peep(ventilator_signals, pressure, peep)
    indices = detect_occluded_breath_peaks(
        pressure,
        sample_frequency=fs,
        peep=peep,
    )
    return {"pocc_indices": np.asarray(indices, dtype=int)}


@register_step(
    "ventilator.pocc_intervals",
    aliases=("emg.pocc_intervals",),
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
        "pressure_baseline",
    ),
    summary="Find Pocc manoeuvre start/end indices from the pressure channel.",
    description="Find Pocc manoeuvre start/end indices around each detected peak where the pressure crosses its moving baseline, and record BreathEvents.",
    category="detection",
    modality="ventilator",
    optional_packages=_RESURFEMG,
    session_writes=("session.events.pocc_breaths",),
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle from 'ventilator.channels'.",
        ),
        StepArtifact(
            name="pocc_indices",
            artifact_type="index_array",
            description="Pocc peak indices from 'ventilator.find_occluded_breaths'.",
        ),
    ),
    parameters=(
        StepParameter(
            name="baseline_window_seconds",
            value_type="number",
            default=7.5,
            unit="s",
            minimum=0,
            description="Moving-baseline window length on the airway pressure.",
        ),
        StepParameter(
            name="baseline_step_seconds",
            value_type="number",
            default=0.2,
            unit="s",
            minimum=0,
            description="Step between successive moving-baseline windows.",
        ),
        StepParameter(
            name="baseline_percentile",
            value_type="number",
            default=33.0,
            minimum=0,
            maximum=100,
            description="Percentile of the airway pressure taken within each window.",
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
        StepArtifact(
            name="pressure_baseline",
            artifact_type="signal_array",
            unit="cmH2O",
            description="Moving baseline of the airway pressure, one value per sample.",
        ),
    ),
)
def pocc_intervals(
    session: M3Session,
    ventilator_signals: Any,
    pocc_indices: Any,
    *,
    baseline_window_seconds: float = 7.5,
    baseline_step_seconds: float = 0.2,
    baseline_percentile: float = 33.0,
) -> dict[str, Any]:
    pressure = np.asarray(ventilator_signals["pressure"], dtype=float)
    fs = float(ventilator_signals["fs"])
    peaks = np.asarray(pocc_indices, dtype=int)

    # PEEP locates the occlusions (ventilator.find_occluded_breaths); their
    # start and end are where pressure crosses its moving baseline.
    baseline = _pressure_baseline(
        session,
        pressure,
        fs,
        window_seconds=baseline_window_seconds,
        step_seconds=baseline_step_seconds,
        percentile=baseline_percentile,
    )

    starts, ends, valid_starts, valid_ends, valid_peaks = onoff_from_baseline_crossings(
        pressure, baseline, peaks
    )

    events: list[BreathEvent] = []
    for index, peak in enumerate(peaks):
        events.append(
            BreathEvent(
                modality="ventilator",
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
                    "valid": bool(valid_peaks[index]),
                    "valid_start": bool(valid_starts[index]),
                    "valid_end": bool(valid_ends[index]),
                },
            )
        )
    session.add_events("pocc_breaths", events)

    _record_step(
        session,
        "ventilator.pocc_intervals",
        metadata=_upstream_metadata(
            source_function="m3resp.processing.intervals.onoff_from_baseline_crossings",
            operation="ventilator.pocc_intervals",
            parameters={
                "baseline_window_seconds": baseline_window_seconds,
                "baseline_step_seconds": baseline_step_seconds,
                "baseline_percentile": baseline_percentile,
            },
            source_package="m3resp",
            implementation="m3resp.processing.intervals",
        ),
    )
    return {
        "pocc_start_indices": starts,
        "pocc_end_indices": ends,
        "pocc_interval_validity": np.asarray(valid_peaks, dtype=bool),
        "pocc_events": events,
        "pressure_baseline": baseline,
    }


@register_step(
    "ventilator.pocc_time_product",
    aliases=("emg.pocc_time_product",),
    reads={
        "session": "session",
        "ventilator_signals": "ventilator_signals",
        "pocc_start_indices": "pocc_start_indices",
        "pocc_end_indices": "pocc_end_indices",
        "pressure_baseline": "pressure_baseline",
        "pocc_indices": "pocc_indices",
    },
    writes=("pocc_time_products", "pocc_time_product_result"),
    summary="Compute the pressure-time product for each Pocc manoeuvre.",
    description="Integrate pressure against its moving baseline over each Pocc manoeuvre's start/end window, plus the area under the baseline (ReSurfEMG PTPocc).",
    category="parameters",
    modality="ventilator",
    input_artifacts=(
        _SESSION_ARTIFACT,
        StepArtifact(
            name="ventilator_signals",
            artifact_type="ventilator_channel_bundle",
            description="Ventilator channel bundle from 'ventilator.channels'.",
        ),
        StepArtifact(
            name="pocc_start_indices",
            artifact_type="index_array",
            description="Pocc manoeuvre start indices from 'ventilator.pocc_intervals'.",
        ),
        StepArtifact(
            name="pocc_end_indices",
            artifact_type="index_array",
            description="Pocc manoeuvre end indices from 'ventilator.pocc_intervals'.",
        ),
        StepArtifact(
            name="pressure_baseline",
            artifact_type="signal_array",
            unit="cmH2O",
            description="Moving pressure baseline from 'ventilator.pocc_intervals', so both steps measure against the same baseline.",
        ),
        StepArtifact(
            name="pocc_indices",
            artifact_type="index_array",
            description="Pocc peak indices from 'ventilator.find_occluded_breaths', around which the area under the baseline is sought.",
        ),
    ),
    parameters=(
        StepParameter(
            name="include_aub",
            value_type="boolean",
            default=True,
            description="Add the area under the baseline to the time product, as ReSurfEMG's PTPocc does.",
        ),
        StepParameter(
            name="aub_window_seconds",
            value_type="number",
            default=5.0,
            unit="s",
            minimum=0,
            description="Half-width of the window around each Pocc peak in which the highest baseline value is taken as the reference for the area under the baseline.",
        ),
    ),
    output_artifacts=(
        StepArtifact(
            name="pocc_time_products",
            artifact_type="array",
            unit="cmH2O*s",
            description="Pressure-time product per Pocc manoeuvre, including manoeuvres marked invalid in 'pocc_interval_validity' (their window may overlap a neighbouring manoeuvre).",
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
    pressure_baseline: Any,
    *,
    pocc_indices: Any = None,
    include_aub: bool = True,
    aub_window_seconds: float = 5.0,
) -> dict[str, Any]:
    pressure = np.asarray(ventilator_signals["pressure"], dtype=float)
    fs = float(ventilator_signals["fs"])
    baseline = np.asarray(pressure_baseline, dtype=float)

    time_products = window_integral(
        pressure, fs, pocc_start_indices, pocc_end_indices, baseline
    )
    aub = None
    if include_aub:
        if pocc_indices is None:
            raise ValueError(
                "The area under the baseline is sought around each Pocc peak; "
                "pass `pocc_indices` or set include_aub=False."
            )
        # ReSurfEMG PTPocc (calculate_time_products with the pressure
        # baseline as reference): adds the area between the baseline and its
        # highest value near the occlusion, so a baseline that dips during
        # the manoeuvre does not shrink the product.
        aub, _ = area_under_baseline(
            pressure,
            fs,
            pocc_indices,
            pocc_start_indices,
            pocc_end_indices,
            window=int(aub_window_seconds * fs),
            baseline=baseline,
            reference_values=baseline,
        )
        time_products = time_products + aub

    pressure_unit = ventilator_signals.get("unit") or "cmH2O"
    parameters = {
        "include_aub": include_aub,
        "aub_window_seconds": aub_window_seconds,
    }
    result = ParameterResult(
        name="pocc_time_product",
        value=time_products,
        modality="ventilator",
        category="airway_pressure",
        unit=f"{pressure_unit}*s",
        method="m3resp.processing.metrics.window_integral",
        metadata={
            **parameters,
            "area_under_baseline": None if aub is None else aub.tolist(),
            "start_indices": np.asarray(pocc_start_indices, dtype=int).tolist(),
            "end_indices": np.asarray(pocc_end_indices, dtype=int).tolist(),
        },
    )

    _record_step(
        session,
        "ventilator.pocc_time_product",
        metadata=_upstream_metadata(
            source_function="m3resp.processing.metrics.window_integral",
            operation="ventilator.pocc_time_product",
            parameters=parameters,
            source_package="m3resp",
            implementation="m3resp.processing.metrics",
        ),
    )
    return {"pocc_time_products": time_products, "pocc_time_product_result": result}


_POCC_CRITERIA_ROW_NAMES = ("dp_up_10", "dp_up_90", "dp_up_90_norm")


@register_step(
    "ventilator.pocc_quality",
    aliases=("emg.pocc_quality",),
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
    modality="ventilator",
    optional_packages=_RESURFEMG,
    session_writes=("session.quality", "session.parameter_results"),
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
            description="Pocc peak indices from 'ventilator.find_occluded_breaths'.",
        ),
        StepArtifact(
            name="pocc_end_indices",
            artifact_type="index_array",
            description="Pocc end indices from 'ventilator.pocc_intervals'.",
        ),
        StepArtifact(
            name="pocc_time_products",
            artifact_type="array",
            description="Pocc time products from 'ventilator.pocc_time_product'.",
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
        modality="ventilator",
        category="airway_pressure",
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
                modality="ventilator",
                category="airway_pressure",
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
        "ventilator.pocc_quality",
        metadata=_upstream_metadata(
            source_function="resurfemg.postprocessing.quality_assessment.pocc_quality",
            operation="ventilator.pocc_quality",
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
