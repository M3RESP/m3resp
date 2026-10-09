"""Registered ventilator breath and Pocc event-detection workflow steps."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.exceptions import MissingModalityDataError
from m3resp.core.session import M3Session
from m3resp.data import ParameterResult
from m3resp.data.events import BreathEvent
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
from m3resp.workflows.steps._per_breath import _per_breath_flags, _per_breath_results

from ._shared import (
    _RESURFEMG,
    _SESSION_ARTIFACT,
    _airway_pressure,
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
            description="Ventilator channel bundle from 'ventilator.channels' with an 'airway_pressure' channel.",
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
    """Find spontaneous breaths as dips in airway pressure.

    Args:
        ventilator_signals (Any): Channel bundle containing the main airway
            pressure and fs in Hz. Qualified primary channel keys are supported.
        smoothing_seconds (float): Moving-average window in seconds. Defaults
            to 0.2.
        min_depth (float): Minimum dip prominence in the pressure's recorded unit.
            Defaults to 0.15.
        min_interval_seconds (float): Minimum separation between detections in
            seconds. Defaults to 2.0.

    Returns:
        dict[str, Any]: ventilator_breath_indices, an integer array of pressure
            minima on the smoothed signal. Missing samples are bridged for
            smoothing with a warning; detections on missing samples are excluded.

    Raises:
        MissingModalityDataError: If airway pressure is unavailable.
        ValueError: If every pressure sample is missing.
    """

    pressure, _ = _airway_pressure(
        ventilator_signals, "ventilator.detect_pressure_breaths"
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
            description="PEEP baseline. When unset, PEEP is estimated from the airway pressure at the end of each breath out, found from the volume channel (Warnaar et al. 2024), so the volume channel is then needed.",
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
    """Detect occluded inspiratory efforts from airway-pressure dips.

    Args:
        ventilator_signals (Any): Channel bundle containing airway pressure and
            fs in Hz. Volume is also required when peep is None.
        peep (float | None): PEEP in the pressure's recorded unit. None estimates
            PEEP from airway pressure at end-expiratory volume minima.

    Returns:
        dict[str, Any]: pocc_indices, an integer array with one pressure-minimum
            sample index per detected occlusion manoeuvre.

    Raises:
        MissingModalityDataError: If airway pressure is missing, or volume is
            missing when PEEP must be estimated.
    """

    import numpy as np

    pressure, _ = _airway_pressure(
        ventilator_signals, "ventilator.find_occluded_breaths"
    )
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
    """Find the start and end of each occluded breath from airway-pressure crossings.

    Boundaries are found from moving-baseline crossings around each pressure
    minimum, using the signal edges when crossings are missing. Breaths are stored
    in ``session.events['pocc_breaths']`` with ``metadata['event_type']`` set
    to ``'pocc'``. Their times are in seconds from the first pressure sample,
    and their extremum is the pressure minimum. Every detected minimum is
    kept, with validity flags for its boundaries.
    The processing settings are recorded in the session's history.

    Args:
        session (M3Session): Session whose EMG adapter computes the pressure
            baseline and which stores the occluded breaths and processing
            history.
        ventilator_signals (Any): Channel bundle containing the airway pressure
            and its sampling rate ``fs`` in Hz. Baseline values use the
            pressure unit.
        pocc_indices (Any): One-dimensional sample indices of the occlusion
            pressure minima, in the pressure signal.
        baseline_window_seconds (float): Moving-baseline window length in
            seconds. Defaults to 7.5.
        baseline_step_seconds (float): Step between baseline windows in
            seconds. Defaults to 0.2.
        baseline_percentile (float): Pressure percentile within each window,
            from 0 to 100. Defaults to 33.0.

    Returns:
        dict[str, Any]: ``pocc_start_indices`` and ``pocc_end_indices`` in
            samples, ``pocc_interval_validity`` as a boolean array, and
            ``pocc_events`` as ``BreathEvent`` objects, all in
            ``pocc_indices`` order. ``pressure_baseline`` has one value per
            pressure sample.

    Raises:
        MissingModalityDataError: If airway pressure is unavailable.
        OptionalDependencyError: If ReSurfEMG is unavailable for baseline
            estimation.
        ValueError: If baseline inputs are invalid or a breath ends before
            its start.
    """

    pressure, _ = _airway_pressure(ventilator_signals, "ventilator.pocc_intervals")
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

    # An occluded breath is an effort to breathe in against a closed airway.
    # Its turning point is the deepest pressure, at `extremum_index`.
    events: list[BreathEvent] = []
    for index, peak in enumerate(peaks):
        events.append(
            BreathEvent(
                modality="ventilator",
                start_time=float(starts[index]) / fs,
                end_time=float(ends[index]) / fs,
                extremum_time=float(peak) / fs,
                start_index=int(starts[index]),
                extremum_index=int(peak),
                end_index=int(ends[index]),
                sample_frequency=fs,
                signal_name="airway_pressure",
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
    """Compute the pressure-time product for each occlusion window.

    Takes the absolute integral of airway pressure minus the supplied baseline,
    including both boundary samples, and optionally adds the area under that
    baseline, following ReSurfEMG's PTPocc calculation.
    Every supplied window is evaluated, including windows marked invalid by
    pocc_intervals. Review pocc_interval_validity when selecting results.

    Args:
        session (M3Session): Session receiving step provenance.
        ventilator_signals (Any): Channel bundle containing airway pressure and
            fs in Hz.
        pocc_start_indices (Any): Start sample indices, one per manoeuvre.
        pocc_end_indices (Any): End sample indices in the same order.
        pressure_baseline (Any): Baseline array aligned with pressure samples,
            in the same pressure unit.
        pocc_indices (Any): Pressure-minimum sample indices, required when
            include_aub is True.
        include_aub (bool): Add the area under the baseline. Defaults to True.
        aub_window_seconds (float): Half-width in seconds around each pressure
            minimum for finding the highest baseline reference. Defaults to 5.0.

    Returns:
        dict[str, Any]: pocc_time_products array and its array-valued
            pocc_time_product_result. The result unit is the recorded pressure
            unit times seconds, such as "cmH2O*s". The returned ParameterResult
            includes window indices and the area-under-baseline contribution.

    Raises:
        MissingModalityDataError: If airway pressure is unavailable.
        ValueError: If include_aub is True and pocc_indices is missing.
    """

    pressure, pressure_unit = _airway_pressure(
        ventilator_signals, "ventilator.pocc_time_product"
    )
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
            description="Minimum acceptable dP at 10% of the upslope, in the pressure's own unit (the default is for cmH2O).",
        ),
        StepParameter(
            name="dp_up_90_threshold",
            value_type="number",
            default=2.0,
            description="Minimum acceptable dP at 90% of the upslope, in the pressure's own unit (the default is for cmH2O).",
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
    """Assess occlusion quality from airway-pressure recovery after each dip.

    Uses the ReSurfEMG implementation of the Warnaar et al. (2024) criteria.
    Adds criterion results, quality flags and step provenance to the session.

    Args:
        session (M3Session): Supplies the quality calculation and stores results.
        ventilator_signals (Any): Channel bundle with the main airway pressure
            and its recorded unit.
        pocc_indices (Any): Pressure-minimum sample indices, one per manoeuvre.
        pocc_end_indices (Any): Corresponding manoeuvre end sample indices.
        pocc_time_products (Any): Corresponding pressure-time products.
        dp_up_10_threshold (float): Threshold at 10% of the pressure upslope, in
            the pressure's unit. Defaults to 0.0.
        dp_up_90_threshold (float): Threshold at 90% of the pressure upslope, in
            the pressure's unit. Defaults to 2.0, chosen for cmH2O recordings.
        dp_up_90_norm_threshold (float): Normalized upslope threshold.
            Defaults to 0.8. Set pressure thresholds for the recorded unit.

    Returns:
        dict[str, Any]: pocc_quality pass/fail array, pocc_quality_criteria matrix
            shaped (3, manoeuvres), pocc_quality_results with three measurements
            per manoeuvre, and pocc_quality_flags with one flag per manoeuvre.
            Each result and flag records its pressure-minimum sample index.

    Raises:
        MissingModalityDataError: If airway pressure is unavailable.
        ValueError: If per-breath validity or criterion values differ in length
            from pocc_indices.
    """

    pressure, pressure_unit = _airway_pressure(
        ventilator_signals, "ventilator.pocc_quality"
    )

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
        extremum_indices=pocc_indices,
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
                extremum_indices=pocc_indices,
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
