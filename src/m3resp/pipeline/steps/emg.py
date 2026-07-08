"""Registered EMG pipeline steps.

Loading/preprocessing steps wrap the ``M3Session`` EMG stage methods.
Postprocessing steps each wrap a single ``resurfemg.postprocessing`` function,
mirroring the one-step-per-operation structure used by ``eit.py``. Upstream
imports are deferred to call time so the package installs without the
optional ``resurfemg`` dependency.
"""

from __future__ import annotations

from typing import Any

from m3resp.adapters.resurfemg_adapter import (
    _peak_indices_from_events,
    _ventilator_signals,
)
from m3resp.core.session import (
    M3Session,
    _iter_ventilator_detections,
    _normalize_ventilator_breath,
)
from m3resp.pipeline.registry import register_step


@register_step(
    "emg.load",
    reads={"session": "session"},
    writes=(),
    summary="Load an EMG recording into the session.",
)
def load(session: M3Session, *, file: str) -> dict[str, Any]:
    session.load_emg(file, verbose=False)
    return {}


@register_step(
    "emg.load_ventilator",
    reads={"session": "session"},
    writes=("ventilator_raw",),
    summary="Load a ventilator recording into the session.",
)
def load_ventilator(session: M3Session, *, file: str) -> dict[str, Any]:
    recording = session.emg_adapter.load(str(file), verbose=False)
    # Stored on the session too so `session.sync_raw` (which crops
    # `session.raw["vent"]` in place) keeps this same dict object in sync.
    session.raw["vent"] = recording
    return {"ventilator_raw": recording}


@register_step(
    "emg.ventilator_channels",
    reads={"ventilator_raw": "ventilator_raw"},
    writes=("ventilator_signals",),
    summary="Split a raw ventilator recording into pressure/flow/volume channels.",
)
def ventilator_channels(
    ventilator_raw: Any,
    *,
    pressure_channel: int = 0,
    flow_channel: int = 1,
    volume_channel: int = 2,
    fs: float | None = None,
) -> dict[str, Any]:
    signals = _ventilator_signals(
        ventilator_raw,
        pressure_channel=pressure_channel,
        flow_channel=flow_channel,
        volume_channel=volume_channel,
        fs=fs,
    )
    return {"ventilator_signals": signals}


@register_step(
    "emg.preprocess",
    reads={"session": "session"},
    writes=("processed_emg",),
    summary="Filter EMG and compute its envelope.",
)
def preprocess(session: M3Session, **kwargs: Any) -> dict[str, Any]:
    return {"processed_emg": session.preprocess_emg(**kwargs)}


@register_step(
    "emg.detect_breaths",
    reads={"session": "session"},
    writes=("emg_breath_events",),
    summary="Detect EMG breaths from the envelope.",
)
def detect_breaths(session: M3Session, **kwargs: Any) -> dict[str, Any]:
    events = session.detect_emg_breaths(**kwargs)
    return {"emg_breath_events": events}


@register_step(
    "emg.peak_indices",
    reads={"events": "emg_breath_events", "processed_emg": "processed_emg"},
    writes=("peak_indices",),
    summary="Derive EMG breath peak sample indices from detected breath events.",
)
def peak_indices(events: Any, processed_emg: Any) -> dict[str, Any]:
    import numpy as np

    fs = float(processed_emg["fs"])
    return {
        "peak_indices": np.asarray(_peak_indices_from_events(events, fs), dtype=int)
    }


# --- baseline -----------------------------------------------------------
#
# `emg.moving_baseline` and `emg.slopesum_baseline` both write `baseline`.
# A pipeline picks exactly one (or renames one via `out:`) - this makes the
# baseline choice an explicit YAML decision instead of the previous silent
# "whichever ran last wins when both are enabled" behavior.


@register_step(
    "emg.moving_baseline",
    reads={"processed_emg": "processed_emg"},
    writes=("baseline",),
    summary="Compute a moving-percentile EMG baseline.",
)
def moving_baseline(
    processed_emg: Any,
    *,
    window_seconds: float = 30.0,
    step_seconds: float = 1.0,
    percentile: float = 33.0,
) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.baseline import moving_baseline as _moving_baseline

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    window_samples = max(1, int(window_seconds * fs))
    step_samples = max(1, int(step_seconds * fs))
    baseline = _moving_baseline(
        envelope, window_samples, step_samples, set_percentile=percentile
    )
    return {"baseline": baseline}


@register_step(
    "emg.slopesum_baseline",
    reads={"processed_emg": "processed_emg"},
    writes=("baseline", "slopesum_baseline_detail"),
    summary="Compute a slope-sum EMG baseline.",
)
def slopesum_baseline(
    processed_emg: Any,
    *,
    window_seconds: float = 30.0,
    step_seconds: float = 1.0,
    percentile: float = 33.0,
) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.baseline import (
        slopesum_baseline as _slopesum_baseline,
    )

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    window_samples = max(1, int(window_seconds * fs))
    step_samples = max(1, int(step_seconds * fs))
    result = _slopesum_baseline(
        envelope,
        window_samples,
        step_samples,
        fs,
        set_percentile=percentile,
        ma_window=max(1, int(fs // 2)),
        perc_window=max(1, int(fs)),
    )
    return {
        "baseline": result[0],
        "slopesum_baseline_detail": {
            "running_mean": result[1],
            "running_std": result[2],
            "series": result[3],
        },
    }


# --- event_detection ------------------------------------------------------


@register_step(
    "emg.detect_ventilator_breath",
    reads={"ventilator_signals": "ventilator_signals"},
    writes=("ventilator_breath_indices",),
    summary="Detect ventilator breaths from the ventilator volume channel.",
)
def detect_ventilator_breath(
    ventilator_signals: Any, *, breath_width_seconds: float = 0.5
) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.event_detection import (
        detect_ventilator_breath as _detect_ventilator_breath,
    )

    volume = ventilator_signals["volume"]
    fs = float(ventilator_signals["fs"])
    width_samples = max(1, int(breath_width_seconds * fs))
    indices = _detect_ventilator_breath(volume, 0, len(volume) - 1, width_samples)
    return {"ventilator_breath_indices": np.asarray(indices, dtype=int)}


@register_step(
    "emg.find_occluded_breaths",
    reads={"ventilator_signals": "ventilator_signals"},
    writes=("pocc_indices",),
    summary="Detect occluded (Pocc) breaths from the ventilator pressure channel.",
)
def find_occluded_breaths(
    ventilator_signals: Any, *, peep: float | None = None
) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.event_detection import (
        find_occluded_breaths as _find_occluded_breaths,
    )

    pressure = ventilator_signals["pressure"]
    fs = float(ventilator_signals["fs"])
    if peep is None:
        peep = float(np.nanmedian(pressure))
    indices = _find_occluded_breaths(pressure, fs, peep)
    return {"pocc_indices": np.asarray(indices, dtype=int)}


@register_step(
    "emg.onoffpeak_baseline_crossing",
    reads={
        "processed_emg": "processed_emg",
        "baseline": "baseline",
        "peak_indices": "peak_indices",
    },
    writes=("start_indices", "end_indices"),
    summary="Find EMG breath on/offset indices by baseline crossing.",
)
def onoffpeak_baseline_crossing(
    processed_emg: Any, baseline: Any, peak_indices: Any
) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.event_detection import (
        onoffpeak_baseline_crossing as _onoffpeak_baseline_crossing,
    )

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    start_indices, end_indices, *_rest = _onoffpeak_baseline_crossing(
        envelope, baseline, peak_indices
    )
    return {"start_indices": start_indices, "end_indices": end_indices}


@register_step(
    "emg.onoffpeak_slope_extrapolation",
    reads={"processed_emg": "processed_emg", "peak_indices": "peak_indices"},
    writes=("onoffpeak_slope_result",),
    summary="Find EMG breath on/offset indices by slope extrapolation.",
)
def onoffpeak_slope_extrapolation(
    processed_emg: Any, peak_indices: Any, *, slope_window_seconds: float = 0.5
) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.event_detection import (
        onoffpeak_slope_extrapolation as _onoffpeak_slope_extrapolation,
    )

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    slope_window_samples = max(1, int(slope_window_seconds * fs))
    result = _onoffpeak_slope_extrapolation(
        envelope, fs, peak_indices, slope_window_samples
    )
    return {"onoffpeak_slope_result": result}


# --- features ---------------------------------------------------------


@register_step(
    "emg.time_to_peak",
    reads={
        "processed_emg": "processed_emg",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
    },
    writes=("time_to_peak",),
    summary="Compute EMG breath time-to-peak.",
)
def time_to_peak(
    processed_emg: Any, start_indices: Any, end_indices: Any
) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.features import time_to_peak as _time_to_peak

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    return {"time_to_peak": _time_to_peak(envelope, start_indices, end_indices)}


@register_step(
    "emg.pseudo_slope",
    reads={
        "processed_emg": "processed_emg",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
    },
    writes=("pseudo_slope",),
    summary="Compute EMG breath pseudo-slope.",
)
def pseudo_slope(
    processed_emg: Any, start_indices: Any, end_indices: Any
) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.features import pseudo_slope as _pseudo_slope

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    return {"pseudo_slope": _pseudo_slope(envelope, start_indices, end_indices)}


@register_step(
    "emg.amplitude",
    reads={
        "processed_emg": "processed_emg",
        "peak_indices": "peak_indices",
        "baseline": "baseline",
    },
    writes=("amplitude",),
    summary="Compute EMG breath amplitude above baseline.",
)
def amplitude(processed_emg: Any, peak_indices: Any, baseline: Any) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.features import amplitude as _amplitude

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    return {"amplitude": _amplitude(envelope, peak_indices, baseline)}


@register_step(
    "emg.time_product",
    reads={
        "processed_emg": "processed_emg",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "baseline": "baseline",
    },
    writes=("time_product",),
    summary="Compute EMG breath time-product (area above baseline).",
)
def time_product(
    processed_emg: Any, start_indices: Any, end_indices: Any, baseline: Any
) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.features import time_product as _time_product

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    return {
        "time_product": _time_product(
            envelope, fs, start_indices, end_indices, baseline
        )
    }


@register_step(
    "emg.area_under_baseline",
    reads={
        "processed_emg": "processed_emg",
        "peak_indices": "peak_indices",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "baseline": "baseline",
    },
    writes=("area_under_baseline",),
    summary="Compute EMG area under baseline around each breath peak.",
)
def area_under_baseline(
    processed_emg: Any,
    peak_indices: Any,
    start_indices: Any,
    end_indices: Any,
    baseline: Any,
    *,
    window_seconds: float = 5.0,
) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.features import (
        area_under_baseline as _area_under_baseline,
    )

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    window_samples = max(1, int(window_seconds * fs))
    result = _area_under_baseline(
        envelope,
        fs,
        peak_indices,
        start_indices,
        end_indices,
        window_samples,
        baseline,
    )
    return {"area_under_baseline": result}


@register_step(
    "emg.respiratory_rate",
    reads={"peak_indices": "peak_indices", "processed_emg": "processed_emg"},
    writes=("respiratory_rate",),
    summary="Compute respiratory rate from detected EMG breath peaks.",
)
def respiratory_rate(peak_indices: Any, processed_emg: Any) -> dict[str, Any]:
    from resurfemg.postprocessing.features import (
        respiratory_rate as _respiratory_rate,
    )

    fs = float(processed_emg["fs"])
    return {"respiratory_rate": _respiratory_rate(peak_indices, fs)}


@register_step(
    "emg.ventilator_respiratory_rate",
    reads={
        "ventilator_breath_indices": "ventilator_breath_indices",
        "ventilator_signals": "ventilator_signals",
    },
    writes=("ventilator_respiratory_rate",),
    summary="Compute respiratory rate from detected ventilator breaths.",
)
def ventilator_respiratory_rate(
    ventilator_breath_indices: Any, ventilator_signals: Any
) -> dict[str, Any]:
    from resurfemg.postprocessing.features import (
        respiratory_rate as _respiratory_rate,
    )

    fs = float(ventilator_signals["fs"])
    return {
        "ventilator_respiratory_rate": _respiratory_rate(ventilator_breath_indices, fs)
    }


# --- quality_assessment -------------------------------------------------


@register_step(
    "emg.snr_pseudo",
    reads={
        "processed_emg": "processed_emg",
        "peak_indices": "peak_indices",
        "baseline": "baseline",
    },
    writes=("snr_pseudo",),
    summary="Compute a pseudo signal-to-noise ratio for detected EMG breaths.",
)
def snr_pseudo(processed_emg: Any, peak_indices: Any, baseline: Any) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.quality_assessment import (
        snr_pseudo as _snr_pseudo,
    )

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    return {"snr_pseudo": _snr_pseudo(envelope, peak_indices, baseline, fs)}


@register_step(
    "emg.percentage_under_baseline",
    reads={
        "processed_emg": "processed_emg",
        "peak_indices": "peak_indices",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "baseline": "baseline",
    },
    writes=("percentage_under_baseline",),
    summary="Compute the percentage of each EMG breath spent under baseline.",
)
def percentage_under_baseline(
    processed_emg: Any,
    peak_indices: Any,
    start_indices: Any,
    end_indices: Any,
    baseline: Any,
) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.quality_assessment import (
        percentage_under_baseline as _percentage_under_baseline,
    )

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    result = _percentage_under_baseline(
        envelope, fs, peak_indices, start_indices, end_indices, baseline
    )
    return {"percentage_under_baseline": result}


@register_step(
    "emg.detect_local_high_aub",
    reads={"area_under_baseline": "area_under_baseline"},
    writes=("detect_local_high_aub",),
    summary="Flag EMG breaths with locally elevated area-under-baseline.",
)
def detect_local_high_aub(area_under_baseline: Any) -> dict[str, Any]:
    from resurfemg.postprocessing.quality_assessment import (
        detect_local_high_aub as _detect_local_high_aub,
    )

    aubs = area_under_baseline[0]
    return {"detect_local_high_aub": _detect_local_high_aub(aubs)}


@register_step(
    "emg.detect_extreme_time_products",
    reads={"time_product": "time_product"},
    writes=("detect_extreme_time_products",),
    summary="Flag EMG breaths with extreme time-products.",
)
def detect_extreme_time_products(time_product: Any) -> dict[str, Any]:
    from resurfemg.postprocessing.quality_assessment import (
        detect_extreme_time_products as _detect_extreme_time_products,
    )

    return {"detect_extreme_time_products": _detect_extreme_time_products(time_product)}


@register_step(
    "emg.detect_non_consecutive_manoeuvres",
    reads={
        "ventilator_breath_indices": "ventilator_breath_indices",
        "pocc_indices": "pocc_indices",
    },
    writes=("detect_non_consecutive_manoeuvres",),
    summary="Flag non-consecutive occlusion manoeuvres against ventilator breaths.",
)
def detect_non_consecutive_manoeuvres(
    ventilator_breath_indices: Any, pocc_indices: Any
) -> dict[str, Any]:
    from resurfemg.postprocessing.quality_assessment import (
        detect_non_consecutive_manoeuvres as _detect_non_consecutive_manoeuvres,
    )

    result = _detect_non_consecutive_manoeuvres(ventilator_breath_indices, pocc_indices)
    return {"detect_non_consecutive_manoeuvres": result}


@register_step(
    "emg.evaluate_bell_curve_error",
    reads={
        "peak_indices": "peak_indices",
        "start_indices": "start_indices",
        "end_indices": "end_indices",
        "processed_emg": "processed_emg",
        "time_product": "time_product",
    },
    writes=("evaluate_bell_curve_error",),
    summary="Score how well each EMG breath matches a bell-curve shape.",
)
def evaluate_bell_curve_error(
    peak_indices: Any,
    start_indices: Any,
    end_indices: Any,
    processed_emg: Any,
    time_product: Any,
) -> dict[str, Any]:
    import numpy as np
    from resurfemg.postprocessing.quality_assessment import (
        evaluate_bell_curve_error as _evaluate_bell_curve_error,
    )

    envelope = np.asarray(processed_emg["envelope"], dtype=float)
    fs = float(processed_emg["fs"])
    result = _evaluate_bell_curve_error(
        peak_indices, start_indices, end_indices, envelope, fs, time_product
    )
    return {"evaluate_bell_curve_error": result}


@register_step(
    "emg.evaluate_event_timing",
    reads={
        "peak_indices": "peak_indices",
        "processed_emg": "processed_emg",
        "ventilator_breath_indices": "ventilator_breath_indices",
        "ventilator_signals": "ventilator_signals",
    },
    writes=("evaluate_event_timing",),
    summary="Score the timing agreement between EMG and ventilator breaths.",
)
def evaluate_event_timing(
    peak_indices: Any,
    processed_emg: Any,
    ventilator_breath_indices: Any,
    ventilator_signals: Any,
) -> dict[str, Any]:
    from resurfemg.postprocessing.quality_assessment import (
        evaluate_event_timing as _evaluate_event_timing,
    )

    fs = float(processed_emg["fs"])
    vent_fs = float(ventilator_signals["fs"])
    paired_count = min(len(peak_indices), len(ventilator_breath_indices))
    result = _evaluate_event_timing(
        peak_indices[:paired_count] / fs,
        ventilator_breath_indices[:paired_count] / vent_fs,
    )
    return {"evaluate_event_timing": result}


@register_step(
    "emg.evaluate_respiratory_rates",
    reads={
        "peak_indices": "peak_indices",
        "processed_emg": "processed_emg",
        "ventilator_respiratory_rate": "ventilator_respiratory_rate",
    },
    writes=("evaluate_respiratory_rates",),
    summary="Score agreement between EMG-derived and ventilator-derived respiratory rate.",
)
def evaluate_respiratory_rates(
    peak_indices: Any, processed_emg: Any, ventilator_respiratory_rate: Any
) -> dict[str, Any]:
    from resurfemg.postprocessing.quality_assessment import (
        evaluate_respiratory_rates as _evaluate_respiratory_rates,
    )

    fs = float(processed_emg["fs"])
    envelope = processed_emg["envelope"]
    rr_vent = ventilator_respiratory_rate[0]
    result = _evaluate_respiratory_rates(peak_indices, len(envelope) / fs, rr_vent)
    return {"evaluate_respiratory_rates": result}


# --- event normalization -------------------------------------------------


@register_step(
    "emg.normalize_ventilator_breaths",
    reads={
        "ventilator_breath_indices": "ventilator_breath_indices",
        "ventilator_signals": "ventilator_signals",
        "session": "session",
    },
    writes=(),
    summary="Normalize detected ventilator breath indices into session events.",
)
def normalize_ventilator_breaths(
    ventilator_breath_indices: Any,
    ventilator_signals: Any,
    session: M3Session,
    *,
    breath_width_seconds: float = 0.5,
) -> dict[str, Any]:
    fs = float(ventilator_signals["fs"])
    events = [
        _normalize_ventilator_breath(
            detection, fs=fs, width_seconds=breath_width_seconds
        )
        for detection in _iter_ventilator_detections(ventilator_breath_indices)
    ]
    session.add_events("ventilator_breaths", events)
    return {}
