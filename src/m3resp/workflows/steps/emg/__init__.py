"""Registered EMG pipeline steps.

Loading/preprocessing steps wrap the ``M3Session`` EMG stage methods.
Postprocessing steps keep the one-step-per-operation structure used by
``eit.py``. Factored signal-processing primitives are used where available;
remaining upstream imports are deferred to call time so the package installs
without the optional ``resurfemg`` dependency.

This package mirrors the former single ``emg.py`` module, split by pipeline
stage for readability. Importing it registers every step below (each
submodule's ``@register_step`` decorators run on import), and every public
step function is re-exported here so ``from m3resp.workflows.steps.emg
import <name>`` keeps working unchanged.
"""

from __future__ import annotations

from .baseline import moving_baseline, slopesum_baseline
from .ecg_detection import ecg_detect_peaks
from .ecg_gating import ecg_gating
from .ecg_removal import ecg_estimated_subtraction
from .ecg_wavelet import ecg_wavelet_denoising
from .event_detection import (
    detect_ventilator_breath,
    find_occluded_breaths,
    pocc_intervals,
    pocc_quality,
    pocc_time_product,
)
from .event_normalization import normalize_ventilator_breaths
from .features import (
    amplitude,
    area_under_baseline,
    pseudo_slope,
    respiratory_rate,
    time_product,
    time_to_peak,
    ventilator_respiratory_rate,
)
from .loading import (
    detect_breaths,
    load,
    load_ventilator,
    peak_indices,
    preprocess,
    ventilator_channels,
)
from .onoffpeak import (
    interpeak_dist,
    onoffpeak_baseline_crossing,
    onoffpeak_slope_extrapolation,
)
from .quality_events import (
    detect_non_consecutive_manoeuvres,
    evaluate_bell_curve_error,
    evaluate_event_timing,
    evaluate_respiratory_rates,
)
from .quality_snr import (
    detect_extreme_time_products,
    detect_local_high_aub,
    percentage_under_baseline,
    snr_pseudo,
)

__all__ = [
    "amplitude",
    "area_under_baseline",
    "detect_breaths",
    "detect_extreme_time_products",
    "detect_local_high_aub",
    "detect_non_consecutive_manoeuvres",
    "detect_ventilator_breath",
    "ecg_detect_peaks",
    "ecg_estimated_subtraction",
    "ecg_gating",
    "ecg_wavelet_denoising",
    "evaluate_bell_curve_error",
    "evaluate_event_timing",
    "evaluate_respiratory_rates",
    "find_occluded_breaths",
    "interpeak_dist",
    "load",
    "load_ventilator",
    "moving_baseline",
    "normalize_ventilator_breaths",
    "onoffpeak_baseline_crossing",
    "onoffpeak_slope_extrapolation",
    "peak_indices",
    "percentage_under_baseline",
    "pocc_intervals",
    "pocc_quality",
    "pocc_time_product",
    "preprocess",
    "pseudo_slope",
    "respiratory_rate",
    "slopesum_baseline",
    "snr_pseudo",
    "time_product",
    "time_to_peak",
    "ventilator_channels",
    "ventilator_respiratory_rate",
]
