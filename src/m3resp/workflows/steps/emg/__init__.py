"""Registered EMG pipeline steps.

Loading/preprocessing steps wrap the ``M3Session`` EMG stage methods.
Postprocessing steps keep the one-step-per-operation structure used by
``eit.py``. Factored signal-processing primitives are used where available;
remaining upstream imports are deferred to call time so the package installs
without the optional ``resurfemg`` dependency.

Ventilator-only steps (loading, channel splitting, breath/Pocc detection,
respiratory rate, non-consecutive-manoeuvre quality) live in
``m3resp.workflows.steps.ventilator``, not here - see that package's
docstring. A handful of steps below still read ventilator artifacts alongside
EMG ones (``emg.evaluate_event_timing``, ``emg.evaluate_respiratory_rates``):
those stay in this package because they are genuinely cross-modal, scoring
*EMG* detection quality against a ventilator reference.

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
from .features import (
    amplitude,
    area_under_baseline,
    pseudo_slope,
    respiratory_rate,
    time_product,
    time_to_peak,
)
from .loading import (
    detect_breaths,
    load,
    peak_indices,
    preprocess,
)
from .onoffpeak import (
    interpeak_dist,
    onoffpeak_baseline_crossing,
    onoffpeak_slope_extrapolation,
)
from .quality_events import (
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
    "ecg_detect_peaks",
    "ecg_estimated_subtraction",
    "ecg_gating",
    "ecg_wavelet_denoising",
    "evaluate_bell_curve_error",
    "evaluate_event_timing",
    "evaluate_respiratory_rates",
    "interpeak_dist",
    "load",
    "moving_baseline",
    "onoffpeak_baseline_crossing",
    "onoffpeak_slope_extrapolation",
    "peak_indices",
    "percentage_under_baseline",
    "preprocess",
    "pseudo_slope",
    "respiratory_rate",
    "slopesum_baseline",
    "snr_pseudo",
    "time_product",
    "time_to_peak",
]
