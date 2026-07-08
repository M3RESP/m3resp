"""Shared signal-processing primitives used by modality adapters."""

from m3resp.processing.filters import (
    bandpass_filter,
    bandstop_filter,
    butterworth_filter,
    compute_power_loss,
    harmonic_notch_filter,
    highpass_filter,
    lowpass_filter,
    notch_filter,
)
from m3resp.processing.windows import (
    moving_average,
    naive_rolling_rms,
    rolling_arv,
    rolling_arv_ci,
    rolling_rms,
    rolling_rms_ci,
)

__all__ = [
    "bandpass_filter",
    "bandstop_filter",
    "butterworth_filter",
    "compute_power_loss",
    "harmonic_notch_filter",
    "highpass_filter",
    "lowpass_filter",
    "moving_average",
    "naive_rolling_rms",
    "notch_filter",
    "rolling_arv",
    "rolling_arv_ci",
    "rolling_rms",
    "rolling_rms_ci",
]
