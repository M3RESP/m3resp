"""Shared per-breath metric primitives.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from ReSurfEMG.

    Source:     https://github.com/resurfemg-org/ReSurfEMG
    Revision:   m3resp-integration (c63668689030e4581d5f985e7d09d3a8c01e7a77)
    Original:   resurfemg/postprocessing/features.py::time_to_peak, pseudo_slope,
                amplitude, time_product, area_under_baseline, respiratory_rate
    Copyright:  Copyright (c) 2022 Netherlands eScience Center and
                University of Twente
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - `amplitude` renamed to `amplitude_at_peaks`, `time_product` renamed to
      `window_integral`, `respiratory_rate` renamed to
      `respiratory_rate_from_indices`.
    - Parameters renamed and reorganized as keyword-only arguments.
    - `tidal_variation` below is independent M3RESP code (EIT-style tidal
      variation), not derived from ReSurfEMG.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.integrate import trapezoid

from m3resp.processing.windows import running_smoother


def time_to_peak(
    values: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    *,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute absolute and relative time-to-peak within breath windows."""

    data = np.asarray(values)
    starts = np.asarray(start_indices)
    ends = np.asarray(end_indices)
    absolute_times = np.zeros(starts.shape)
    percent_times = np.zeros(starts.shape)
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        breath_arc = data[start:end]
        peak_source = running_smoother(breath_arc) if smooth else breath_arc
        absolute_times[index] = peak_source.argmax()
        percent_times[index] = absolute_times[index] / len(breath_arc)
    return absolute_times, percent_times


def pseudo_slope(
    values: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    *,
    smooth: bool = True,
    smoothing: bool | None = None,
) -> np.ndarray:
    """Compute the initial per-breath pseudo-slope in units per sample.

    An approximate rise rate, not a true slope: for each breath, this
    divides the peak height by the time it took to reach the peak, rather
    than fitting an actual slope to the rising edge.
    """

    if smoothing is not None:
        smooth = smoothing

    data = np.asarray(values)
    starts = np.asarray(start_indices)
    ends = np.asarray(end_indices)
    slopes = np.zeros(starts.shape)
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        breath_arc = data[start:end]
        positive_arc = abs(breath_arc)
        if smooth:
            smoothed_breath = running_smoother(positive_arc)
            absolute_time = smoothed_breath.argmax()
        else:
            absolute_time = positive_arc.argmax()
        absolute_height = positive_arc[absolute_time]
        slopes[index] = absolute_height / absolute_time
    return slopes


def amplitude_at_peaks(
    values: np.ndarray,
    peak_indices: np.ndarray,
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Compute peak amplitudes relative to a baseline or zero."""

    data = np.asarray(values)
    peaks = np.asarray(peak_indices, dtype=int)
    if baseline is None:
        baseline = np.zeros(data.shape)
    return np.array(data[peaks] - np.asarray(baseline)[peaks])


def window_integral(
    values: np.ndarray,
    sample_frequency: float,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate signal-minus-baseline over each inclusive sample window."""

    data = np.asarray(values)
    starts = np.asarray(start_indices)
    ends = np.asarray(end_indices)
    if baseline is None:
        baseline = np.zeros(data.shape)
    baseline = np.asarray(baseline)

    integrals = np.zeros(starts.shape)
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        delta = data[start : end + 1] - baseline[start : end + 1]
        _warn_if_crosses_baseline(delta, index)
        integrals[index] = np.abs(trapezoid(delta, dx=1 / sample_frequency))
    return integrals


def area_under_baseline(
    values: np.ndarray,
    sample_frequency: float,
    peak_indices: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    window: int,
    baseline: np.ndarray,
    reference_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ReSurfEMG-style area under baseline around each breath peak."""

    data = np.asarray(values)
    if reference_values is None:
        reference_values = data
    reference = np.asarray(reference_values)
    baseline = np.asarray(baseline)
    peaks = np.asarray(peak_indices)
    starts = np.asarray(start_indices)
    ends = np.asarray(end_indices)

    areas = np.zeros(peaks.shape)
    references = np.zeros(peaks.shape)
    for index, (start, peak, end) in enumerate(zip(starts, peaks, ends, strict=True)):
        delta_curve = data[start : end + 1] - baseline[start : end + 1]
        reference_start = max([0, peak - window])
        reference_end = min([len(data) - 1, peak + window])
        _warn_if_crosses_baseline(delta_curve, index)

        if np.median(np.sign(delta_curve[1:]) >= 0):
            y_ref = min(reference[reference_start:reference_end])
            delta = baseline[start : end + 1] - y_ref
        else:
            y_ref = max(reference[reference_start:reference_end])
            delta = y_ref - baseline[start : end + 1]

        areas[index] = np.abs(trapezoid(delta, dx=1 / sample_frequency))
        references[index] = y_ref
    return areas, references


def respiratory_rate_from_indices(
    indices: np.ndarray,
    sample_frequency: float,
    *,
    outlier_percentile: float = 33,
    outlier_factor: float = 3,
) -> tuple[float, np.ndarray]:
    """Estimate median and breath-to-breath respiratory rate in breaths/min."""

    breath_indices = np.asarray(indices)
    breath_interval = breath_indices[1:] - breath_indices[:-1]
    breath_to_breath = 60 * sample_frequency / breath_interval
    outlier_threshold = outlier_factor * np.percentile(
        breath_to_breath, outlier_percentile
    )
    breath_to_breath[breath_to_breath > outlier_threshold] = np.nan
    median_rate = float(np.nanmedian(breath_to_breath))
    return median_rate, breath_to_breath


def tidal_variation(
    values: np.ndarray,
    time: np.ndarray,
    breaths: list[Any] | np.ndarray,
    *,
    method: str = "inspiratory",
) -> np.ndarray:
    """Compute EIT-style tidal variation for breath-like timing objects."""

    data = np.asarray(values)
    sample_time = np.asarray(time)
    breaths_array = np.asarray(breaths, dtype=object)
    valid_breath_indices = np.flatnonzero(
        [breath is not None for breath in breaths_array]
    )
    valid_breaths = breaths_array[valid_breath_indices]

    if not len(valid_breaths):
        return np.full(len(breaths_array), np.nan)

    start_indices = np.searchsorted(
        sample_time, [_breath_time(breath, "start_time") for breath in valid_breaths]
    )
    middle_indices = np.searchsorted(
        sample_time, [_breath_time(breath, "middle_time") for breath in valid_breaths]
    )
    end_indices = np.searchsorted(
        sample_time, [_breath_time(breath, "end_time") for breath in valid_breaths]
    )

    if method == "inspiratory":
        variation = np.squeeze(
            np.diff(data[[start_indices, middle_indices]], axis=0), axis=0
        )
    elif method == "expiratory":
        variation = np.squeeze(
            np.diff(data[[end_indices, middle_indices]], axis=0), axis=0
        )
    elif method == "mean":
        mean_outer_values = data[[start_indices, end_indices]].mean(axis=0)
        end_inspiratory_values = data[middle_indices]
        variation = end_inspiratory_values - mean_outer_values
    else:
        msg = f"`method` ({method}) not valid."
        raise ValueError(
            msg
            + " Valid values for `method` are 'inspiratory', 'expiratory' and 'mean'."
        )

    output_shape = (len(breaths_array),) + np.shape(variation)[1:]
    output = np.full(output_shape, np.nan)
    output[valid_breath_indices] = variation
    return output


def _breath_time(breath: Any, name: str) -> float:
    try:
        return float(getattr(breath, name))
    except AttributeError:
        return float(breath[name])


def _warn_if_crosses_baseline(delta: np.ndarray, peak_index: int) -> None:
    if not np.all(np.sign(delta[1:]) >= 0) and not np.all(np.sign(delta[1:]) <= 0):
        warnings.warn(
            "Warning: Curve for peak idx"
            + str(peak_index)
            + " not entirely above or below baseline. The calculated integrals "
            + "will cancel out.",
            stacklevel=2,
        )
