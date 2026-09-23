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

Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   1.8.7
    Original:   eitprocessing/parameters/tidal_impedance_variation.py::
                TIV._calculate_tiv_values
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - Extracted from the private `TIV._calculate_tiv_values` method into the
      free function `tidal_variation`, so it is reusable without constructing
      a `TIV` object.
    - Breath start/middle/end times are read by attribute or by key, so plain
      mappings work alongside upstream `Breath` objects.
    - The output array keeps the trailing dimensions of the input, so
      pixel-level data is preserved rather than flattened.

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
    """Compute absolute and relative time-to-peak within breath windows.

    Args:
        values (np.ndarray): The signal containing the breath data.
        start_indices (np.ndarray): The starting indices (in samples) of each breath
            window.
        end_indices (np.ndarray): The ending indices (in samples) of each breath window.
        smooth (bool): Whether to smooth the breath data before finding the peak.
            Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: The absolute and relative time-to-peak for each
            breath.
    """

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

"""
    The pseudo-slope is an approximate rise rate, not a true slope: for each breath,
    it's computed as the ratio between the peak height and the rising time.
    A true slope would require fitting an actual slope to the rising edge and would
    depend on sampling rate and pre-processing.

    Args:
        values (np.ndarray): The signal containing the breath data.
        start_indices (np.ndarray): The starting indices (in samples) of each breath
            window.
        end_indices (np.ndarray): The ending indices (in samples) of each breath window.
        smooth (bool): Whether to smooth the breath data before finding the peak.
            Defaults to True.
        smoothing (bool | None): ReSurfEMG's backward-compatible parameter name for
            `smooth`. If provided, it overrides `smooth`. Defaults to None.
    """
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
    """Compute peak amplitudes relative to a baseline or zero.

    Calculate the peak height of signal and the baseline for the windows
    at the peak_indices relative to the baseline. If no baseline is provided, the
    peak height relative to zero is determined.

    Args:
        values (numpy.ndarray): Signal to determine the peak heights in.
        peak_indices (numpy.ndarray): List of individual peak indices.
        baseline (numpy.ndarray, optional): Running baseline of the signal.

    Returns:
        numpy.ndarray: List of peak amplitudes.
    """

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
    """Integrate signal-minus-baseline over each inclusive sample window.

    Args:
        values (numpy.ndarray): Signal to calculate the time product over.
        sample_frequency (float): Sampling frequency.
        start_indices (numpy.ndarray): List of individual peak start indices.
        end_indices (numpy.ndarray): List of individual peak end indices.
        baseline (numpy.ndarray, optional): Running baseline of the signal.
            If None, a zero baseline is used.

    Returns:
        numpy.ndarray: The calculated time products.
    """

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
"""Compute ReSurfEMG-style area under baseline around each breath peak.

    Calculate the area between the baseline and the nadir of the
    reference signal in a window around each peak.
    The nadir is the minimum (or maximum) value of the reference signal in the window
    around the peak, depending on whether the signal is above or below the baseline

    Args:
        values (numpy.ndarray): Signal to calculate the time product over.
        sample_frequency (float): Sampling frequency.
        peak_indices (list[int]): List of individual peak indices.
        start_indices (list[int]): List of individual peak start indices.
        end_indices (list[int]): List of individual peak end indices.
        window (int): Number of samples before and after peak_indices to look
            for the nadir.
        baseline (numpy.ndarray): Running baseline of the signal.
        reference_values (numpy.ndarray, optional): Signal in which the nadir is searched.

    Returns:
        tuple:
            - numpy.ndarray: The calculated areas under the baseline.
            - numpy.ndarray: The reference signal nadir values.
    """

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
    """Estimate median and breath-to-breath respiratory rate in breaths/min.

    Breath-by-breath respiratory rate larger than the threshold defined by
    the product (outlier_percentile * outlier_factor) are excluded from the computation.

    Args:
        indices (numpy.ndarray): Breath indices.
        sample_frequency (float): Sampling frequency of the signal the indices are from.
        outlier_percentile (float): Respiratory rate outlier percentile.
        outlier_factor (float): Respiratory rate outlier factor.

    Returns:
        tuple:
            - float: Median respiratory rate.
            - numpy.ndarray: Breath-to-breath respiratory rate.
    """

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
