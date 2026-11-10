"""Breaths in an impedance waveform, found from its peaks and valleys.

A breath runs from a valley (start of inspiration) over a peak (end of
inspiration, start of expiration) to the next valley (end of expiration).
``find_breaths`` finds the breaths in these steps:

1. Values far outside the usual range of the waveform are invalid
   (``detect_invalid_data``). They are replaced by the nearest valid value
   (``remove_invalid_data``).
2. Peaks above a moving average of the waveform and valleys below it are
   found, at least ``minimum_duration`` apart (``detect_peaks_and_valleys``).
3. A first or last valley that can be cut off by the start or end of the
   waveform is removed, together with the peaks before the first valley and
   after the last valley (``remove_edge_cases``).
4. Only the highest peak between two valleys and the lowest valley between
   two peaks are kept. Breaths with an amplitude below a fraction of the
   median amplitude are removed.
5. Each peak and the valleys before and after it make one breath
   (``create_breaths_from_peak_valley_data``). Breaths near invalid data
   are removed (``remove_breaths_around_invalid_data``).

The minimum duration must be shorter than the shortest breath in the data.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/features/breath_detection.py::BreathDetection
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - The settings are the record `BreathDetectionSettings`; the calculation
      is the function `find_breaths`, with the waveform as a `Signal`. The
      result is a list of `BreathEvent`s; the upstream middle time is
      `extremum_time`.
    - Each private method of `BreathDetection` is a function of this module.
      The moving average and the removal of double and small breaths use the
      shared functions `moving_average`, `detect_peaks_above_moving_average`,
      `remove_duplicate_extrema` and `remove_low_amplitude_peaks`.
    - The breaths are pairs of sample indices until the last step, so the
      removal near invalid data reads the indices of a breath directly.
    - Bug fix: when the removal of the edge valleys leaves no valley, the
      result has no breaths (eitprocessing raises an IndexError).
    - An averaging window shorter than one sample is one sample long, as in
      eitprocessing.
    - Data of whole numbers is changed to floats before the invalid values
      are replaced.
    - Docstrings are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from m3resp.data.events import BreathEvent
from m3resp.data.signals import Signal
from m3resp.processing.peaks import (
    detect_peaks_above_moving_average,
    remove_duplicate_extrema,
    remove_low_amplitude_peaks,
)
from m3resp.processing.windows import moving_average

METHOD = "m3resp.eit.processing.find_breaths"

#: One breath as sample indices: (start valley, peak, end valley).
BreathIndices = tuple[int, int, int]


@dataclass(frozen=True, kw_only=True)
class BreathDetectionSettings:
    """Settings of the breath detection.

    Attributes:
        minimum_duration (float): Shortest expected breath in seconds, 2/3 s
            by default. Two peaks, and two valleys, are at least this far
            apart.
        averaging_window_duration (float): Length of the moving-average
            window in seconds, 15 s by default.
        averaging_window_function (Callable | None): Function that gives the
            weights of the moving-average window for a number of samples,
            ``np.blackman`` by default. None gives equal weights.
        amplitude_cutoff_fraction (float | None): Breaths with an amplitude
            below this fraction of the median amplitude are removed, 0.25 by
            default. The amplitude of a breath is the mean of its rise and
            its fall. None or 0 keeps all breaths.
        invalid_data_removal_window_length (float): Breaths that start or end
            within this time of invalid data are removed, in seconds, 0.5 s
            by default.
        invalid_data_removal_percentile (int): Percentile n for the limits of
            valid data, 5 by default. See ``detect_invalid_data``.
        invalid_data_removal_multiplier (int): Multiplier m for the limits of
            valid data, 4 by default. See ``detect_invalid_data``.
    """

    minimum_duration: float = 2 / 3
    averaging_window_duration: float = 15
    #: Weights of the moving-average window; ``np.blackman`` by default.
    averaging_window_function: Callable[[int], np.ndarray] | None = np.blackman
    amplitude_cutoff_fraction: float | None = 0.25
    invalid_data_removal_window_length: float = 0.5
    invalid_data_removal_percentile: int = 5
    invalid_data_removal_multiplier: int = 4


def find_breaths(
    waveform: Signal, settings: BreathDetectionSettings | None = None
) -> list[BreathEvent]:
    """Return the breaths of an impedance waveform.

    Args:
        waveform (Signal): One value per sample, for example the global
            impedance, with time in seconds and the samples per second in
            ``sample_frequency``.
        settings (BreathDetectionSettings | None): The settings. None gives
            the default settings.

    Returns:
        list[BreathEvent]: The breaths in time order. Each breath has the
            start, peak (``extremum_time``) and end times in seconds on the
            time axis of ``waveform``, the sample indices of these three
            points, the modality of ``waveform`` and its name in
            ``signal_name``.

    Raises:
        TypeError: If ``waveform`` is not a ``Signal``.
        ValueError: If the values of ``waveform`` do not have one axis, if
            ``sample_frequency`` is missing, or if every value is NaN.
    """

    if not isinstance(waveform, Signal):
        raise TypeError(f"find_breaths needs a Signal, got {type(waveform).__name__}.")
    settings = settings or BreathDetectionSettings()
    data = np.asarray(waveform.values)
    if data.ndim != 1:
        raise ValueError(
            "find_breaths needs one value per sample; "
            f"{waveform.name!r} has shape {data.shape}."
        )
    if waveform.sample_frequency is None:
        raise ValueError(
            f"find_breaths needs the samples per second of {waveform.name!r} "
            "(sample_frequency)."
        )
    sample_frequency = float(waveform.sample_frequency)
    time = np.asarray(waveform.time)

    invalid_data_indices = detect_invalid_data(
        data,
        percentile=settings.invalid_data_removal_percentile,
        multiplier=settings.invalid_data_removal_multiplier,
    )
    data = remove_invalid_data(data, invalid_data_indices)
    peak_indices, valley_indices = detect_peaks_and_valleys(
        data, sample_frequency, settings
    )
    breaths = create_breaths_from_peak_valley_data(peak_indices, valley_indices)
    breaths = remove_breaths_around_invalid_data(
        breaths,
        len(data),
        sample_frequency,
        invalid_data_indices,
        window_length=settings.invalid_data_removal_window_length,
    )
    return [
        BreathEvent(
            waveform.modality,
            float(time[start]),
            float(time[end]),
            extremum_time=float(time[peak]),
            start_index=start,
            end_index=end,
            extremum_index=peak,
            sample_frequency=sample_frequency,
            signal_name=waveform.name,
            source=METHOD,
        )
        for start, peak, end in breaths
    ]


def detect_invalid_data(
    data: np.ndarray, *, percentile: float, multiplier: float
) -> np.ndarray:
    """Return the indices of values outside the limits of valid data.

    The lower limit lies below the mean by ``multiplier`` times the distance
    from the mean to the ``percentile``-th percentile. The upper limit lies
    above the mean by ``multiplier`` times the distance from the mean to the
    (100 - ``percentile``)-th percentile. For example, with multiplier 4,
    percentile 5, mean 100, 5th percentile 90 and 95th percentile 120, the
    limits are 100 - 4 × 10 = 60 and 100 + 4 × 20 = 180.

    Args:
        data (np.ndarray): One value per sample.
        percentile (float): The percentile n, from 0 to 100.
        multiplier (float): The multiplier m.

    Returns:
        np.ndarray: Indices of the values below the lower limit or above the
            upper limit. When ``data`` holds a NaN, the limits are NaN and
            no index is returned.
    """

    data_mean = np.mean(data)
    lower_percentile = np.percentile(data, percentile)
    cutoff_low = data_mean - (data_mean - lower_percentile) * multiplier
    upper_percentile = np.percentile(data, 100 - percentile)
    cutoff_high = data_mean + (upper_percentile - data_mean) * multiplier
    return np.flatnonzero((data < cutoff_low) | (data > cutoff_high))


def remove_invalid_data(
    data: np.ndarray, invalid_data_indices: np.ndarray
) -> np.ndarray:
    """Return a copy of the data with invalid values and NaN filled in.

    Each invalid value and each NaN is replaced by the nearest valid value
    (``fill_nan_with_nearest_neighbour``).

    Args:
        data (np.ndarray): One value per sample.
        invalid_data_indices (np.ndarray): Indices of the invalid values.

    Returns:
        np.ndarray: The filled copy.

    Raises:
        ValueError: If no valid value is left.
    """

    data = np.asarray(data)
    data = (
        np.copy(data) if np.issubdtype(data.dtype, np.floating) else data.astype(float)
    )
    data[invalid_data_indices] = np.nan
    return fill_nan_with_nearest_neighbour(data)


def fill_nan_with_nearest_neighbour(data: np.ndarray) -> np.ndarray:
    """Return a copy with each NaN replaced by the nearest value that is not NaN.

    In a run of NaN values, the first half takes the value before the run
    and the second half the value after it; the middle value of a run of odd
    length takes the value after it. A run at the start takes the first
    value after it, a run at the end the last value before it. For example,
    ``[nan, 1, nan, nan, nan, 3, nan, nan]`` gives
    ``[1, 1, 1, 3, 3, 3, 3, 3]``.

    Args:
        data (np.ndarray): One value per sample.

    Returns:
        np.ndarray: The filled copy.

    Raises:
        ValueError: If every value is NaN.
    """

    data = np.copy(data)
    nan_indices = np.flatnonzero(np.isnan(data))
    if not len(nan_indices):
        return data
    if len(nan_indices) == len(data):
        raise ValueError("Every value of the data is NaN, so no breath can be found.")

    runs = np.split(nan_indices, np.flatnonzero(np.diff(nan_indices) != 1) + 1)
    for run in runs:
        if run[0] == 0:
            data[run] = data[run[-1] + 1]
            continue
        if run[-1] == len(data) - 1:
            data[run] = data[run[0] - 1]
            continue
        middle = len(run) // 2
        data[run[:middle]] = data[run[0] - 1]
        data[run[middle:]] = data[run[-1] + 1]
    return data


def detect_peaks_and_valleys(
    data: np.ndarray, sample_frequency: float, settings: BreathDetectionSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Return the peaks and valleys that make the breaths.

    Peaks lie above the moving average and valleys below it, at least
    ``settings.minimum_duration`` apart. Then the edge cases, the double
    peaks and valleys and the breaths with a small amplitude are removed.
    After this, each pair of neighbouring valleys has exactly one peak
    between them.

    Args:
        data (np.ndarray): One value per sample, without NaN.
        sample_frequency (float): Samples per second.
        settings (BreathDetectionSettings): The settings.

    Returns:
        tuple[np.ndarray, np.ndarray]: The peak indices and the valley
            indices. Both are empty when fewer than 2 valleys or no peak are
            found.
    """

    window_size = int(sample_frequency * settings.averaging_window_duration)
    average = moving_average(
        data,
        window_size=max(window_size, 1),
        window_function=settings.averaging_window_function,
    )
    minimum_distance = settings.minimum_duration * sample_frequency
    peak_indices = detect_peaks_above_moving_average(
        data, average, minimum_distance=minimum_distance
    )
    valley_indices = detect_peaks_above_moving_average(
        data, average, minimum_distance=minimum_distance, invert=True
    )
    if len(valley_indices) < 2 or len(peak_indices) < 1:
        return np.array([], dtype=int), np.array([], dtype=int)

    peak_indices, valley_indices = remove_edge_cases(
        data, peak_indices, valley_indices, average
    )
    peak_indices, valley_indices = remove_duplicate_extrema(
        data, peak_indices, valley_indices
    )
    return remove_low_amplitude_peaks(
        data, peak_indices, valley_indices, fraction=settings.amplitude_cutoff_fraction
    )


def remove_edge_cases(
    data: np.ndarray,
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
    moving_average: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove valleys that can be cut off by the edges, and the peaks outside.

    The first valley is removed when all data before it stays below the
    moving average at that valley: the real valley can lie before the start
    of the data. The same holds for the last valley and the data after it.
    Then the peaks before the first valley and after the last valley are
    removed, so every peak lies between two valleys.

    Args:
        data (np.ndarray): One value per sample.
        peak_indices (np.ndarray): Indices of the peaks.
        valley_indices (np.ndarray): Indices of the valleys, at least one.
        moving_average (np.ndarray): The moving average of ``data``.

    Returns:
        tuple[np.ndarray, np.ndarray]: The kept peak indices and valley
            indices. Both are empty when no valley is left.
    """

    if np.max(data[: valley_indices[0]]) < moving_average[valley_indices[0]]:
        valley_indices = np.delete(valley_indices, 0)
    if len(valley_indices) and (
        np.max(data[valley_indices[-1] :]) < moving_average[valley_indices[-1]]
    ):
        valley_indices = np.delete(valley_indices, -1)
    if not len(valley_indices):
        return np.array([], dtype=int), valley_indices

    peak_indices = peak_indices[peak_indices > valley_indices[0]]
    peak_indices = peak_indices[peak_indices < valley_indices[-1]]
    return peak_indices, valley_indices


def create_breaths_from_peak_valley_data(
    peak_indices: np.ndarray, valley_indices: np.ndarray
) -> list[BreathIndices]:
    """Join each peak with the valleys before and after it into one breath.

    Args:
        peak_indices (np.ndarray): Indices of the peaks, one between each
            pair of neighbouring valleys.
        valley_indices (np.ndarray): Indices of the valleys.

    Returns:
        list[BreathIndices]: One (start, peak, end) index triple per breath.

    Raises:
        ValueError: If the number of peaks is not one less than the number
            of valleys.
    """

    return [
        (int(start), int(peak), int(end))
        for peak, (start, end) in zip(
            peak_indices, pairwise(valley_indices), strict=True
        )
    ]


def remove_breaths_around_invalid_data(
    breaths: list[BreathIndices],
    n_samples: int,
    sample_frequency: float,
    invalid_data_indices: np.ndarray,
    *,
    window_length: float,
) -> list[BreathIndices]:
    """Remove the breaths with invalid data in them or close to them.

    A breath is removed when invalid data lies between ``window_length``
    seconds (rounded up to whole samples) before its start and the same time
    after its end.

    Args:
        breaths (list[BreathIndices]): (start, peak, end) index triples.
        n_samples (int): Number of samples of the data.
        sample_frequency (float): Samples per second.
        invalid_data_indices (np.ndarray): Indices of the invalid values.
        window_length (float): The time before and after a breath, in
            seconds.

    Returns:
        list[BreathIndices]: The kept breaths, in the same order.
    """

    if not len(invalid_data_indices):
        return list(breaths)

    invalid = np.zeros(n_samples, dtype=bool)
    invalid[invalid_data_indices] = True
    window = math.ceil(window_length * sample_frequency)
    return [
        (start, peak, end)
        for start, peak, end in breaths
        if not np.any(invalid[max(0, start - window) : min(n_samples, end + window)])
    ]
