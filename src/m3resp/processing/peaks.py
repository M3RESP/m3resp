"""Shared peak-detection primitives.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from ReSurfEMG.

    Source:     https://github.com/resurfemg-org/ReSurfEMG
    Revision:   m3resp-integration (c63668689030e4581d5f985e7d09d3a8c01e7a77)
    Original:   resurfemg/postprocessing/event_detection.py::detect_emg_breaths,
                detect_ventilator_breath, find_occluded_breaths, find_linked_peaks
    Copyright:  Copyright (c) 2022 Netherlands eScience Center and
                University of Twente
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - `find_occluded_breaths` renamed to `detect_occluded_breath_peaks`.
    - Functions renamed to `detect_emg_breath_peaks`, `detect_ventilator_breath_peaks`,
      `detect_occluded_breath_peaks`, `closest_event_indices` to fit M3RESP naming
      conventions.
    - Parameters renamed and reorganized as keyword-only arguments.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
from itertools import pairwise
from typing import Any, overload

import numpy as np

from m3resp.core.exceptions import OptionalDependencyError, MutuallyExclusiveArgsError
from m3resp.emg.processing.shared import (
    _require_sampling_frequency,
    _validate_incompatible_kwargs,
)
from m3resp.processing.filters import capture_value


def detect_peaks(
    values: np.ndarray,
    *,
    height: float | np.ndarray | None = None,
    prominence: float | None = None,
    width: float | None = None,
    distance: float | None = None,
    threshold: float | None = None,
    invert: bool = False,
    captures: dict[str, Any] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> np.ndarray:
    """Detect peaks in a signal.

    Detect peaks in an input signal using SciPy, optionally after inverting the signal.
    Returns only the peak indices. Pass a dict via `captures` to also
    receive SciPy's peak `properties` (heights/widths/prominences/...)
    under `captures["properties"]`.

    Args:
        values (np.ndarray): Input signal.
        height (float | np.ndarray | None): Required height of peaks.
        prominence (float | None): Required prominence of peaks.
        width (float | None): Required width of peaks.
        distance (float | None): Required minimum distance between peaks.
        threshold (float | None): Required threshold of peaks.
        invert (bool): If True, detect valleys instead of peaks.
        captures (dict[str, Any] | None): Optional dict to capture peak properties.
        kwargs (Any): Additional keyword arguments passed to
            `scipy.signal.find_peaks`.

    Returns:
        np.ndarray: Indices of detected peaks.
    """
    signal = _scipy_signal()
    data = -np.asarray(values) if invert else np.asarray(values)
    indices, properties = signal.find_peaks(
        data,
        height=height,
        prominence=prominence,
        width=width,
        distance=distance,
        threshold=threshold,
        **kwargs,
    )
    capture_value(captures, "properties", properties)
    return indices


def detect_peaks_above_moving_average(
    values: np.ndarray,
    moving_average: np.ndarray,
    *,
    minimum_distance: float,
    invert: bool = False,
) -> np.ndarray:
    """Detect extrema whose height is above a moving-average baseline."""
    data = np.asarray(values)
    baseline = np.asarray(moving_average)
    if invert:
        data = -data
        baseline = -baseline
    return detect_peaks(data, distance=max(minimum_distance, 1), height=baseline)


def detect_emg_breath_peaks(
    envelope: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
    threshold: float = 0,
    prominence_factor: float = 0.5,
    min_peak_width_samples: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Identify breaths in EMG envelope.

    Identify the electrophysiological breaths from the EMG envelope and return
    an array breath peak indices. Input of baseline threshold, peak prominence
    factor, and minimal peak width are optional.

    Args:
        envelope (numpy.ndarray): 1D EMG envelope signal.
        baseline (numpy.ndarray, optional): EMG baseline. If none provided,
            0 baseline is used.
        threshold (float): Required threshold of peaks, vertical threshold to
            neighbouring samples.
        prominence_factor (float): Required prominence of peaks, relative to the
            75th - 50th percentile of the emg_env above the baseline.
        min_peak_width_samples (int): Required width of peak in samples.
        kwargs: ReSurfEMG backwards-compatible alternative keyword arguments for
            `baseline` and `min_peak_width_samples`:
            - `emg_baseline` (numpy.ndarray, optional): EMG baseline. If none provided,
                0 baseline is used.
            - `min_peak_width_s` (int): Required width of peak in samples.

    Returns:
        list[int]: List of EMG breath peak indices.
    """
    envelope = np.asarray(envelope)
    if baseline is not None and kwargs.get("emg_baseline") is not None:
        msg = "baseline and emg_baseline cannot both be set at the same time."
        raise MutuallyExclusiveArgsError(msg)

    if (
        min_peak_width_samples is not None
        and kwargs.get("min_peak_width_s") is not None
    ):
        msg_0 = "min_peak_width_samples and min_peak_width_s cannot both be set at the same time."
        raise MutuallyExclusiveArgsError(msg_0)

    baseline = baseline or kwargs.get("emg_baseline", np.zeros_like(envelope))

    min_peak_width_samples = min_peak_width_samples or kwargs.get("min_peak_width_s", 1)

    delta = envelope - baseline
    prominence = prominence_factor * (
        np.nanpercentile(delta, 75) + np.nanpercentile(delta, 50)
    )
    return detect_peaks(
        envelope,
        height=threshold,
        prominence=prominence,
        width=min_peak_width_samples,
    )


# NOTE ported from resurfemg.postprocessing.event_detection.detect_ventilator_breath
# NOTE: CHANGELOG `if no start and end index are provided, the entire signal is used
def detect_ventilator_breath_peaks(
    volume: np.ndarray,
    *,
    start_index: int | None = 0,
    end_index: int | None = None,
    width_samples: int = 1,
    threshold: float | None = None,
    prominence: float | None = None,
    threshold_refined: float | None = None,
    prominence_refined: float | None = None,
    **kwargs,
) -> np.ndarray:
    """Identify breaths in ventilator volume data.

    Identify the breaths from the ventilator signal and return an array
    of ventilator peak breath indices, in two steps of peak detection.
    Input of threshold and prominence values is optional.

    The peaks are detected in a window of the volume signal, specified by `start_index`
    and `end_index`. If no start or end indexes are provided, the entire signal is used.

    Args:
        volume (numpy.ndarray): Ventilator volume signal.
        start_index (int): Start sample of the window where to search for breaths.
        end_index (int): End sample of the window where to search for breaths.
        width_samples (int): Required width of peak in samples.
        threshold (int | None, optional): Required threshold of peaks, vertical
            threshold to neighbouring samples. Defaults to None.
        prominence (int | None, optional): Required prominence of peaks.
            Defaults to None.
        threshold_refined (int | None, optional): Refined threshold for peak
            detection. Defaults to None.
        prominence_refined (int | None, optional): Refined prominence for peak
            detection. Defaults to None.
        kwargs (Any): ReSurfEMG backwards-compatible alternative keyword arguments for
            `threshold_refined` and `prominence_refined`:
            - `threshold_new` (int | None, optional): Refined threshold for peak
                detection. Defaults to None.
                Mutually exclusive with `threshold_refined`.
            - `prominence_new` (int | None, optional): Refined prominence for peak
                detection. Defaults to None.
                Mutually exclusive with `prominence_refined`.

    Returns:
        np.ndarray: List of ventilator breath peak indices.
    """
    start_index = start_index or 0
    end_index = end_index or len(volume) - 1
    volume_slice = np.asarray(volume)[int(start_index) : int(end_index)]
    if threshold is None:
        threshold = 0.25 * np.percentile(volume_slice, 90)
    if prominence is None:
        prominence = 0.10 * np.percentile(volume_slice, 90)

    first_pass = detect_peaks(
        volume_slice,
        height=threshold,
        prominence=prominence,
        width=width_samples,
    )
    threshold_refined = _validate_incompatible_kwargs(
        arg_name1="threshold_refined",
        arg_value1=threshold_refined,
        arg_name2="threshold_new",
        default_value=0.5 * np.percentile(volume_slice[first_pass], 90),
        **kwargs,
    )
    prominence_refined = _validate_incompatible_kwargs(
        arg_name1="prominence_refined",
        arg_value1=prominence_refined,
        arg_name2="prominence_new",
        default_value=0.5 * np.percentile(volume_slice, 90),
        **kwargs,
    )

    return detect_peaks(
        volume_slice,
        height=threshold_refined,
        prominence=prominence_refined,
        width=width_samples,
    )


def detect_occluded_breath_peaks(
    pressure: np.ndarray,
    *,
    sample_frequency: float,
    peep: float,
    start_index: int = 0,
    end_index: int | None = None,
    prominence_factor: float = 0.8,
    min_width_seconds: float | None = None,
    distance_seconds: float | None = None,
    **kwargs,
) -> np.ndarray:
    """Find occlusion manoeuvres in ventilator pressure data.

    Find end-expiratory occlusion manoeuvres (Pocc) in ventilator pressure
    timeseries data. start_idx and end_idx specify the samples to look into.
    The prominence_factor, min_width_s, and distance_s specify the minimal
    peak prominence relative to the PEEP level, peak width in samples, and
    distance to other peaks.

    Args:
        pressure (numpy.ndarray): Ventilator pressure signal.
        sample_frequency (float): Sampling rate.
        peep (float): Positive end-expiratory pressure.
        start_index (int): Start index to start looking for Pocc manoeuvres.
        end_index (int, optional): End index to stop looking for Pocc manoeuvres.
        prominence_factor (float): Multiplier in setting the minimum peak prominence.
        min_width_seconds (float, optional): Minimum peak width in seconds.
        distance_seconds (float, optional): Minimum interpeak distance in seconds.seconds
        kwargs: ReSurfEMG backwards-compatible alternative keyword arguments for
            `min_width_seconds` and `distance_seconds`:
            - `min_width_s` (int, optional): Minimum peak width in seconds.
                Mutually exclusive with `min_width_seconds`.
            - `distance_s` (int, optional): Minimum interpeak distance in seconds.
                Mutually exclusive with `distance_seconds`.

    Returns:
        numpy.ndarray: List of Pocc peak indices.
    """
    pressure = np.asarray(pressure)
    if end_index is None:
        end_index = len(pressure) - 1

    sample_frequency = _require_sampling_frequency(sample_frequency)
    min_width_seconds = float(
        _validate_incompatible_kwargs(
            arg_name1="min_width_seconds",
            arg_name2="min_width_s",
            arg_value1=min_width_seconds,
            default_value=0.1,
            kwargs=kwargs,
        )
    )
    distance_seconds = float(
        _validate_incompatible_kwargs(
            arg_name1="distance_seconds",
            arg_name2="distance_s",
            arg_value1=distance_seconds,
            default_value=0.5,
            kwargs=kwargs,
        )
    )
    min_width_samples = int(sample_frequency * min_width_seconds)
    distance_samples = int(sample_frequency * distance_seconds)

    prominence = prominence_factor * np.abs(peep - min(pressure))
    height = prominence - peep
    return detect_peaks(
        pressure[start_index:end_index],
        invert=True,
        height=height,
        prominence=prominence,
        width=min_width_samples,
        distance=distance_samples,
    )


def merge_close_peaks(
    peak_indices: np.ndarray,
    values: np.ndarray,
    *,
    min_distance_samples: int,
) -> np.ndarray:
    """Merge peaks that are closer together than ``min_distance_samples``.

    Peak detection can report one breath's flat top as two or three peaks a
    few samples apart. Going through the peaks in time order, a peak closer
    than ``min_distance_samples`` to the last kept peak replaces it only if
    it is higher in ``values``; otherwise it is dropped. On a tie the earlier
    peak is kept.

    Returns the kept peak indices in time order.
    """

    peaks = np.sort(np.asarray(peak_indices, dtype=int))
    if peaks.size < 2:
        return peaks
    data = np.asarray(values)
    kept = [int(peaks[0])]
    for peak in peaks[1:]:
        if peak - kept[-1] < min_distance_samples:
            if data[peak] > data[kept[-1]]:
                kept[-1] = int(peak)
        else:
            kept.append(int(peak))
    return np.asarray(kept, dtype=int)


def detect_pressure_dip_breaths(
    pressure: np.ndarray,
    *,
    sample_frequency: float,
    smoothing_seconds: float = 0.2,
    min_depth: float = 0.15,
    min_interval_seconds: float = 2.0,
) -> np.ndarray:
    """Find breaths as dips in the airway pressure of a spontaneously
    breathing subject.

    Breathing in through a mouthpiece or mask lowers the airway pressure, so
    each breath shows up as a dip. The pressure is first smoothed with a
    moving average of ``smoothing_seconds`` to remove fast noise. A dip counts
    as a breath when it is at least ``min_depth`` deep compared with the
    pressure around it (in the pressure's own unit, e.g. cmH2O) and at least
    ``min_interval_seconds`` after the previous breath (2 s allows up to
    30 breaths/min).

    Returns the sample index of the lowest point of each dip, which is the
    moment of strongest breathing-in effort.

    Not for mechanically ventilated breaths, where breathing in raises the
    airway pressure. Missing samples (NaN) are bridged for the smoothing
    only, with a warning, and no breath is placed on a missing sample.
    """

    values = np.asarray(pressure, dtype=float)
    missing = ~np.isfinite(values)
    if missing.all():
        raise ValueError(
            "detect_pressure_dip_breaths: every pressure sample is missing."
        )
    if missing.any():
        warnings.warn(
            f"{int(missing.sum())} airway pressure samples are missing (NaN); "
            "they are bridged for smoothing and no breath is placed on them.",
            UserWarning,
            stacklevel=2,
        )
        positions = np.arange(values.size)
        values = values.copy()
        values[missing] = np.interp(
            positions[missing], positions[~missing], values[~missing]
        )

    window = max(1, round(smoothing_seconds * sample_frequency))
    smoothed = np.convolve(values, np.ones(window) / window, mode="same")
    indices = detect_peaks(
        smoothed,
        invert=True,
        prominence=min_depth,
        distance=max(1, round(min_interval_seconds * sample_frequency)),
    )
    return indices[~missing[indices]]


def pair_valley_peak_valley(
    values: np.ndarray,
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Pair each peak with the adjacent valley indices around it."""
    peaks = np.asarray(peak_indices, dtype=int)
    valleys = np.asarray(valley_indices, dtype=int)
    pairs: list[tuple[int, int, int]] = []
    for start, end in pairwise(valleys):
        between = peaks[(peaks > start) & (peaks < end)]
        if len(between):
            pairs.append((int(start), int(between[0]), int(end)))
    return pairs


def remove_duplicate_extrema(
    values: np.ndarray,
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove duplicate peaks/valleys between adjacent extrema."""
    data = np.asarray(values)
    peaks = np.asarray(peak_indices, dtype=int).copy()
    valleys = np.asarray(valley_indices, dtype=int).copy()
    peak_values = data[peaks]
    valley_values = data[valleys]

    current_valley_index = 0
    while current_valley_index < len(valleys) - 1:
        start = valleys[current_valley_index]
        end = valleys[current_valley_index + 1]
        peaks_between = np.argwhere((peaks > start) & (peaks < end))
        if not len(peaks_between):
            delete_valley_index = (
                current_valley_index
                if valley_values[current_valley_index]
                > valley_values[current_valley_index + 1]
                else current_valley_index + 1
            )
            valleys = np.delete(valleys, delete_valley_index)
            valley_values = np.delete(valley_values, delete_valley_index)
            continue

        if len(peaks_between) > 1:
            delete_peak_index = (
                peaks_between[0]
                if peak_values[peaks_between[0]] < peak_values[peaks_between[1]]
                else peaks_between[1]
            )
            peaks = np.delete(peaks, delete_peak_index)
            peak_values = np.delete(peak_values, delete_peak_index)
            continue

        current_valley_index += 1

    return peaks, valleys


def remove_low_amplitude_peaks(
    values: np.ndarray,
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
    *,
    fraction: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove peaks below a fraction of the median valley-to-peak amplitude."""
    if not fraction:
        return np.asarray(peak_indices), np.asarray(valley_indices)

    data = np.asarray(values)
    peaks = np.asarray(peak_indices, dtype=int)
    valleys = np.asarray(valley_indices, dtype=int)
    if len(peaks) == 0 or len(valleys) == 0:
        return peaks, valleys

    peak_values = data[peaks]
    valley_values = data[valleys]
    inspiratory_amplitude = peak_values - valley_values[:-1]
    expiratory_amplitude = peak_values - valley_values[1:]
    amplitude = (inspiratory_amplitude + expiratory_amplitude) / 2
    amplitude_cutoff = fraction * np.median(amplitude)
    delete_peaks = np.argwhere(amplitude < amplitude_cutoff)
    peaks = np.delete(peaks, delete_peaks)
    return remove_duplicate_extrema(data, peaks, valleys)


# tr
def closest_event_indices(
    reference_times: np.ndarray,
    candidate_times: np.ndarray,
) -> np.ndarray:
    """Find indices of candidate times nearest to each reference time."""
    reference = np.asarray(reference_times)
    candidates = np.asarray(candidate_times)
    closest = np.zeros(reference.shape, dtype=int)
    for index, time in enumerate(reference):
        closest[index] = np.argmin(np.abs(candidates - time))
    return closest


def _scipy_signal():
    try:
        from scipy import signal
    except ImportError as exc:
        msg = (
            "Peak detection requires SciPy. Install `scipy` to use "
            "`m3resp.processing.peaks`."
        )
        raise OptionalDependencyError(msg) from exc
    return signal
