"""Shared peak-detection primitives."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.exceptions import OptionalDependencyError


def detect_peaks(
    values: np.ndarray,
    *,
    height: float | np.ndarray | None = None,
    prominence: float | None = None,
    width: float | None = None,
    distance: float | None = None,
    threshold: float | None = None,
    invert: bool = False,
    return_properties: bool = False,
    **kwargs: Any,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Detect peaks using SciPy, optionally after inverting the signal."""

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
    return (indices, properties) if return_properties else indices


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
    emg_baseline: np.ndarray | None = None,
    threshold: float = 0,
    prominence_factor: float = 0.5,
    min_peak_width_samples: int = 1,
    min_peak_width_s: int | None = None,
) -> np.ndarray:
    """Detect EMG breath peaks with ReSurfEMG-compatible defaults."""

    envelope = np.asarray(envelope)
    if baseline is None and emg_baseline is not None:
        baseline = emg_baseline
    if baseline is None:
        baseline = np.zeros(envelope.shape)
    if min_peak_width_s is not None:
        min_peak_width_samples = min_peak_width_s
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


def detect_ventilator_breath_peaks(
    volume: np.ndarray,
    *,
    start_index: int,
    end_index: int,
    width_samples: int,
    threshold: float | None = None,
    prominence: float | None = None,
    threshold_refined: float | None = None,
    prominence_refined: float | None = None,
    threshold_new: float | None = None,
    prominence_new: float | None = None,
) -> np.ndarray:
    """Detect ventilator breath peaks with ReSurfEMG-compatible two-step logic."""

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

    if threshold_refined is None and threshold_new is not None:
        threshold_refined = threshold_new
    if prominence_refined is None and prominence_new is not None:
        prominence_refined = prominence_new
    if threshold_refined is None:
        threshold_refined = 0.5 * np.percentile(volume_slice[first_pass], 90)
    if prominence_refined is None:
        prominence_refined = 0.5 * np.percentile(volume_slice, 90)

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
    min_width_s: float | None = None,
    distance_s: float | None = None,
) -> np.ndarray:
    """Detect occlusion manoeuvre peaks in ventilator pressure."""

    pressure = np.asarray(pressure)
    if end_index is None:
        end_index = len(pressure) - 1
    if min_width_seconds is None and min_width_s is not None:
        min_width_seconds = min_width_s
    if distance_seconds is None and distance_s is not None:
        distance_seconds = distance_s
    if min_width_seconds is None:
        min_width_seconds = 0.1
    if distance_seconds is None:
        distance_seconds = 0.5

    min_width_samples = int(min_width_seconds * sample_frequency)
    distance_samples = int(distance_seconds * sample_frequency)
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


def pair_valley_peak_valley(
    values: np.ndarray,
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Pair each peak with the adjacent valley indices around it."""

    peaks = np.asarray(peak_indices, dtype=int)
    valleys = np.asarray(valley_indices, dtype=int)
    pairs: list[tuple[int, int, int]] = []
    for start, end in zip(valleys[:-1], valleys[1:]):
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
        raise OptionalDependencyError(
            "Peak detection requires SciPy. Install `scipy` to use "
            "`m3resp.processing.peaks`."
        ) from exc
    return signal
