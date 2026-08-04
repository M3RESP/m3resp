"""Regression tests for shared peak/interval primitives."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from m3resp.core.events import BreathEvent
from m3resp.processing.intervals import (
    baseline_crossings,
    onoff_from_baseline_crossings,
    onoff_from_slope,
    sample_intervals_to_breath_events,
)
from m3resp.processing.peaks import (
    closest_event_indices,
    detect_emg_breath_peaks,
    detect_occluded_breath_peaks,
    detect_peaks_above_moving_average,
    detect_ventilator_breath_peaks,
    pair_valley_peak_valley,
    remove_duplicate_extrema,
    remove_low_amplitude_peaks,
)


def _add_sibling_checkout_to_path(name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sibling = repo_root.parent / name
    if sibling.exists() and str(sibling) not in sys.path:
        sys.path.insert(0, os.fspath(sibling))


def _respiratory_wave(fs: float = 20.0, duration_seconds: float = 60.0):
    time = np.arange(int(fs * duration_seconds)) / fs
    values = (
        np.sin(2 * np.pi * 0.25 * time)
        + 0.08 * np.sin(2 * np.pi * 1.5 * time)
        + 0.002 * time
    )
    return time, values


def test_eit_peak_primitives_match_breath_detection_private_steps():
    _add_sibling_checkout_to_path("eitprocessing")
    pytest.importorskip("eitprocessing")
    from eitprocessing.features.breath_detection import BreathDetection
    from eitprocessing.features.moving_average import MovingAverage

    fs = 20.0
    _, values = _respiratory_wave(fs=fs)
    detector = BreathDetection(minimum_duration=2.0, amplitude_cutoff_fraction=0.25)
    average = MovingAverage(
        window_size=int(fs * detector.averaging_window_duration),
        window_function=detector.averaging_window_function,
    ).apply(values)

    expected_peaks = detector._find_extrema(values, average, fs)
    expected_valleys = detector._find_extrema(values, average, fs, invert=True)
    actual_peaks = detect_peaks_above_moving_average(
        values,
        average,
        minimum_distance=detector.minimum_duration * fs,
    )
    actual_valleys = detect_peaks_above_moving_average(
        values,
        average,
        minimum_distance=detector.minimum_duration * fs,
        invert=True,
    )

    np.testing.assert_array_equal(actual_peaks, expected_peaks)
    np.testing.assert_array_equal(actual_valleys, expected_valleys)

    expected_peaks, expected_valleys = detector._remove_edge_cases(
        values, expected_peaks, expected_valleys, average
    )
    actual_peaks, actual_valleys = detector._remove_edge_cases(
        values, actual_peaks, actual_valleys, average
    )
    np.testing.assert_array_equal(actual_peaks, expected_peaks)
    np.testing.assert_array_equal(actual_valleys, expected_valleys)

    expected_peaks, expected_valleys = detector._remove_doubles(
        values, expected_peaks, expected_valleys
    )
    actual_peaks, actual_valleys = remove_duplicate_extrema(
        values, actual_peaks, actual_valleys
    )
    np.testing.assert_array_equal(actual_peaks, expected_peaks)
    np.testing.assert_array_equal(actual_valleys, expected_valleys)

    expected_peaks, expected_valleys = detector._remove_low_amplitudes(
        values, expected_peaks, expected_valleys
    )
    actual_peaks, actual_valleys = remove_low_amplitude_peaks(
        values,
        actual_peaks,
        actual_valleys,
        fraction=detector.amplitude_cutoff_fraction,
    )
    np.testing.assert_array_equal(actual_peaks, expected_peaks)
    np.testing.assert_array_equal(actual_valleys, expected_valleys)


def test_pair_valley_peak_valley_returns_adjacent_triplets():
    _, values = _respiratory_wave()
    peaks = np.array([10, 30, 50])
    valleys = np.array([0, 20, 40, 60])

    assert pair_valley_peak_valley(values, peaks, valleys) == [
        (0, 10, 20),
        (20, 30, 40),
        (40, 50, 60),
    ]


def test_resurfemg_emg_and_ventilator_peak_primitives_match_upstream():
    _add_sibling_checkout_to_path("ReSurfEMG")
    pytest.importorskip("resurfemg")
    from resurfemg.postprocessing.event_detection import (
        detect_emg_breaths,
        detect_ventilator_breath,
        find_occluded_breaths,
    )

    fs = 100.0
    time = np.arange(int(fs * 30)) / fs
    envelope = 0.1 + np.maximum(0, np.sin(2 * np.pi * 0.3 * time))
    volume = 0.2 + 0.8 * np.maximum(0, np.sin(2 * np.pi * 0.3 * time))
    pressure = 8.0 - 3.0 * np.maximum(0, np.sin(2 * np.pi * 0.12 * time))

    np.testing.assert_array_equal(
        detect_emg_breath_peaks(envelope, min_peak_width_samples=int(fs)),
        detect_emg_breaths(envelope, min_peak_width_s=int(fs)),
    )
    np.testing.assert_array_equal(
        detect_emg_breath_peaks(
            envelope,
            emg_baseline=np.zeros(envelope.shape),
            min_peak_width_s=int(fs),
        ),
        detect_emg_breaths(
            envelope,
            emg_baseline=np.zeros(envelope.shape),
            min_peak_width_s=int(fs),
        ),
    )
    np.testing.assert_array_equal(
        detect_ventilator_breath_peaks(
            volume,
            start_index=0,
            end_index=len(volume) - 1,
            width_samples=int(0.5 * fs),
            threshold_new=None,
            prominence_new=None,
        ),
        detect_ventilator_breath(volume, 0, len(volume) - 1, int(0.5 * fs)),
    )
    np.testing.assert_array_equal(
        detect_occluded_breath_peaks(
            pressure,
            sample_frequency=fs,
            peep=float(np.nanmedian(pressure)),
            min_width_s=None,
            distance_s=None,
        ),
        find_occluded_breaths(pressure, fs, float(np.nanmedian(pressure))),
    )


def test_resurfemg_interval_and_linking_primitives_match_upstream():
    _add_sibling_checkout_to_path("ReSurfEMG")
    pytest.importorskip("resurfemg")
    from resurfemg.postprocessing.event_detection import (
        find_linked_peaks,
        onoffpeak_baseline_crossing,
        onoffpeak_slope_extrapolation,
    )

    fs = 100.0
    time = np.arange(int(fs * 20)) / fs
    values = np.maximum(0, np.sin(2 * np.pi * 0.4 * time))
    baseline = np.full(values.shape, 0.15)
    peaks = np.array([63, 313, 563, 813, 1063, 1313, 1563, 1813])

    np.testing.assert_array_equal(
        baseline_crossings(values, baseline),
        np.nonzero(np.diff(np.sign(values - baseline)) != 0)[0],
    )
    expected_crossing = onoffpeak_baseline_crossing(values, baseline, peaks)
    actual_crossing = onoff_from_baseline_crossings(values, baseline, peaks)
    for actual, expected in zip(
        actual_crossing[:4], expected_crossing[:4], strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert actual_crossing[4] == expected_crossing[4]

    expected_slope = onoffpeak_slope_extrapolation(values, fs, peaks, 5)
    actual_slope = onoff_from_slope(
        values,
        sample_frequency=fs,
        peak_indices=peaks,
        slope_window=5,
    )
    for actual, expected in zip(actual_slope[:4], expected_slope[:4], strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert actual_slope[4] == expected_slope[4]

    reference = np.array([0.9, 2.1, 4.8])
    candidates = np.array([0.5, 1.1, 2.0, 5.0])
    np.testing.assert_array_equal(
        closest_event_indices(reference, candidates),
        find_linked_peaks(reference, candidates),
    )


def test_sample_intervals_to_breath_events_sets_indices_and_times():
    events = sample_intervals_to_breath_events(
        start_indices=[10, 30],
        peak_indices=[20, 40],
        end_indices=[25, 45],
        sample_frequency=10.0,
        modality="emg",
        source="test",
    )

    assert events == [
        BreathEvent(
            modality="emg",
            start_time=1.0,
            end_time=2.5,
            peak_time=2.0,
            start_index=10,
            peak_index=20,
            end_index=25,
            source="test",
        ),
        BreathEvent(
            modality="emg",
            start_time=3.0,
            end_time=4.5,
            peak_time=4.0,
            start_index=30,
            peak_index=40,
            end_index=45,
            source="test",
        ),
    ]
