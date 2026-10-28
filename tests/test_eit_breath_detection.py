"""Breaths in an impedance waveform, found by m3resp code.

``find_breaths`` and each of its parts must give the numbers of
eitprocessing 1.8.7: the golden files ``breaths.npz``, and the methods of
``BreathDetection`` on the same data (40 random waveforms and hand-made
cases).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from m3resp.data import BreathEvent, Signal
from m3resp.eit.processing.breath_detection import (
    BreathDetectionSettings,
    create_breaths_from_peak_valley_data,
    detect_invalid_data,
    detect_peaks_and_valleys,
    fill_nan_with_nearest_neighbour,
    find_breaths,
    remove_breaths_around_invalid_data,
    remove_edge_cases,
    remove_invalid_data,
)
from m3resp.processing.peaks import (
    detect_peaks_above_moving_average,
    remove_duplicate_extrema,
    remove_low_amplitude_peaks,
)
from m3resp.processing.windows import moving_average

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")
with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as _handle:
    MANIFEST = json.load(_handle)
DATASETS = sorted(MANIFEST["inputs"])
RELATIVE_TOLERANCE = 1e-9
SEEDS = range(40)


def _waveform(values: np.ndarray, fs: float = 20.0) -> Signal:
    return Signal(
        values=np.asarray(values, dtype=float),
        time=np.arange(len(values)) / fs,
        sample_frequency=fs,
        name="global_impedance",
        modality="eit",
    )


def _random_waveform(seed: int) -> Signal:
    """A breathing-like waveform with the problems breath detection must handle.

    Each seed gives its own breathing rate, amplitude, drift and noise. Some
    waveforms have a second harmonic (double peaks), stretches of very
    small breaths, single large outliers, flat stretches or runs of NaN.
    """

    rng = np.random.default_rng(seed)
    fs = float(rng.choice([20.0, 50.0, 50.2]))
    seconds = rng.uniform(20.0, 120.0)
    time = np.arange(int(seconds * fs)) / fs
    rate = rng.uniform(0.15, 0.8)
    amplitude = rng.uniform(0.5, 5.0)
    values = amplitude * np.sin(2 * np.pi * rate * time + rng.uniform(0, 2 * np.pi))
    values += (
        rng.uniform(0.0, 0.6)
        * amplitude
        * np.sin(4 * np.pi * rate * time + rng.uniform(0, 2 * np.pi))
    )
    values += rng.uniform(-1.0, 1.0) * amplitude * time / seconds
    values += rng.normal(0.0, rng.uniform(0.01, 0.3) * amplitude, len(time))
    n = len(time)
    if rng.random() < 0.4:
        start = int(rng.integers(0, n // 2))
        values[start : start + int(rng.integers(n // 10, n // 3))] *= 0.1
    for index in rng.integers(0, n, int(rng.integers(0, 4))):
        values[index] += rng.choice([-1.0, 1.0]) * amplitude * rng.uniform(10, 50)
    if rng.random() < 0.3:
        start = int(rng.integers(0, n - 10))
        values[start : start + int(rng.integers(5, 5 * fs))] = values[start]
    if rng.random() < 0.3:
        start = int(rng.integers(0, n - 10))
        values[start : start + int(rng.integers(1, 2 * fs))] = np.nan
    return Signal(
        values=values,
        time=time,
        sample_frequency=fs,
        name=f"random_{seed}",
        modality="eit",
    )


def _upstream():
    """A default eitprocessing ``BreathDetection``."""

    pytest.importorskip("eitprocessing")
    from eitprocessing.features.breath_detection import BreathDetection

    return BreathDetection()


def _upstream_breaths(waveform: Signal) -> np.ndarray:
    """(start, middle, end) times of eitprocessing ``find_breaths``."""

    from m3resp.adapters.eitprocessing_adapter._signals import as_continuous_data

    breaths = _upstream().find_breaths(as_continuous_data(waveform)).values
    return np.array(
        [(b.start_time, b.middle_time, b.end_time) for b in breaths], dtype=float
    ).reshape(-1, 3)


def _times(breaths: list[BreathEvent]) -> np.ndarray:
    return np.array(
        [(b.start_time, b.extremum_time, b.end_time) for b in breaths], dtype=float
    ).reshape(-1, 3)


# Golden files


@pytest.mark.parametrize("dataset", DATASETS)
def test_breaths_match_golden(dataset: str) -> None:
    with np.load(os.path.join(GOLDEN_DIR, dataset, "global_impedance.npz")) as data:
        values, time = data["filtered/values"], data["filtered/time"]
    with np.load(os.path.join(GOLDEN_DIR, dataset, "reader.npz")) as data:
        sample_frequency = float(data["sample_frequency"])
    with np.load(os.path.join(GOLDEN_DIR, dataset, "breaths.npz")) as data:
        golden = data["breaths"]
    waveform = Signal(
        values=values,
        time=time,
        sample_frequency=sample_frequency,
        name="global_impedance",
        modality="eit",
    )
    settings = BreathDetectionSettings(
        minimum_duration=MANIFEST["settings"]["breath_detection"]["minimum_duration_s"]
    )
    actual = _times(find_breaths(waveform, settings))
    assert actual.shape == golden.shape
    np.testing.assert_allclose(
        actual, golden, rtol=0, atol=RELATIVE_TOLERANCE * np.max(np.abs(golden))
    )


# Each part against eitprocessing, on random waveforms


@pytest.mark.parametrize("seed", SEEDS)
def test_whole_detection_equals_eitprocessing(seed: int) -> None:
    waveform = _random_waveform(seed)
    try:
        expected = _upstream_breaths(waveform)
    except IndexError:
        # eitprocessing fails when the edge removal leaves no valley.
        assert find_breaths(waveform) == []
        return
    np.testing.assert_array_equal(_times(find_breaths(waveform)), expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_each_part_equals_eitprocessing(seed: int) -> None:
    upstream = _upstream()
    settings = BreathDetectionSettings()
    waveform = _random_waveform(seed)
    data = waveform.values
    fs = waveform.sample_frequency

    invalid = detect_invalid_data(data, percentile=5, multiplier=4)
    np.testing.assert_array_equal(invalid, upstream._detect_invalid_data(data))
    filled = remove_invalid_data(data, invalid)
    np.testing.assert_array_equal(filled, upstream._remove_invalid_data(data, invalid))

    from eitprocessing.features.moving_average import MovingAverage

    window_size = int(fs * settings.averaging_window_duration)
    average = moving_average(
        filled, window_size=window_size, window_function=np.blackman
    )
    np.testing.assert_array_equal(
        average,
        MovingAverage(window_size=window_size, window_function=np.blackman).apply(
            filled
        ),
    )

    peaks = detect_peaks_above_moving_average(
        filled, average, minimum_distance=settings.minimum_duration * fs
    )
    valleys = detect_peaks_above_moving_average(
        filled, average, minimum_distance=settings.minimum_duration * fs, invert=True
    )
    np.testing.assert_array_equal(peaks, upstream._find_extrema(filled, average, fs))
    np.testing.assert_array_equal(
        valleys, upstream._find_extrema(filled, average, fs, invert=True)
    )
    if len(valleys) < 2 or len(peaks) < 1:
        return

    try:
        expected = upstream._remove_edge_cases(filled, peaks, valleys, average)
    except IndexError:
        assert all(
            len(part) == 0
            for part in remove_edge_cases(filled, peaks, valleys, average)
        )
        return
    peaks, valleys = remove_edge_cases(filled, peaks, valleys, average)
    np.testing.assert_array_equal(peaks, expected[0])
    np.testing.assert_array_equal(valleys, expected[1])

    expected = upstream._remove_doubles(filled, peaks, valleys)
    peaks, valleys = remove_duplicate_extrema(filled, peaks, valleys)
    np.testing.assert_array_equal(peaks, expected[0])
    np.testing.assert_array_equal(valleys, expected[1])

    expected = upstream._remove_low_amplitudes(filled, peaks, valleys)
    peaks, valleys = remove_low_amplitude_peaks(filled, peaks, valleys, fraction=0.25)
    np.testing.assert_array_equal(peaks, expected[0])
    np.testing.assert_array_equal(valleys, expected[1])

    breaths = create_breaths_from_peak_valley_data(peaks, valleys)
    upstream_breaths = upstream._create_breaths_from_peak_valley_data(
        waveform.time, peaks, valleys
    )
    assert [
        (waveform.time[s], waveform.time[p], waveform.time[e]) for s, p, e in breaths
    ] == [(b.start_time, b.middle_time, b.end_time) for b in upstream_breaths]

    kept = remove_breaths_around_invalid_data(
        breaths, len(filled), fs, invalid, window_length=0.5
    )
    upstream_kept = upstream._remove_breaths_around_invalid_data(
        upstream_breaths, waveform.time, fs, invalid
    )
    assert [
        (waveform.time[s], waveform.time[p], waveform.time[e]) for s, p, e in kept
    ] == [(b.start_time, b.middle_time, b.end_time) for b in upstream_kept]


def test_random_waveforms_cover_the_cases_of_the_tests() -> None:
    """The 40 waveforms include invalid data, NaN and removed breaths."""

    upstream = _upstream()
    with_invalid = with_nan = with_removed_breaths = 0
    for seed in SEEDS:
        waveform = _random_waveform(seed)
        data = waveform.values
        invalid = upstream._detect_invalid_data(data)
        with_invalid += bool(len(invalid))
        with_nan += bool(np.isnan(data).any())
        settings = BreathDetectionSettings()
        filled = remove_invalid_data(data, invalid)
        peaks, valleys = detect_peaks_and_valleys(
            filled, waveform.sample_frequency, settings
        )
        breaths = create_breaths_from_peak_valley_data(peaks, valleys)
        with_removed_breaths += len(find_breaths(waveform)) < len(breaths)
    assert with_invalid >= 5
    assert with_nan >= 5
    assert with_removed_breaths >= 3


# Hand-made cases


def test_fill_nan_takes_the_nearest_value() -> None:
    data = np.array([np.nan, 1, np.nan, np.nan, np.nan, 3, np.nan, np.nan])
    expected = np.array([1, 1, 1, 3, 3, 3, 3, 3], dtype=float)
    np.testing.assert_array_equal(fill_nan_with_nearest_neighbour(data), expected)
    upstream = _upstream()
    np.testing.assert_array_equal(
        upstream._fill_nan_with_nearest_neighbour(data), expected
    )
    assert np.isnan(data[0])


def test_only_nan_is_refused() -> None:
    with pytest.raises(ValueError):
        _upstream()._fill_nan_with_nearest_neighbour(np.full(5, np.nan))
    with pytest.raises(ValueError, match="Every value"):
        find_breaths(_waveform(np.full(100, np.nan)))


def _sine(seconds: float = 30.0, rate: float = 0.25, fs: float = 20.0) -> np.ndarray:
    time = np.arange(int(seconds * fs)) / fs
    return np.sin(2 * np.pi * rate * time)


@pytest.mark.parametrize(
    "case",
    [
        "clean",
        "flat_stretch",
        "invalid_at_start",
        "invalid_at_end",
        "double_peaks",
        "small_breaths",
        "integers",
    ],
)
def test_hand_made_cases_equal_eitprocessing(case: str) -> None:
    values = _sine()
    if case == "flat_stretch":
        values[200:320] = values[200]
    elif case == "invalid_at_start":
        values[2] = 80.0
    elif case == "invalid_at_end":
        values[-3] = -80.0
    elif case == "double_peaks":
        values += 0.6 * np.sin(2 * np.pi * 0.75 * np.arange(len(values)) / 20.0)
    elif case == "small_breaths":
        values[240:400] *= 0.1
    elif case == "integers":
        values = np.round(values * 100)
    waveform = _waveform(values)
    np.testing.assert_array_equal(
        _times(find_breaths(waveform)), _upstream_breaths(waveform)
    )


def test_invalid_data_removes_the_breaths_around_it() -> None:
    values = _sine()
    values[300] = 80.0
    breaths = find_breaths(_waveform(values))
    clean = find_breaths(_waveform(_sine()))
    assert len(breaths) < len(clean)
    for breath in breaths:
        assert not breath.start_time - 0.5 <= 15.0 < breath.end_time + 0.5


def test_no_valley_left_gives_no_breaths() -> None:
    # Two valleys; the data before the first and after the last stays below
    # the moving average, so both valleys are removed.
    data = np.array([0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0])
    average = np.full(7, 1.5)
    peaks, valleys = remove_edge_cases(data, np.array([3]), np.array([2, 4]), average)
    assert len(peaks) == 0
    assert len(valleys) == 0
    with pytest.raises(IndexError):
        _upstream()._remove_edge_cases(data, np.array([3]), np.array([2, 4]), average)


def test_short_or_flat_data_gives_no_breaths() -> None:
    assert find_breaths(_waveform(np.ones(200))) == []
    assert find_breaths(_waveform(_sine(seconds=3.0))) == []


def test_breath_events_carry_times_indices_and_origin() -> None:
    waveform = _waveform(_sine())
    breaths = find_breaths(waveform)
    assert len(breaths) == 6
    for breath in breaths:
        assert isinstance(breath, BreathEvent)
        assert breath.modality == "eit"
        assert breath.signal_name == "global_impedance"
        assert breath.sample_frequency == 20.0
        assert breath.source == "m3resp.eit.processing.find_breaths"
        assert breath.start_time == waveform.time[breath.start_index]
        assert breath.extremum_time == waveform.time[breath.extremum_index]
        assert breath.end_time == waveform.time[breath.end_index]
        assert breath.start_time < breath.extremum_time < breath.end_time


def test_settings_change_the_result_as_in_eitprocessing() -> None:
    pytest.importorskip("eitprocessing")
    from eitprocessing.features.breath_detection import BreathDetection

    from m3resp.adapters.eitprocessing_adapter._signals import as_continuous_data

    waveform = _random_waveform(7)
    options = {
        "minimum_duration": 1.5,
        "averaging_window_duration": 8.0,
        "averaging_window_function": None,
        "amplitude_cutoff_fraction": None,
        "invalid_data_removal_window_length": 2.0,
        "invalid_data_removal_percentile": 10,
        "invalid_data_removal_multiplier": 2,
    }
    expected = BreathDetection(**options).find_breaths(as_continuous_data(waveform))
    actual = find_breaths(waveform, BreathDetectionSettings(**options))
    assert _times(actual).tolist() == [
        [b.start_time, b.middle_time, b.end_time] for b in expected.values
    ]


def test_wrong_input_is_refused() -> None:
    with pytest.raises(TypeError, match="Signal"):
        find_breaths(np.zeros(100))  # type: ignore[arg-type]
    images = Signal(
        values=np.zeros((10, 2, 2)), time=np.arange(10) / 20.0, sample_frequency=20.0
    )
    with pytest.raises(ValueError, match="one value per sample"):
        find_breaths(images)
    no_rate = Signal(values=np.zeros(10), time=np.arange(10) / 20.0, name="w")
    with pytest.raises(ValueError, match="samples per second"):
        find_breaths(no_rate)
