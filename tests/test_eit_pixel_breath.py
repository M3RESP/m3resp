"""Pixel breaths of EIT images, found by m3resp code.

``find_pixel_breaths`` must give the pixel breaths of eitprocessing 1.8.7
``PixelBreath``: the golden files ``pixel.npz``, and ``PixelBreath`` with
the three phase correction modes on the same data, with the same pixels left
out. The exceptions are two bug fixes: a search window partly outside the
data, and a pixel with every breath left out (see
``_assert_equals_upstream``).
"""

from __future__ import annotations

import functools
import json
import os
import re
import warnings
from typing import Any

import numpy as np
import pytest

from m3resp.data import BreathEvent, PixelBreaths, Signal
from m3resp.eit.processing import (
    BreathDetectionSettings,
    find_breaths,
    find_pixel_breaths,
    global_impedance,
    pixel_tidal_impedance_variation,
)
from m3resp.eit.processing.pixel_breath import (
    FLAT,
    NO_CORRELATION_PEAK,
    NO_TIV,
    TOO_MANY_BREATHS_SKIPPED,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "regression", "golden", "eit")
with open(os.path.join(GOLDEN_DIR, "manifest.json"), encoding="utf-8") as _handle:
    MANIFEST = json.load(_handle)
DATASETS = sorted(MANIFEST["inputs"])
MINIMUM_DURATION = MANIFEST["settings"]["breath_detection"]["minimum_duration_s"]
MODES = ("negative amplitude", "phase shift", "none", None)
SEEDS = range(12)
UPSTREAM_SKIP = re.compile(
    r"Skipping pixel \((\d+), (\d+)\) because (no cross|more than)"
)


def _golden(dataset: str, name: str) -> dict[str, np.ndarray]:
    """Return all arrays of one golden ``.npz`` file."""

    with np.load(os.path.join(GOLDEN_DIR, dataset, f"{name}.npz")) as data:
        return {key: data[key] for key in data.files}


def _signal(values: np.ndarray, time: np.ndarray, *, name: str) -> Signal:
    return Signal(
        values=np.asarray(values, dtype=float),
        time=np.asarray(time, dtype=float),
        sample_frequency=1 / float(np.median(np.diff(time))),
        unit="AU",
        name=name,
        modality="eit",
        category="impedance",
        source="draeger",
    )


@functools.cache
def _golden_data(dataset: str, kind: str) -> tuple[Signal, Signal]:
    """The raw or the MDN-filtered pixels and global impedance of a golden file."""

    reader = _golden(dataset, "reader")
    pixels = reader["pixel_impedance"]
    if kind == "filtered":
        pixels = _golden(dataset, "mdn_filter")["pixel_impedance"]
    waveform = _golden(dataset, "global_impedance")
    return (
        _signal(pixels, reader["time"], name="pixels"),
        _signal(waveform[f"{kind}/values"], waveform[f"{kind}/time"], name="global"),
    )


def _random_data(seed: int) -> tuple[Signal, Signal]:
    """An 8 x 8 image that breathes, with pixels that are out of phase,
    upside down, flat, NaN, a random walk, or flat with one spike. The
    waveform is the sum of the pixels."""

    rng = np.random.default_rng(seed)
    fs = float(rng.choice([20.0, 50.0]))
    time = 100.0 + np.arange(int(rng.uniform(30.0, 60.0) * fs)) / fs
    rate = rng.uniform(0.2, 0.4)
    n = 8
    amplitude = rng.uniform(0.5, 2.0, (n, n))
    amplitude[rng.random((n, n)) < 0.2] *= -1
    phase = np.where(rng.random((n, n)) < 0.3, rng.uniform(0.3, 2.5, (n, n)), 0.0)
    drift = rng.uniform(-0.5, 0.5, (n, n)) * (time[:, None, None] - time[0]) / 30
    values = amplitude * np.sin(2 * np.pi * rate * time[:, None, None] - phase) + drift
    values += rng.normal(0.0, 0.1, values.shape)
    rows, columns = np.unravel_index(rng.permutation(n * n), (n, n))
    values[:, rows[:2], columns[:2]] = 3.0
    values[:, rows[2:4], columns[2:4]] = np.nan
    walk = np.cumsum(rng.normal(0.0, 0.3, (len(time), 4)), axis=0)
    values[:, rows[4:8], columns[4:8]] = walk
    values[:, rows[8], columns[8]] = 1.0
    values[int(rng.integers(len(time))), rows[8], columns[8]] = 5.0
    pixels = _signal(values, time, name="pixels")
    return pixels, _signal(global_impedance(pixels).values, time, name="global")


def _breaths(
    waveform: Signal, minimum_duration: float = MINIMUM_DURATION
) -> list[BreathEvent]:
    settings = BreathDetectionSettings(minimum_duration=minimum_duration)
    return find_breaths(waveform, settings)


def _upstream(
    pixels: Signal,
    waveform: Signal,
    mode: Any,
    minimum_duration: float = MINIMUM_DURATION,
) -> tuple[np.ndarray, list[tuple[float, float]], dict[str, set]]:
    """eitprocessing ``PixelBreath``: the (breath, row, column, 3) times, the
    global start and end times, and the pixels it warns about."""

    pytest.importorskip("eitprocessing")
    from eitprocessing.features.breath_detection import BreathDetection
    from eitprocessing.features.pixel_breath import PixelBreath

    from m3resp.adapters.eitprocessing_adapter._signals import (
        as_continuous_data,
        as_eit_data,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = PixelBreath(
            breath_detection=BreathDetection(minimum_duration=minimum_duration),
            phase_correction_mode=mode,
        ).find_pixel_breaths(
            as_eit_data(pixels), as_continuous_data(waveform), store=False
        )
    per_pixel = np.asarray(result.values, dtype=object)
    times = np.full((*per_pixel.shape, 3), np.nan)
    for index, breath in np.ndenumerate(per_pixel):
        if breath is not None:
            times[index] = (breath.start_time, breath.middle_time, breath.end_time)
    warned: dict[str, set] = {
        NO_CORRELATION_PEAK: set(),
        TOO_MANY_BREATHS_SKIPPED: set(),
    }
    for warning in caught:
        found = UPSTREAM_SKIP.match(str(warning.message))
        if found:
            reason = (
                NO_CORRELATION_PEAK
                if found[3] == "no cross"
                else TOO_MANY_BREATHS_SKIPPED
            )
            warned[reason].add((int(found[1]), int(found[2])))
    intervals = [(float(start), float(end)) for start, end in result.intervals]
    return times, intervals, warned


def _times(result: PixelBreaths) -> np.ndarray:
    return np.stack([result.start_time, result.extremum_time, result.end_time], axis=-1)


def _skipped(result: PixelBreaths, reason: str) -> set:
    return {tuple(position) for position in result.metadata["skipped_pixels"][reason]}


def _assert_equals_upstream(
    pixels: Signal, waveform: Signal, mode: Any, breaths: list[BreathEvent]
) -> None:
    """The pixel breaths equal those of eitprocessing, apart from two bug fixes.

    1. In the "phase shift" mode, the lag can move the end of inspiration of
       the first or the last global breath outside the data. eitprocessing
       moves it to the first or the last sample and keeps the pixel breath
       of that window; m3resp leaves it out.
    2. A pixel with every breath left out is listed under
       "too_many_breaths_skipped"; eitprocessing gives no warning for it.
    """

    times, intervals, warned = _upstream(pixels, waveform, mode)
    assert [(b.start_time, b.end_time) for b in breaths] == intervals
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = find_pixel_breaths(
            pixels, waveform, breaths, phase_correction_mode=mode
        )
    found = _times(result)
    same = (found == times) | (np.isnan(found) & np.isnan(times))
    differs = ~same.all(axis=-1)
    if mode == "phase shift":
        assert np.isnan(found[differs]).all()
        assert set(np.flatnonzero(differs.any(axis=(1, 2)))) <= {1, len(breaths) - 2}
    else:
        assert not differs.any()
    for reason, positions in warned.items():
        extra = _skipped(result, reason) - positions
        assert positions <= _skipped(result, reason)
        if reason == TOO_MANY_BREATHS_SKIPPED:
            for row, column in extra:
                assert np.isnan(times[:, row, column]).all()
        else:
            assert not extra
    # Pixels left out without a warning have no breaths in eitprocessing too.
    for reason in (NO_TIV, FLAT):
        for row, column in _skipped(result, reason):
            assert np.isnan(times[:, row, column]).all()


# Golden files and eitprocessing


@pytest.mark.parametrize("dataset", DATASETS)
def test_pixel_breaths_match_golden(dataset: str) -> None:
    pixels, waveform = _golden_data(dataset, "filtered")
    breaths = _breaths(waveform)
    golden = _golden(dataset, "pixel")

    result = find_pixel_breaths(pixels, waveform, breaths)

    np.testing.assert_array_equal(_times(result), golden["breaths/landmarks"])
    np.testing.assert_array_equal(
        [(b.start_time, b.end_time) for b in result.breaths],
        golden["breaths/start_end"],
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["raw", "filtered"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_pixel_breaths_equal_eitprocessing(dataset: str, kind: str, mode: Any) -> None:
    pixels, waveform = _golden_data(dataset, kind)
    _assert_equals_upstream(pixels, waveform, mode, _breaths(waveform))


@pytest.mark.parametrize("seed", SEEDS)
def test_random_pixels_equal_eitprocessing(seed: int) -> None:
    pixels, waveform = _random_data(seed)
    breaths = _breaths(waveform)
    for mode in MODES:
        _assert_equals_upstream(pixels, waveform, mode, breaths)


def test_random_pixels_cover_every_kind_of_pixel() -> None:
    """Over the seeds, the random data has pixels with a negative mean TIV
    (upside down) and pixels left out for each reason."""

    seen = dict.fromkeys(
        (NO_TIV, FLAT, NO_CORRELATION_PEAK, TOO_MANY_BREATHS_SKIPPED), 0
    )
    upside_down = 0
    for seed in SEEDS:
        pixels, waveform = _random_data(seed)
        breaths = _breaths(waveform)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for mode in ("negative amplitude", "phase shift", "none"):
                result = find_pixel_breaths(
                    pixels, waveform, breaths, phase_correction_mode=mode
                )
                for reason in seen:
                    seen[reason] += len(_skipped(result, reason))
            tiv = pixel_tidal_impedance_variation(pixels, breaths)
            mean_tiv = np.nanmean(
                [pixel_map.values for pixel_map in tiv.values], axis=0
            )
        upside_down += int((mean_tiv < 0).sum())
    assert all(count > 0 for count in seen.values()), seen
    assert upside_down > 0


def test_sign_breaths_give_the_sign_as_in_eitprocessing() -> None:
    """eitprocessing takes the sign of each pixel from the breaths of the
    default breath detection, also when the search windows come from other
    breaths."""

    pixels, waveform = _golden_data("sentec_healthy_volunteer_1b", "raw")
    breaths = _breaths(waveform, minimum_duration=1.0)
    default_breaths = find_breaths(waveform, BreathDetectionSettings())
    times, intervals, _ = _upstream(
        pixels, waveform, "negative amplitude", minimum_duration=1.0
    )
    assert [(b.start_time, b.end_time) for b in breaths] == intervals

    result = find_pixel_breaths(pixels, waveform, breaths, sign_breaths=default_breaths)
    same_breaths = find_pixel_breaths(pixels, waveform, breaths)

    np.testing.assert_array_equal(_times(result), times)
    assert not np.array_equal(_times(same_breaths), times, equal_nan=True)


# The result


def test_the_result_keeps_the_breaths_and_describes_the_pixel_breaths() -> None:
    pixels, waveform = _golden_data(DATASETS[0], "filtered")
    breaths = _breaths(waveform)

    result = find_pixel_breaths(pixels, waveform, breaths, name="test_breaths")

    assert result.name == "test_breaths"
    assert result.breaths == breaths
    assert result.shape == (32, 32)
    assert result.modality == "eit"
    assert result.method == "m3resp.eit.processing.find_pixel_breaths"
    assert result.metadata["phase_correction_mode"] == "negative amplitude"
    assert result.metadata["signal_name"] == "global"
    assert set(result.metadata["skipped_pixels"]) == {
        NO_TIV,
        FLAT,
        NO_CORRELATION_PEAK,
        TOO_MANY_BREATHS_SKIPPED,
    }
    assert not result.has_breath[0].any()
    assert not result.has_breath[-1].any()
    assert result.has_breath[1:-1].any()
    # The times are sample times of the pixels.
    assert np.isin(result.start_time[result.has_breath], pixels.time).all()


def test_warnings_come_once_per_kind_with_the_pixel_positions() -> None:
    pixels, waveform = _golden_data("timpel_healthy_volunteer_1_60-120s", "raw")
    breaths = _breaths(waveform)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = find_pixel_breaths(
            pixels, waveform, breaths, phase_correction_mode="phase shift"
        )

    messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
    assert len(messages) == 2
    no_peak = result.metadata["skipped_pixels"][NO_CORRELATION_PEAK]
    assert f"{len(no_peak)} pixel(s) have no cross-correlation peak" in messages[0]
    assert "({}, {})".format(*no_peak[0]) in messages[0]
    assert "more than 25% of their breaths left out" in messages[1]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_fewer_than_three_breaths_give_no_pixel_breaths(count: int) -> None:
    pixels, waveform = _golden_data(DATASETS[0], "filtered")

    result = find_pixel_breaths(pixels, waveform, _breaths(waveform)[:count])

    assert result.start_time.shape == (count, 32, 32)
    assert not result.has_breath.any()


def test_the_input_is_left_unchanged() -> None:
    pixels, waveform = _golden_data(DATASETS[0], "raw")
    values, waveform_values = pixels.values.copy(), waveform.values.copy()

    for mode in MODES:
        find_pixel_breaths(
            pixels, waveform, _breaths(waveform), phase_correction_mode=mode
        )

    np.testing.assert_array_equal(pixels.values, values)
    np.testing.assert_array_equal(waveform.values, waveform_values)


# Breaths at the edges of the data


def _breathing_images(duration: float) -> tuple[Signal, Signal]:
    """2 x 2 images at 20 Hz with one breath every 3 s, and their sum."""

    time = np.arange(int(duration * 20)) / 20.0
    breathing = np.sin(2 * np.pi * time / 3.0)
    pixels = _signal(
        10.0 + breathing[:, None, None] * np.array([[1.0, 2.0], [3.0, 4.0]]),
        time,
        name="pixels",
    )
    return pixels, _signal(global_impedance(pixels).values, time, name="global")


def _sliced(signal: Signal, n_samples: int) -> Signal:
    return _signal(signal.values[:n_samples], signal.time[:n_samples], name=signal.name)


def test_breaths_after_the_end_of_the_data_have_no_pixel_breaths() -> None:
    """The last two global breaths end after the data, as for the breaths of
    a whole recording on a slice of its EIT data."""

    pixels, waveform = _breathing_images(42.0)
    breaths = _breaths(waveform)
    middles = [breath.extremum_time for breath in breaths]
    n_samples = int(np.searchsorted(pixels.time, (middles[-3] + middles[-2]) / 2))
    short_pixels, short_waveform = (
        _sliced(pixels, n_samples),
        _sliced(waveform, n_samples),
    )
    inside = breaths[:-2]

    # The sign of each pixel comes from the same breaths: three of them end
    # after the data.
    with pytest.warns(UserWarning, match="find_pixel_breaths: 3 of 13 breaths"):
        result = find_pixel_breaths(short_pixels, short_waveform, breaths)
    with pytest.warns(UserWarning, match="find_pixel_breaths: 1 of 11 breaths"):
        expected = find_pixel_breaths(short_pixels, short_waveform, inside)

    # Pixel breath i uses the ends of inspiration of global breaths i - 1,
    # i and i + 1, so global breaths -3, -2 and -1 have no pixel breaths.
    assert result.has_breath[1:-3].all()
    assert not result.has_breath[-3:].any()
    np.testing.assert_array_equal(_times(result)[:-3], _times(expected)[:-1])


def test_a_pixel_with_every_breath_left_out_is_listed_with_a_warning() -> None:
    pixels, waveform = _breathing_images(30.0)
    values = pixels.values.copy()
    values[:, 1, 1] = 1.0
    values[100, 1, 1] = 5.0
    spike = _signal(values, pixels.time, name="pixels")

    with pytest.warns(RuntimeWarning, match=r"1 pixel\(s\) have more than 25%"):
        result = find_pixel_breaths(spike, waveform, _breaths(waveform))

    assert result.metadata["skipped_pixels"][TOO_MANY_BREATHS_SKIPPED] == [[1, 1]]
    assert not result.has_breath[:, 1, 1].any()
    assert result.has_breath[1:-1, 0, 0].all()


def test_no_pixel_breaths_give_a_warning_with_the_reasons() -> None:
    pixels, waveform = _breathing_images(30.0)

    with pytest.warns(RuntimeWarning, match="no pixel has a breath") as caught:
        result = find_pixel_breaths(
            pixels, waveform, _breaths(waveform), sign_breaths=[]
        )

    assert not result.has_breath.any()
    assert result.metadata["skipped_pixels"][NO_TIV] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    [warning] = [w for w in caught if "no pixel has a breath" in str(w.message)]
    assert (
        "4 pixel(s): the TIV is NaN in every breath for the sign of the pixel"
        in str(warning.message)
    )


def test_warnings_about_the_sign_breaths_name_find_pixel_breaths() -> None:
    pixels, waveform = _breathing_images(30.0)
    breaths = _breaths(waveform)
    late = BreathEvent("eit", 28.0, 31.0, extremum_time=29.5)
    expected = f"find_pixel_breaths: 1 of {len(breaths) + 1} breaths"

    with pytest.warns(UserWarning, match=expected) as caught:
        find_pixel_breaths(pixels, waveform, breaths, sign_breaths=[*breaths, late])

    [warning] = [w for w in caught if expected in str(w.message)]
    assert warning.filename == __file__


# Wrong input


def test_wrong_input_is_refused() -> None:
    pixels, waveform = _golden_data(DATASETS[0], "filtered")
    breaths = _breaths(waveform)

    with pytest.raises(ValueError, match="Unknown phase correction mode"):
        find_pixel_breaths(pixels, waveform, breaths, phase_correction_mode="both")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="extremum_time"):
        find_pixel_breaths(pixels, waveform, [BreathEvent("eit", 1.0, 2.0)])
    with pytest.raises(ValueError, match=r"shape \(time, row, column\)"):
        find_pixel_breaths(waveform, waveform, breaths)
    with pytest.raises(ValueError, match="one value per sample"):
        find_pixel_breaths(pixels, pixels, breaths)
    half = len(waveform.time) // 2
    with pytest.raises(ValueError, match="on the time axis of the EIT images"):
        find_pixel_breaths(pixels, _sliced(waveform, half), breaths)
    later = _signal(waveform.values, waveform.time + 1.0, name="global")
    with pytest.raises(ValueError, match="on the time axis of the EIT images"):
        find_pixel_breaths(pixels, later, breaths)
    with pytest.raises(TypeError, match="needs BreathEvent objects"):
        find_pixel_breaths(pixels, waveform, [(0.0, 1.0, 2.0)])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="needs BreathEvent objects"):
        find_pixel_breaths(pixels, waveform, breaths, sign_breaths=[1.0])  # type: ignore[list-item]
