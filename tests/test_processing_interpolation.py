"""Tests for `m3resp.processing.interpolation.interpolate_to_time`."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from m3resp import interpolate_to_time
from m3resp.data import Signal, TimeSeries


def _ramp(time) -> Signal:
    """A signal whose value equals its time in seconds, so the expected
    interpolated value at any time point is that time point."""

    time = np.asarray(time, dtype=float)
    return Signal(
        values=time.copy(),
        time=time,
        sample_frequency=1.0,
        modality="emg",
        unit="µV",
        channel="ch1",
        metadata={"source_file": "a.poly5"},
    )


def test_reads_values_off_a_finer_time_axis():
    signal = _ramp([0.0, 1.0, 2.0])

    result = interpolate_to_time(signal, [0.0, 0.5, 1.0, 1.5, 2.0])

    np.testing.assert_allclose(result.values, [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(result.time, [0.0, 0.5, 1.0, 1.5, 2.0])
    assert result.sample_frequency == pytest.approx(2.0)


def test_follows_an_uneven_time_axis_exactly():
    signal = _ramp(np.arange(0.0, 10.0, 0.1))
    new_time = np.array([0.0, 0.31, 0.65, 1.02, 1.33, 1.71])

    result = interpolate_to_time(signal, new_time)

    np.testing.assert_allclose(result.time, new_time)
    np.testing.assert_allclose(result.values, new_time)


def test_onto_another_signals_time_points():
    emg = _ramp(np.arange(0.0, 5.0, 0.001))
    eit_time = np.arange(0.0, 4.0, 0.05) + 0.0123
    eit = Signal(values=np.zeros(eit_time.size), time=eit_time, modality="eit")

    result = interpolate_to_time(emg, eit.time)

    assert result.n_samples == eit.n_samples
    np.testing.assert_allclose(result.values, eit_time)


def test_keeps_the_signal_description_and_marks_the_interpolation():
    signal = _ramp([0.0, 1.0, 2.0])

    result = interpolate_to_time(signal, [0.5, 1.5])

    assert isinstance(result, Signal)
    assert (result.modality, result.unit, result.channel) == ("emg", "µV", "ch1")
    assert result.metadata == {"source_file": "a.poly5", "interpolation": "linear"}
    np.testing.assert_allclose(signal.values, [0.0, 1.0, 2.0])


def test_interpolates_each_channel_of_2d_values():
    time = np.array([0.0, 1.0, 2.0])
    values = np.column_stack([time, 10.0 * time])
    signal = TimeSeries(values=values, time=time)

    result = interpolate_to_time(signal, [0.5, 1.0, 1.5])

    np.testing.assert_allclose(result.values, [[0.5, 5.0], [1.0, 10.0], [1.5, 15.0]])


def test_interpolates_each_pixel_of_eit_frames():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    pixels = np.arange(6.0).reshape(2, 3)
    frames = time[:, None, None] * pixels
    signal = Signal(values=frames, time=time, modality="eit")

    result = interpolate_to_time(signal, [0.5])

    assert result.values.shape == (1, 2, 3)
    np.testing.assert_allclose(result.values[0], 0.5 * pixels)


def test_time_points_outside_the_recording_get_nan_and_a_warning():
    signal = _ramp([1.0, 2.0, 3.0])

    with pytest.warns(UserWarning, match="2 of 5 time points"):
        result = interpolate_to_time(signal, [0.5, 1.0, 2.0, 3.0, 3.5])

    np.testing.assert_allclose(result.values, [np.nan, 1.0, 2.0, 3.0, np.nan])


def test_no_warning_when_every_time_point_is_inside():
    signal = _ramp([1.0, 2.0, 3.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = interpolate_to_time(signal, [1.0, 3.0])

    np.testing.assert_allclose(result.values, [1.0, 3.0])


def test_a_single_time_point_has_no_sample_frequency():
    result = interpolate_to_time(_ramp([0.0, 1.0]), [0.25])

    np.testing.assert_allclose(result.values, [0.25])
    assert result.sample_frequency is None


@pytest.mark.parametrize(
    ("time", "message"),
    [
        ([], "no time points"),
        ([[0.0, 1.0]], "one-dimensional"),
        ([0.0, np.nan], "NaN or infinite"),
        ([0.0, 1.0, 1.0], "strictly increasing"),
        ([1.0, 0.5], "strictly increasing"),
    ],
)
def test_refuses_an_unusable_new_time_axis(time, message):
    with pytest.raises(ValueError, match=message):
        interpolate_to_time(_ramp([0.0, 1.0, 2.0]), time)


def test_refuses_a_signal_with_one_sample():
    signal = Signal(values=[3.0], time=[7.0], modality="eit")

    with pytest.raises(ValueError, match="at least 2 samples"):
        interpolate_to_time(signal, [7.0])


def test_refuses_a_signal_whose_time_goes_back():
    signal = Signal(values=[0.0, 1.0, 2.0], time=[0.0, 2.0, 1.0], modality="eit")

    with pytest.raises(ValueError, match="signal.time must be strictly increasing"):
        interpolate_to_time(signal, [0.5])


def test_the_sample_frequency_of_an_even_axis_is_exact():
    eit_time = np.arange(0.0, 60.0, 0.05)

    result = interpolate_to_time(_ramp(np.arange(0.0, 61.0, 0.001)), eit_time)

    assert result.sample_frequency == 20.0
