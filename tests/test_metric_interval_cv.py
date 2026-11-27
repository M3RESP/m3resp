"""Tests for the interval duration statistics of ``metric.interval_cv``."""

from types import SimpleNamespace

import numpy as np
import pytest

from m3resp.data.events import BreathEvent, Interval
from m3resp.workflows.steps.metrics import interval_cv

# Durations 2 s, 4 s and 6 s: mean 4 s, population std sqrt(8/3) s.
DURATIONS = (2.0, 4.0, 6.0)
STARTS = (0.0, 10.0, 20.0)
EXPECTED_MEAN = 4.0
EXPECTED_STD = float(np.sqrt(8.0 / 3.0))


def _check(result: dict) -> None:
    assert result["n"] == len(DURATIONS)
    assert result["mean"] == pytest.approx(EXPECTED_MEAN, abs=1e-12)
    assert result["std"] == pytest.approx(EXPECTED_STD, abs=1e-12)
    assert result["cv"] == pytest.approx(EXPECTED_STD / EXPECTED_MEAN, abs=1e-12)


def test_durations_of_m3resp_intervals():
    intervals = [
        Interval("ventilator", start, start + duration, name="breath")
        for start, duration in zip(STARTS, DURATIONS, strict=True)
    ]
    _check(interval_cv(SimpleNamespace(intervals=intervals)))


def test_durations_of_breath_events():
    breaths = [
        BreathEvent(
            "eit",
            start,
            start + duration,
            extremum_time=start + duration / 2,
        )
        for start, duration in zip(STARTS, DURATIONS, strict=True)
    ]
    _check(interval_cv(SimpleNamespace(intervals=breaths)))


def test_durations_of_start_end_pairs():
    pairs = [
        (start, start + duration)
        for start, duration in zip(STARTS, DURATIONS, strict=True)
    ]
    _check(interval_cv(SimpleNamespace(intervals=pairs)))


def test_three_times_per_interval_raise_an_error():
    # (start, middle, end) is no (start, end) pair: middle - start would be
    # the inspiration time, not the breath duration.
    triples = [(0.0, 1.0, 2.0), (10.0, 12.0, 14.0)]
    with pytest.raises(ValueError, match=r"\(start, end\) pair"):
        interval_cv(SimpleNamespace(intervals=triples))


def test_a_single_number_raises_an_error():
    with pytest.raises(ValueError, match=r"\(start, end\) pair"):
        interval_cv(SimpleNamespace(intervals=[3.0, 4.0]))


def test_a_pair_that_ends_before_it_starts_raises_an_error():
    with pytest.raises(ValueError, match="must not be before start"):
        interval_cv(SimpleNamespace(intervals=[(0.0, 2.0), (5.0, 4.0)]))
