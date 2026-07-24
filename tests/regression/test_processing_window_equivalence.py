"""Regression tests for shared window primitives against upstream packages."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from m3resp.processing.windows import (
    moving_average,
    naive_rolling_rms,
    rolling_arv,
    rolling_arv_ci,
    rolling_rms,
    rolling_rms_ci,
)


def _add_sibling_checkout_to_path(name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sibling = repo_root.parent / name
    if sibling.exists() and str(sibling) not in sys.path:
        sys.path.insert(0, os.fspath(sibling))


def _synthetic_signal(fs: float = 1000.0, duration_seconds: float = 3.0):
    time = np.arange(int(fs * duration_seconds)) / fs
    return (
        np.sin(2 * np.pi * 0.3 * time)
        + 0.3 * np.sin(2 * np.pi * 85 * time)
        + 0.05 * np.sin(2 * np.pi * 180 * time)
    )


def test_moving_average_reproduces_eitprocessing_exactly():
    _add_sibling_checkout_to_path("eitprocessing")
    pytest.importorskip("eitprocessing")
    from eitprocessing.features.moving_average import MovingAverage

    values = _synthetic_signal(fs=20.0, duration_seconds=20.0)

    expected = MovingAverage(
        window_size=30,
        window_function=np.blackman,
        padding_type="edge",
    ).apply(values)
    actual = moving_average(
        values,
        window_size=30,
        window_function=np.blackman,
        padding_type="edge",
    )

    np.testing.assert_array_equal(actual, expected)


def test_moving_average_shortens_overlong_window_like_eitprocessing():
    _add_sibling_checkout_to_path("eitprocessing")
    pytest.importorskip("eitprocessing")
    from eitprocessing.features.moving_average import MovingAverage

    values = np.array([1.0, 2.0, 4.0, 8.0])

    expected = MovingAverage(window_size=99).apply(values)
    actual = moving_average(values, window_size=99)

    np.testing.assert_array_equal(actual, expected)


def test_resurfemg_rms_and_arv_envelopes_match_shared_windows_exactly():
    _add_sibling_checkout_to_path("ReSurfEMG")
    pytest.importorskip("resurfemg")
    from resurfemg.preprocessing.envelope import (
        full_rolling_arv,
        full_rolling_rms,
    )
    from resurfemg.preprocessing.envelope import (
        naive_rolling_rms as resurfemg_naive_rolling_rms,
    )

    values = _synthetic_signal()
    window_length = 250

    np.testing.assert_array_equal(
        rolling_rms(values, window_length=window_length),
        full_rolling_rms(values, window_length),
    )
    np.testing.assert_array_equal(
        naive_rolling_rms(values, window_length=window_length),
        resurfemg_naive_rolling_rms(values, window_length),
    )
    np.testing.assert_array_equal(
        rolling_arv(values, window_length=window_length),
        full_rolling_arv(values, window_length),
    )


def test_resurfemg_rolling_confidence_intervals_match_shared_windows_exactly():
    _add_sibling_checkout_to_path("ReSurfEMG")
    pytest.importorskip("resurfemg")
    pytest.importorskip("scipy")
    from resurfemg.preprocessing.envelope import rolling_arv_ci as resurfemg_arv_ci
    from resurfemg.preprocessing.envelope import rolling_rms_ci as resurfemg_rms_ci

    values = _synthetic_signal(duration_seconds=1.0)
    window_length = 100
    alpha = 0.1

    expected_rms = resurfemg_rms_ci(values, window_length, alpha=alpha)
    actual_rms = rolling_rms_ci(values, window_length=window_length, alpha=alpha)
    np.testing.assert_array_equal(actual_rms[0], expected_rms[0])
    np.testing.assert_array_equal(actual_rms[1], expected_rms[1])

    expected_arv = resurfemg_arv_ci(values, window_length, alpha=alpha)
    actual_arv = rolling_arv_ci(values, window_length=window_length, alpha=alpha)
    np.testing.assert_array_equal(actual_arv[0], expected_arv[0])
    np.testing.assert_array_equal(actual_arv[1], expected_arv[1])
