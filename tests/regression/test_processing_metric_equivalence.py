"""Regression tests for shared per-breath metric primitives."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from m3resp.processing.metrics import (
    amplitude_at_peaks,
    area_under_baseline,
    pseudo_slope,
    respiratory_rate_from_indices,
    tidal_variation,
    time_to_peak,
    window_integral,
)


def _add_sibling_checkout_to_path(name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sibling = repo_root.parent / name
    if sibling.exists() and str(sibling) not in sys.path:
        sys.path.insert(0, os.fspath(sibling))


def test_resurfemg_feature_primitives_match_upstream():
    pytest.importorskip("resurfemg.postprocessing.features")
    from resurfemg.postprocessing.features import (
        amplitude,
        area_under_baseline as upstream_area_under_baseline,
        pseudo_slope as upstream_pseudo_slope,
        respiratory_rate,
        time_product,
        time_to_peak as upstream_time_to_peak,
    )

    fs = 100.0
    time = np.arange(int(fs * 12)) / fs
    envelope = 0.2 + np.maximum(0, np.sin(2 * np.pi * 0.5 * time))
    baseline = np.full(envelope.shape, 0.1)
    peak_indices = np.array([50, 250, 450, 650, 850, 1050])
    start_indices = peak_indices - 35
    end_indices = peak_indices + 45
    window_samples = int(0.5 * fs)

    expected_time_to_peak = upstream_time_to_peak(envelope, start_indices, end_indices)
    actual_time_to_peak = time_to_peak(envelope, start_indices, end_indices)
    for actual, expected in zip(
        actual_time_to_peak, expected_time_to_peak, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)

    np.testing.assert_array_equal(
        pseudo_slope(envelope, start_indices, end_indices),
        upstream_pseudo_slope(envelope, start_indices, end_indices),
    )
    np.testing.assert_array_equal(
        amplitude_at_peaks(envelope, peak_indices, baseline),
        amplitude(envelope, peak_indices, baseline),
    )
    np.testing.assert_allclose(
        window_integral(envelope, fs, start_indices, end_indices, baseline),
        time_product(envelope, fs, start_indices, end_indices, baseline),
    )

    expected_area = upstream_area_under_baseline(
        envelope,
        fs,
        peak_indices,
        start_indices,
        end_indices,
        window_samples,
        baseline,
    )
    actual_area = area_under_baseline(
        envelope,
        fs,
        peak_indices,
        start_indices,
        end_indices,
        window_samples,
        baseline,
    )
    for actual, expected in zip(actual_area, expected_area, strict=True):
        np.testing.assert_allclose(actual, expected)

    expected_rate = respiratory_rate(peak_indices, fs)
    actual_rate = respiratory_rate_from_indices(peak_indices, fs)
    assert actual_rate[0] == expected_rate[0]
    np.testing.assert_array_equal(actual_rate[1], expected_rate[1])


def test_eit_tidal_variation_matches_tiv_private_calculation():
    _add_sibling_checkout_to_path("eitprocessing")
    pytest.importorskip("eitprocessing")
    from eitprocessing.datahandling.breath import Breath
    from eitprocessing.parameters.tidal_impedance_variation import TIV

    sample_time = np.arange(0, 10, 0.5)
    values = 1.0 + np.sin(2 * np.pi * 0.25 * sample_time)
    breaths = [
        Breath(0.0, 1.0, 2.0),
        None,
        Breath(4.0, 5.0, 6.0),
    ]
    calculator = TIV()

    for method in ("inspiratory", "expiratory", "mean"):
        expected = calculator._calculate_tiv_values(
            values,
            sample_time,
            breaths,
            method,
            "continuous",
        )
        actual = tidal_variation(values, sample_time, breaths, method=method)
        np.testing.assert_array_equal(actual, expected)
