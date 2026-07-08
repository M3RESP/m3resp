"""Regression tests for shared filter primitives against upstream packages."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from m3resp.processing.filters import (
    bandpass_filter,
    butterworth_filter,
    highpass_filter,
    lowpass_filter,
    notch_filter,
)

pytest.importorskip("scipy")


def _add_sibling_checkout_to_path(name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sibling = repo_root.parent / name
    if sibling.exists() and str(sibling) not in sys.path:
        sys.path.insert(0, os.fspath(sibling))


def _synthetic_signal(fs: float = 1000.0, duration_seconds: float = 3.0):
    time = np.arange(int(fs * duration_seconds)) / fs
    return (
        np.sin(2 * np.pi * 4 * time)
        + 0.2 * np.sin(2 * np.pi * 110 * time)
        + 0.05 * np.sin(2 * np.pi * 50 * time)
    )


def test_butterworth_filter_reproduces_eitprocessing_lowpass_exactly():
    _add_sibling_checkout_to_path("eitprocessing")
    pytest.importorskip("eitprocessing")
    from eitprocessing.filters.butterworth_filters import ButterworthFilter

    fs = 20.0
    time = np.arange(int(fs * 20)) / fs
    values = np.sin(2 * np.pi * 0.3 * time) + 0.1 * np.sin(2 * np.pi * 3 * time)
    values = values.reshape(-1, 1, 1)

    expected = ButterworthFilter(
        filter_type="lowpass",
        cutoff_frequency=1.0,
        order=4,
        sample_frequency=fs,
    ).apply(values, axis=0)
    captures: dict[str, object] = {}

    actual = butterworth_filter(
        values,
        filter_type="lowpass",
        cutoff_frequency=1.0,
        sample_frequency=fs,
        order=4,
        axis=0,
        captures=captures,
    )

    np.testing.assert_array_equal(actual, expected)
    assert captures["sample_frequency"] == fs
    assert captures["low_pass_frequency"] == 1.0


def test_resurfemg_bandpass_filter_matches_shared_bandpass_exactly():
    _add_sibling_checkout_to_path("ReSurfEMG")
    pytest.importorskip("resurfemg")
    from resurfemg.preprocessing.filtering import emg_bandpass_butter

    fs = 1000.0
    values = _synthetic_signal(fs=fs)

    expected = emg_bandpass_butter(
        values,
        high_pass=20.0,
        low_pass=250.0,
        fs_emg=fs,
        order=3,
    )
    actual = bandpass_filter(
        values,
        cutoff_frequency=(20.0, 250.0),
        sample_frequency=fs,
        order=3,
    )

    np.testing.assert_array_equal(actual, expected)


def test_resurfemg_lowpass_and_highpass_filters_match_shared_filters_exactly():
    _add_sibling_checkout_to_path("ReSurfEMG")
    pytest.importorskip("resurfemg")
    from resurfemg.preprocessing.filtering import (
        emg_highpass_butter,
        emg_lowpass_butter,
    )

    fs = 1000.0
    values = _synthetic_signal(fs=fs)

    np.testing.assert_array_equal(
        lowpass_filter(
            values,
            cutoff_frequency=120.0,
            sample_frequency=fs,
            order=3,
        ),
        emg_lowpass_butter(values, low_pass=120.0, fs_emg=fs, order=3),
    )
    np.testing.assert_array_equal(
        highpass_filter(
            values,
            cutoff_frequency=20.0,
            sample_frequency=fs,
            order=3,
        ),
        emg_highpass_butter(values, high_pass=20.0, fs_emg=fs, order=3),
    )


def test_resurfemg_notch_filter_matches_shared_notch_exactly():
    _add_sibling_checkout_to_path("ReSurfEMG")
    pytest.importorskip("resurfemg")
    from resurfemg.preprocessing.filtering import notch_filter as resurfemg_notch

    fs = 1000.0
    values = _synthetic_signal(fs=fs)

    expected = resurfemg_notch(values, f_notch=50.0, fs_emg=fs, q=30.0)
    actual = notch_filter(
        values,
        frequency=50.0,
        sample_frequency=fs,
        quality_factor=30.0,
    )

    np.testing.assert_array_equal(actual, expected)
