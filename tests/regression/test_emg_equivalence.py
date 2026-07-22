"""Milestone 2.7 - EMG regression tests (plan_stage2.md Sec 25).

`ReSurfEMGAdapter` is a thin wrapper (Stage 1): it does not reimplement
`resurfemg`'s algorithms, it calls them directly. These tests pin that
down - on synthetic data (no private/clinical recordings needed, per
plan_stage2.md Sec 26), the adapter's default preprocessing/detection paths
must reproduce calling the same `resurfemg` functions directly, to a
documented tolerance. Any future change that makes the adapter transform
the data before/after calling `resurfemg` (rounding, unit conversion, a
different default parameter, ...) will show up here as a regression.

Tolerance: exact equality is expected (`atol=0, rtol=0`) for the
preprocessing arrays, since the adapter passes the same array straight
into the same `resurfemg` function with the same arguments - any
divergence at all means the wrapper is no longer a pure pass-through.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.adapters import ReSurfEMGAdapter

pytest.importorskip("resurfemg")


def _synthetic_emg_signal(
    fs: float = 1000.0, duration_seconds: float = 5.0
) -> np.ndarray:
    """A synthetic EMG-like signal: a carrier burst modulated at a respiratory rate."""

    time = np.arange(int(fs * duration_seconds)) / fs
    respiratory_rate_hz = 0.3
    envelope = 0.5 * (1 + np.sin(2 * np.pi * respiratory_rate_hz * time))
    carrier = np.sin(2 * np.pi * 100 * time)
    rng = np.random.default_rng(seed=0)
    noise = rng.normal(scale=0.01, size=time.shape)
    return envelope * carrier + noise


def test_preprocess_reproduces_resurfemg_filtering_and_envelope_exactly():
    from resurfemg.preprocessing.envelope import full_rolling_arv
    from resurfemg.preprocessing.filtering import emg_bandpass_butter

    fs = 1000.0
    raw = _synthetic_emg_signal(fs=fs)
    high_pass_hz = 80.0
    low_pass_hz = 250.0
    envelope_window_seconds = 0.5

    expected_filtered = emg_bandpass_butter(
        emg_raw=raw, high_pass=high_pass_hz, low_pass=low_pass_hz, fs_emg=fs
    )
    envelope_window_samples = int(envelope_window_seconds * fs)
    expected_envelope = full_rolling_arv(expected_filtered, envelope_window_samples)

    adapter = ReSurfEMGAdapter()
    processed = adapter.preprocess(
        {"array": [raw], "metadata": {"fs": fs}},
        high_pass_hz=high_pass_hz,
        low_pass_hz=low_pass_hz,
        envelope_window_seconds=envelope_window_seconds,
    )

    np.testing.assert_array_equal(processed["filtered"], expected_filtered)
    np.testing.assert_array_equal(processed["envelope"], expected_envelope)


def test_detect_breaths_reproduces_resurfemg_peak_detection_exactly():
    from resurfemg.postprocessing.event_detection import detect_emg_breaths

    fs = 1000.0
    raw = _synthetic_emg_signal(fs=fs)
    adapter = ReSurfEMGAdapter()
    processed = adapter.preprocess({"array": [raw], "metadata": {"fs": fs}})

    min_width_samples = int(1.0 * fs)
    expected_peaks = detect_emg_breaths(
        emg_env=processed["envelope"], min_peak_width_s=min_width_samples
    )

    events = adapter.detect_breaths(processed)

    actual_peak_indices = [round(event.peak_time * fs) for event in events]
    assert actual_peak_indices == [int(p) for p in expected_peaks]
