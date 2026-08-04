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
    """Pass-through equivalence, with `envelope_method="arv"` requested.

    `resurfemg`'s envelope helper is ARV; the adapter's *default* is RMS (the
    method respiratory-sEMG literature specifies), so ARV has to be asked for
    explicitly here for this to be an equivalence test rather than a
    comparison of two different envelopes - see
    `test_preprocess_envelope_defaults_to_rms_not_arv` below.
    """

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
        envelope_method="arv",
    )

    np.testing.assert_array_equal(processed["filtered"], expected_filtered)
    np.testing.assert_array_equal(processed["envelope"], expected_envelope)


def test_preprocess_envelope_defaults_to_rms_not_arv():
    """The default envelope is RMS, and RMS is not ARV on real bursty sEMG.

    Pins the default itself, so a silent revert to ARV (which the equivalence
    test above would still pass, since it now requests ARV explicitly) fails
    here instead of going unnoticed.
    """

    from resurfemg.preprocessing.envelope import full_rolling_arv
    from resurfemg.preprocessing.filtering import emg_bandpass_butter

    fs = 1000.0
    raw = _synthetic_emg_signal(fs=fs)
    recording = {"array": [raw], "metadata": {"fs": fs}}
    adapter = ReSurfEMGAdapter()

    processed = adapter.preprocess(recording, high_pass_hz=80.0, low_pass_hz=250.0)

    assert processed["filter"]["envelope_method"] == "rms"

    arv_envelope = full_rolling_arv(
        emg_bandpass_butter(emg_raw=raw, high_pass=80.0, low_pass=250.0, fs_emg=fs),
        int(0.5 * fs),
    )
    assert not np.allclose(processed["envelope"], arv_envelope)
    # RMS >= ARV by Cauchy-Schwarz, over every window with any variation.
    assert np.nanmean(processed["envelope"]) > np.nanmean(arv_envelope)


def test_preprocess_bandpass_defaults_to_the_literature_range():
    """20-500 Hz, capped by Nyquist. The high-pass deliberately does not sit
    low enough to double as ECG suppression - `emg.ecg_gating` owns that."""

    fs = 2000.0
    adapter = ReSurfEMGAdapter()

    processed = adapter.preprocess(
        {"array": [_synthetic_emg_signal(fs=fs)], "metadata": {"fs": fs}}
    )

    assert processed["filter"]["high_pass_hz"] == 20.0
    assert processed["filter"]["low_pass_hz"] == 500.0


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


# -- Phase 1/8 (plan/stage2/2_resurfemg_gap_migration_implementation_plan.md):
# the named adapter methods added for ECG/baseline operations must reproduce
# calling the same resurfemg functions directly, exactly. -------------------


def _synthetic_ecg_contaminated_signal(
    fs: float = 2048.0, duration_seconds: float = 20.0
) -> np.ndarray:
    neurokit2 = pytest.importorskip("neurokit2")
    ecg = np.asarray(
        neurokit2.ecg_simulate(
            duration=duration_seconds, sampling_rate=fs, heart_rate=75, random_state=42
        ),
        dtype=float,
    )
    time = np.arange(len(ecg)) / fs
    emg_like = 0.05 * np.sin(2 * np.pi * 100 * time)
    return ecg + emg_like


def test_detect_ecg_peaks_reproduces_resurfemg_exactly():
    from resurfemg.preprocessing.ecg_removal import detect_ecg_peaks

    fs = 2048.0
    signal = _synthetic_ecg_contaminated_signal(fs=fs)
    expected = detect_ecg_peaks(signal, int(fs))

    adapter = ReSurfEMGAdapter()
    actual = adapter.detect_ecg_peaks(signal, sample_frequency=fs)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fill_method", [0, 1, 2, 3])
def test_gate_ecg_reproduces_resurfemg_exactly_for_every_fill_method(fill_method):
    from resurfemg.preprocessing.ecg_removal import gating

    fs = 2048.0
    signal = _synthetic_ecg_contaminated_signal(fs=fs)
    # Peaks near (but not past) both signal boundaries, plus interior ones,
    # exercise the plan's "boundary windows" characterization requirement.
    peaks = np.array([50, 4096, 8192, len(signal) - 60])

    expected = gating(signal, peaks, gate_width=205, method=fill_method)

    adapter = ReSurfEMGAdapter()
    actual = adapter.gate_ecg(
        signal, peaks, gate_width_samples=205, fill_method=fill_method
    )

    np.testing.assert_array_equal(actual, expected)


def test_wavelet_denoise_ecg_reproduces_resurfemg_exactly_including_padding():
    from resurfemg.preprocessing.ecg_removal import wavelet_denoising

    fs = 2048.0
    # >=15 s: wavelet_denoising's internal noise-estimation window is
    # 15 * fs samples wide (see docs/pipelines.md "ECG-removal alternatives"),
    # so shorter signals raise inside resurfemg itself, independent of this
    # wrapper's own behavior. 20.002 s (not 20.0 s) gives a sample count that
    # is not already a multiple of 2**4, so real zero-padding is exercised.
    signal = _synthetic_ecg_contaminated_signal(fs=fs, duration_seconds=20.002)
    peaks = np.array([50, 4096, 8192, len(signal) - 60])

    expected_cleaned, expected_decomposition, expected_thresholds, expected_mask = (
        wavelet_denoising(signal, peaks, int(fs))
    )

    adapter = ReSurfEMGAdapter()
    actual_cleaned, actual_decomposition, actual_thresholds, actual_mask = (
        adapter.wavelet_denoise_ecg(signal, peaks, sample_frequency=fs)
    )

    np.testing.assert_array_equal(actual_cleaned, expected_cleaned)
    np.testing.assert_array_equal(actual_decomposition, expected_decomposition)
    np.testing.assert_array_equal(actual_thresholds, expected_thresholds)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    # The decomposition stays at the zero-padded length; cleaned/thresholds/
    # mask are trimmed back to the original signal length.
    assert expected_decomposition.shape[-1] != len(signal)
    assert expected_cleaned.shape[-1] == len(signal)


def test_moving_baseline_reproduces_resurfemg_exactly():
    from resurfemg.postprocessing.baseline import moving_baseline

    fs = 100.0
    envelope = _synthetic_emg_signal(fs=fs, duration_seconds=30.0) ** 2
    window_samples = int(5.0 * fs)
    step_samples = int(0.5 * fs)

    expected = moving_baseline(
        envelope, window_samples, step_samples, set_percentile=33.0
    )

    adapter = ReSurfEMGAdapter()
    actual = adapter.moving_baseline(
        envelope,
        window_samples=window_samples,
        step_samples=step_samples,
        percentile=33.0,
    )

    np.testing.assert_array_equal(actual, expected)


def test_slopesum_baseline_reproduces_resurfemg_exactly():
    from resurfemg.postprocessing.baseline import slopesum_baseline

    fs = 100.0
    envelope = _synthetic_emg_signal(fs=fs, duration_seconds=30.0) ** 2
    window_samples = int(5.0 * fs)
    step_samples = int(0.5 * fs)

    expected_baseline, expected_mean, expected_std, expected_series = slopesum_baseline(
        envelope,
        window_samples,
        step_samples,
        int(fs),
        set_percentile=33.0,
        augm_percentile=25.0,
    )

    adapter = ReSurfEMGAdapter()
    actual_baseline, actual_mean, actual_std, actual_series = adapter.slopesum_baseline(
        envelope,
        window_samples=window_samples,
        step_samples=step_samples,
        sample_frequency=fs,
        percentile=33.0,
        augmented_percentile=25.0,
    )

    np.testing.assert_array_equal(actual_baseline, expected_baseline)
    np.testing.assert_array_equal(actual_mean, expected_mean)
    np.testing.assert_array_equal(actual_std, expected_std)
    np.testing.assert_array_equal(actual_series.values, expected_series.values)
