"""Options added so a pipeline can reproduce the multidomain results chain:
median envelope, merging close peaks, notch before band-pass, subtracting the
baseline, and the wavelet step keeping the preprocessing envelope method."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m3resp import M3Session
from m3resp.processing.peaks import merge_close_peaks
from m3resp.processing.windows import rolling_envelope, rolling_rms
from m3resp.workflows.steps.emg.baseline import subtract_baseline

FS = 2000.0


def _noise(seconds=2.0, seed=0):
    return np.random.default_rng(seed).normal(0.0, 1.0, int(seconds * FS))


class TestMedianEnvelope:
    def test_matches_a_centred_rolling_median_of_the_absolute_signal(self):
        values = _noise()

        envelope = rolling_envelope(values, window_length=1700, method="median")

        expected = (
            pd.Series(np.abs(values))
            .rolling(1700, min_periods=1, center=True)
            .median()
            .to_numpy()
        )
        np.testing.assert_array_equal(envelope, expected)

    def test_a_short_spike_barely_moves_it(self):
        """A heartbeat leftover is a short spike; the median ignores it where
        RMS does not."""

        values = _noise() * 0.01
        spiked = values.copy()
        spiked[2000:2020] = 5.0
        centre = 2010

        median_change = (
            rolling_envelope(spiked, window_length=1000, method="median")[centre]
            - rolling_envelope(values, window_length=1000, method="median")[centre]
        )
        rms_change = (
            rolling_rms(spiked, window_length=1000)[centre]
            - rolling_rms(values, window_length=1000)[centre]
        )
        assert median_change < 0.01 * rms_change


class TestMergeClosePeaks:
    def test_close_peaks_keep_the_higher_one(self):
        values = np.zeros(100)
        values[[10, 12, 50, 53]] = [1.0, 2.0, 3.0, 1.0]

        kept = merge_close_peaks([10, 12, 50, 53], values, min_distance_samples=5)

        assert kept.tolist() == [12, 50]

    def test_a_tie_keeps_the_earlier_peak(self):
        values = np.zeros(20)
        values[[4, 6]] = 1.0

        assert merge_close_peaks([6, 4], values, min_distance_samples=5).tolist() == [4]

    def test_peaks_far_enough_apart_are_all_kept(self):
        values = np.ones(100)

        kept = merge_close_peaks([10, 20, 30], values, min_distance_samples=10)

        assert kept.tolist() == [10, 20, 30]


def test_subtract_baseline_clips_at_zero_and_passes_a_zero_baseline():
    session = M3Session()
    processed = {"envelope": np.array([1.0, 2.0, 0.5, 3.0]), "fs": FS}
    baseline = np.array([1.5, 1.0, 1.0, 1.0])

    result = subtract_baseline(session, processed, baseline)

    corrected = result["processed_emg_baseline_subtracted"]
    np.testing.assert_array_equal(corrected["envelope"], [0.0, 1.0, 0.0, 2.0])
    np.testing.assert_array_equal(result["baseline_subtracted"], np.zeros(4))
    np.testing.assert_array_equal(
        corrected["envelope_before_baseline_subtraction"], processed["envelope"]
    )
    assert session.processed["emg"] is corrected


def test_subtract_baseline_refuses_a_baseline_of_another_length():
    with pytest.raises(ValueError, match="same length"):
        subtract_baseline(M3Session(), {"envelope": np.ones(4)}, np.ones(3))


class TestNotchBeforeBandpass:
    @staticmethod
    def _preprocess(**kwargs):
        pytest.importorskip("resurfemg")
        from m3resp.adapters import ReSurfEMGAdapter

        time = np.arange(int(4 * FS)) / FS
        signal = _noise(4.0) + 5.0 * np.sin(2 * np.pi * 50.0 * time)
        recording = {"array": np.asarray([signal]), "metadata": {"fs": FS}}
        settings = {"notch_base_frequency": 50.0, "notch_max_frequency": 950.0}
        return signal, ReSurfEMGAdapter().preprocess(recording, **settings, **kwargs)

    def test_notch_first_equals_notching_the_raw_signal_then_band_passing(self):
        # `_preprocess` skips the test when resurfemg is missing, so it runs
        # before the resurfemg import below.
        signal, processed = self._preprocess(notch_before_bandpass=True)
        from resurfemg.preprocessing.filtering import emg_bandpass_butter

        from m3resp.processing.filters import harmonic_notch_filter

        expected = emg_bandpass_butter(
            emg_raw=harmonic_notch_filter(
                signal, base_frequency=50.0, sample_frequency=FS, max_frequency=950.0
            ),
            high_pass=20.0,
            low_pass=500.0,
            fs_emg=FS,
        )
        np.testing.assert_array_equal(processed["filtered"], expected)
        assert processed["filter"]["notch_before_bandpass"] is True

    def test_default_order_is_unchanged(self):
        signal, processed = self._preprocess()
        from resurfemg.preprocessing.filtering import emg_bandpass_butter

        from m3resp.processing.filters import harmonic_notch_filter

        expected = harmonic_notch_filter(
            emg_bandpass_butter(
                emg_raw=signal, high_pass=20.0, low_pass=500.0, fs_emg=FS
            ),
            base_frequency=50.0,
            sample_frequency=FS,
            max_frequency=950.0,
        )
        np.testing.assert_array_equal(processed["filtered"], expected)


def test_wavelet_step_keeps_the_preprocessing_envelope_method():
    """It used to rebuild the envelope as ARV whatever preprocessing used."""

    pytest.importorskip("resurfemg")
    from m3resp.workflows.steps.emg.ecg_wavelet import ecg_wavelet_denoising

    session = M3Session()
    # Long enough for the 7.5 s background window around each heartbeat.
    filtered = _noise(20.0)
    processed = {
        "fs": FS,
        "channel": 0,
        "filtered": filtered,
        "envelope": np.abs(filtered),
        "filter": {"envelope_window_seconds": 0.5, "envelope_method": "median"},
    }
    peaks = np.arange(1000, len(filtered) - 1000, 1600)

    result = ecg_wavelet_denoising(session, processed, peaks, source="filtered")

    after = result["processed_emg_after_ecg"]
    assert after["filter"]["envelope_method"] == "median"
    np.testing.assert_array_equal(
        after["envelope"],
        rolling_envelope(after["ecg_cleaned"], window_length=1000, method="median"),
    )
