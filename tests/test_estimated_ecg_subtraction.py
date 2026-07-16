"""Tests for the paper-based Estimated ECG Subtraction method."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, Signal
from m3resp.processing.ecg import (
    _correct_qrs_periodicity,
    _template_segments,
    estimated_ecg_subtraction,
)
from m3resp.workflows.steps.emg import ecg_estimated_subtraction


def _synthetic_contaminated_emg() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    sample_frequency = 1000.0
    time = np.arange(12 * int(sample_frequency)) / sample_frequency
    rng = np.random.default_rng(7)
    sos = butter(
        4,
        (80, 220),
        btype="bandpass",
        fs=sample_frequency,
        output="sos",
    )
    clean_emg = 0.12 * sosfiltfilt(sos, rng.normal(size=time.size))
    beat_times = np.arange(0.8, 11.4, 0.8)
    ecg = np.zeros_like(time)
    for index, beat_time in enumerate(beat_times):
        scale = 1 + 0.15 * np.sin(index)
        ecg += scale * (
            -0.6 * np.exp(-0.5 * ((time - (beat_time - 0.025)) / 0.008) ** 2)
            + 2.5 * np.exp(-0.5 * ((time - beat_time) / 0.009) ** 2)
            - 0.9 * np.exp(-0.5 * ((time - (beat_time + 0.035)) / 0.012) ** 2)
            + 0.25 * np.exp(-0.5 * ((time - (beat_time + 0.16)) / 0.035) ** 2)
        )
    return clean_emg + ecg, clean_emg, ecg, beat_times, sample_frequency


def _processed_emg(values: np.ndarray, sample_frequency: float) -> dict[str, Any]:
    return {
        "filtered": values,
        "raw_channel": values,
        "envelope": np.abs(values),
        "fs": sample_frequency,
        "channel": 0,
        "metadata": {"labels": ["EMGdi"], "units": ["uV"]},
        "filter": {"envelope_window_seconds": 0.2},
    }


def test_estimated_ecg_subtraction_reduces_known_ecg_error():
    contaminated, clean, _, beat_times, fs = _synthetic_contaminated_emg()

    result = estimated_ecg_subtraction(contaminated, sample_frequency=fs)

    qrs_times = result.qrs_indices[:, 1] / fs
    assert len(qrs_times) == len(beat_times)
    np.testing.assert_allclose(qrs_times, beat_times, atol=0.005)
    assert result.cleaned.shape == contaminated.shape
    assert result.estimated_ecg.shape == contaminated.shape
    assert result.detection_signal.shape == contaminated.shape
    assert result.dynamic_threshold.shape == contaminated.shape
    assert result.normalized_segments.shape[0] == len(beat_times)
    assert result.normalized_segments.shape[1] == result.normalized_template.size

    qrs_mask = np.zeros(contaminated.size, dtype=bool)
    time = np.arange(contaminated.size) / fs
    for beat_time in beat_times:
        qrs_mask |= np.abs(time - beat_time) < 0.15
    error_before = np.mean((contaminated[qrs_mask] - clean[qrs_mask]) ** 2)
    error_after = np.mean((result.cleaned[qrs_mask] - clean[qrs_mask]) ** 2)
    assert error_after < 0.4 * error_before


def test_periodicity_correction_restores_a_missing_detection():
    detection = np.zeros(600)
    threshold = np.full(600, 0.5)
    detection[[100, 200, 300, 400, 500]] = 1.0

    corrected, rejected, restored = _correct_qrs_periodicity(
        np.asarray([100, 200, 400, 500]),
        detection_signal=detection,
        dynamic_threshold=threshold,
        tolerance=0.2,
    )

    np.testing.assert_array_equal(corrected, [100, 200, 300, 400, 500])
    assert rejected.size == 0
    np.testing.assert_array_equal(restored, [300])


def test_template_creation_skips_a_window_that_crosses_a_recording_boundary():
    signal = np.zeros(1000)
    above_threshold = np.zeros(1000, dtype=bool)
    for peak in (10, 500):
        above_threshold[max(0, peak - 20) : peak + 21] = True
        signal[peak - 10] = -1.0
        signal[peak] = 2.0
        signal[peak + 10] = -0.8

    template_peaks, qrs_indices, normalized, amplitudes = _template_segments(
        signal,
        np.asarray([10, 500]),
        above_threshold=above_threshold,
        window_samples=301,
    )

    np.testing.assert_array_equal(template_peaks, [500])
    np.testing.assert_array_equal(qrs_indices, [[490, 500, 510]])
    assert normalized.shape == (1, 301)
    assert amplitudes.shape == (1, 3)


@pytest.mark.parametrize(
    ("values", "sample_frequency", "message"),
    [
        (np.ones((2, 10)), 1000.0, "one-dimensional"),
        (np.ones(100), 1000.0, "non-zero amplitude"),
        (np.arange(100, dtype=float), 80.0, "below Nyquist"),
    ],
)
def test_estimated_ecg_subtraction_rejects_invalid_inputs(
    values: np.ndarray, sample_frequency: float, message: str
):
    with pytest.raises(ValueError, match=message):
        estimated_ecg_subtraction(values, sample_frequency=sample_frequency)


def test_implausible_detected_qrs_rate_is_rejected_before_subtraction():
    contaminated, _, _, _, fs = _synthetic_contaminated_emg()

    with pytest.raises(ValueError, match="implausibly short median QRS interval"):
        estimated_ecg_subtraction(
            contaminated,
            sample_frequency=fs,
            minimum_qrs_interval_seconds=0.9,
        )


def test_workflow_step_updates_session_and_keeps_diagnostics():
    contaminated, _, _, beat_times, fs = _synthetic_contaminated_emg()
    processed_emg = _processed_emg(contaminated, fs)
    session = M3Session()

    output = ecg_estimated_subtraction(session, processed_emg)

    cleaned = output["ees_cleaned_emg"]
    assert output["processed_emg_after_ecg"]["filtered"] is cleaned
    assert session.processed["emg"] is output["processed_emg_after_ecg"]
    assert len(output["ees_qrs_events"]) == len(beat_times)
    assert all(event.name == "ees_qrs" for event in output["ees_qrs_events"])
    assert session.events["ees_qrs"] == output["ees_qrs_events"]

    for name in (
        "ees_cleaned_signal",
        "ees_estimated_ecg_signal",
        "ees_detection_signal",
        "ees_dynamic_threshold_signal",
    ):
        assert isinstance(output[name], Signal)
        assert output[name].method == "m3resp.estimated_ecg_subtraction"

    for name in (
        "ees_candidate_peaks_result",
        "ees_corrected_peaks_result",
        "ees_rejected_peaks_result",
        "ees_restored_peaks_result",
        "ees_qrs_indices_result",
        "ees_normalized_segments_result",
        "ees_template_result",
    ):
        assert isinstance(output[name], ParameterResult)
        assert any(output[name] is item for item in session.parameter_results)


def test_workflow_step_rejects_an_unknown_source():
    contaminated, _, _, _, fs = _synthetic_contaminated_emg()
    with pytest.raises(ValueError, match="available keys"):
        ecg_estimated_subtraction(
            M3Session(), _processed_emg(contaminated, fs), source="missing"
        )
