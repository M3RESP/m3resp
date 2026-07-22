"""Milestone 2.7 - EIT regression tests (plan_stage2.md Sec 25).

`EITProcessingAdapter` is a thin wrapper (Stage 1): filtering is delegated
straight to `eitprocessing.filters.butterworth_filters.ButterworthFilter`,
with no transformation of the array before/after. This test drives that
Butterworth path through the adapter's public `preprocess()` on synthetic
data (no private/clinical recordings needed, per plan_stage2.md Sec 26) and
checks it reproduces calling `ButterworthFilter` directly on the same raw
array, to a documented tolerance.

Tolerance: exact equality (`atol=0, rtol=0`) - the adapter passes the same
array into the same filter with the same parameters, so any divergence means
the wrapper is doing something other than a pure pass-through.

The heavier default path (rate detection + MDN filtering + breath-interval
dependent TIV/EELI/pixel-TIV) is exercised end-to-end against a *real*
committed sample file in `tests/test_eit.py::
test_eit_real_data_pipeline_uses_committed_sample` instead of here: those
algorithms need a realistic multi-minute respiratory waveform to produce a
meaningful breath rate, which a short synthetic fixture cannot provide
without reimplementing `eitprocessing`'s own test data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from m3resp.adapters import EITProcessingAdapter

pytest.importorskip("eitprocessing")


class _FakeCollection(dict):
    def add(self, value: Any, overwrite: bool = False) -> None:
        if not overwrite and value.label in self:
            raise KeyError(value.label)
        self[value.label] = value


class _FakeContinuousData:
    def __init__(self, label: str, time: np.ndarray, values: np.ndarray):
        self.label = label
        self.time = time
        self.values = values


class _FakeEITData:
    def __init__(
        self, pixel_impedance: np.ndarray, sample_frequency: float, time: np.ndarray
    ):
        self.label = "raw"
        self.pixel_impedance = pixel_impedance
        self.sample_frequency = sample_frequency
        self.time = time

    def get_summed_impedance(self, return_label: str | None = None, **_: Any):
        summed = self.pixel_impedance.sum(axis=(1, 2))
        return _FakeContinuousData(
            return_label or f"summed {self.label}", self.time, summed
        )


class _FakeSequence:
    def __init__(self, raw_eit_data: _FakeEITData):
        self.eit_data = _FakeCollection(raw=raw_eit_data)
        self.continuous_data = _FakeCollection()
        self.sparse_data = _FakeCollection()
        self.interval_data = _FakeCollection()


def _synthetic_pixel_impedance(
    fs: float = 20.0, duration_seconds: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """A single-pixel global-impedance-like sine wave, shaped like real
    `pixel_impedance` (samples, rows, cols)."""

    n_samples = int(fs * duration_seconds)
    time = np.arange(n_samples) / fs
    respiratory_rate_hz = 0.3
    signal = 1.0 + 0.2 * np.sin(2 * np.pi * respiratory_rate_hz * time)
    return signal.reshape(-1, 1, 1), time


def test_lowpass_filter_path_reproduces_butterworth_filter_exactly():
    from eitprocessing.filters.butterworth_filters import ButterworthFilter

    fs = 20.0
    pixel_impedance, time = _synthetic_pixel_impedance(fs=fs)
    raw_eit = _FakeEITData(
        pixel_impedance=pixel_impedance, sample_frequency=fs, time=time
    )
    sequence = _FakeSequence(raw_eit)

    lowpass_hz = 1.0
    filter_order = 4

    expected_filter = ButterworthFilter(
        filter_type="lowpass",
        cutoff_frequency=lowpass_hz,
        order=filter_order,
        sample_frequency=fs,
    )
    expected_filtered = expected_filter.apply(np.nan_to_num(pixel_impedance), axis=0)

    adapter = EITProcessingAdapter(loader=lambda *a, **k: sequence)
    processed = adapter.preprocess(
        sequence,
        filter_mode="lowpass",
        lowpass_hz=lowpass_hz,
        filter_order=filter_order,
        compute_breath_intervals=False,
        compute_continuous_tiv=False,
        compute_eeli=False,
        compute_pixel_tiv=False,
    )

    np.testing.assert_array_equal(
        processed["filtered_eit"].pixel_impedance, expected_filtered
    )


def test_bandpass_filter_path_reproduces_butterworth_filter_exactly():
    from eitprocessing.filters.butterworth_filters import ButterworthFilter

    fs = 20.0
    pixel_impedance, time = _synthetic_pixel_impedance(fs=fs)
    raw_eit = _FakeEITData(
        pixel_impedance=pixel_impedance, sample_frequency=fs, time=time
    )
    sequence = _FakeSequence(raw_eit)

    highpass_hz = 0.05
    lowpass_hz = 1.0
    filter_order = 4

    expected_filter = ButterworthFilter(
        filter_type="bandpass",
        cutoff_frequency=(highpass_hz, lowpass_hz),
        order=filter_order,
        sample_frequency=fs,
    )
    expected_filtered = expected_filter.apply(np.nan_to_num(pixel_impedance), axis=0)

    adapter = EITProcessingAdapter(loader=lambda *a, **k: sequence)
    processed = adapter.preprocess(
        sequence,
        filter_mode="bandpass",
        highpass_hz=highpass_hz,
        lowpass_hz=lowpass_hz,
        filter_order=filter_order,
        compute_breath_intervals=False,
        compute_continuous_tiv=False,
        compute_eeli=False,
        compute_pixel_tiv=False,
    )

    np.testing.assert_array_equal(
        processed["filtered_eit"].pixel_impedance, expected_filtered
    )
