"""Argument checks, captures and the less-used filters in
`m3resp.processing.filters` (band-stop, harmonic notch, power loss).

`test_processing_filters.py` and `test_processing_butterworth_checks.py`
cover the main band-pass/low-pass/high-pass behaviour.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from m3resp.core.exceptions import OptionalDependencyError
from m3resp.processing import filters

FS = 1000.0


def _sines(*frequencies: float, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    return sum(np.sin(2 * np.pi * f * t) for f in frequencies)


def _amplitude_at(values: np.ndarray, frequency: float) -> float:
    spectrum = np.abs(np.fft.rfft(values)) / (len(values) / 2)
    freqs = np.fft.rfftfreq(len(values), 1 / FS)
    return float(spectrum[np.argmin(np.abs(freqs - frequency))])


class TestButterworthArgumentChecks:
    def test_unknown_filter_type(self):
        with pytest.raises(ValueError, match="filter_type must be one of"):
            filters.butterworth_filter(
                _sines(10),
                filter_type="allpass",
                cutoff_frequency=10,
                sample_frequency=FS,
            )

    @pytest.mark.parametrize("cutoff", ["10", True, None, (5, 10)])
    def test_lowpass_cutoff_must_be_one_number(self, cutoff):
        with pytest.raises(TypeError, match="must be numeric for low/high pass"):
            filters.lowpass_filter(
                _sines(10), cutoff_frequency=cutoff, sample_frequency=FS
            )

    @pytest.mark.parametrize("cutoff", [10.0, "10-20", b"ab"])
    def test_bandpass_cutoff_must_be_a_sequence(self, cutoff):
        with pytest.raises(TypeError, match="two-value sequence"):
            filters.bandpass_filter(
                _sines(10), cutoff_frequency=cutoff, sample_frequency=FS
            )

    def test_bandpass_cutoff_must_have_two_values(self):
        with pytest.raises(ValueError, match="must contain two values"):
            filters.bandpass_filter(
                _sines(10), cutoff_frequency=(5, 10, 20), sample_frequency=FS
            )

    def test_bandpass_cutoff_values_must_be_numeric(self):
        with pytest.raises(TypeError, match="values must be numeric"):
            filters.bandpass_filter(
                _sines(10), cutoff_frequency=(5, "10"), sample_frequency=FS
            )

    def test_numpy_array_cutoff_is_accepted(self):
        captures: dict = {}
        filters.bandpass_filter(
            _sines(10),
            cutoff_frequency=np.array([5.0, 50.0]),
            sample_frequency=FS,
            captures=captures,
        )
        assert captures["high_pass_frequency"] == 5.0
        assert captures["low_pass_frequency"] == 50.0

    @pytest.mark.parametrize(
        ("order", "error", "match"),
        [
            (2.5, TypeError, "whole number"),
            (True, TypeError, "whole number"),
            (0, ValueError, "order must be positive"),
        ],
    )
    def test_order_checks(self, order, error, match):
        with pytest.raises(error, match=match):
            filters.lowpass_filter(
                _sines(10), cutoff_frequency=20, sample_frequency=FS, order=order
            )

    @pytest.mark.parametrize(
        ("fs", "error", "match"),
        [
            ("1000", TypeError, "sample_frequency must be numeric"),
            (np.bool_(True), TypeError, "sample_frequency must be numeric"),
            (-1.0, ValueError, "sample_frequency must be positive"),
        ],
    )
    def test_sample_frequency_checks(self, fs, error, match):
        with pytest.raises(error, match=match):
            filters.lowpass_filter(_sines(10), cutoff_frequency=20, sample_frequency=fs)

    def test_non_finite_samples_are_rejected(self):
        values = _sines(10)
        values[5] = np.nan
        with pytest.raises(ValueError, match="NaN or infinite"):
            filters.lowpass_filter(values, cutoff_frequency=20, sample_frequency=FS)


class TestCaptures:
    def test_lowpass_records_low_pass_frequency(self):
        captures: dict = {}
        values = _sines(10)
        filtered = filters.lowpass_filter(
            values, cutoff_frequency=20, sample_frequency=FS, captures=captures
        )
        assert captures["low_pass_frequency"] == 20.0
        assert captures["sample_frequency"] == FS
        np.testing.assert_array_equal(captures["unfiltered_data"], values)
        np.testing.assert_array_equal(captures["filtered_data"], filtered)

    def test_highpass_records_high_pass_frequency(self):
        captures: dict = {}
        filters.highpass_filter(
            _sines(10), cutoff_frequency=5, sample_frequency=FS, captures=captures
        )
        assert captures["high_pass_frequency"] == 5.0

    def test_bandstop_appends_each_band(self):
        captures: dict = {}
        values = _sines(10)
        filters.bandstop_filter(
            values, cutoff_frequency=(45, 55), sample_frequency=FS, captures=captures
        )
        filters.bandstop_filter(
            values, cutoff_frequency=(95, 105), sample_frequency=FS, captures=captures
        )
        assert captures["frequency_bands"] == [(45.0, 55.0), (95.0, 105.0)]

    def test_bandpass_capture_requires_a_tuple(self):
        with pytest.raises(TypeError, match=r"bandpass cutoff must be \(low, high\)"):
            filters._capture_butterworth_parameters({}, "bandpass", 10.0)


class TestBandstopAndHarmonicNotch:
    def test_bandstop_removes_the_band_and_keeps_the_rest(self):
        values = _sines(10, 50)
        filtered = filters.bandstop_filter(
            values, cutoff_frequency=(45, 55), sample_frequency=FS
        )
        assert _amplitude_at(filtered, 50) < 0.05
        assert _amplitude_at(filtered, 10) > 0.9

    def test_harmonic_notch_removes_every_harmonic(self):
        values = _sines(10, 50, 100, 150)
        filtered = filters.harmonic_notch_filter(
            values, base_frequency=50, sample_frequency=FS
        )
        for harmonic in (50, 100, 150):
            assert _amplitude_at(filtered, harmonic) < 0.1
        assert _amplitude_at(filtered, 10) > 0.9

    def test_harmonic_notch_stops_at_max_frequency(self):
        values = _sines(50, 150)
        filtered = filters.harmonic_notch_filter(
            values, base_frequency=50, sample_frequency=FS, max_frequency=120
        )
        assert _amplitude_at(filtered, 50) < 0.1
        assert _amplitude_at(filtered, 150) > 0.9

    def test_harmonic_bandstop_with_distance(self):
        captures: dict = {}
        values = _sines(10, 50, 100)
        filtered = filters.harmonic_notch_filter(
            values,
            base_frequency=50,
            sample_frequency=FS,
            max_frequency=120,
            distance=3,
            captures=captures,
        )
        assert captures["frequency_bands"] == [(47.0, 53.0), (97.0, 103.0)]
        assert _amplitude_at(filtered, 50) < 0.05
        assert _amplitude_at(filtered, 100) < 0.05
        assert _amplitude_at(filtered, 10) > 0.9

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason=(
            "Bug: the band's upper edge is clipped to `nyquist - eps`, but eps "
            "(~2e-16) is below float resolution at e.g. 500 Hz, so the edge "
            "stays at Nyquist and SciPy rejects the band."
        ),
    )
    def test_harmonic_band_reaching_nyquist_is_clipped(self):
        values = _sines(10, 490)
        filtered = filters.harmonic_notch_filter(
            values, base_frequency=490, sample_frequency=FS, distance=20
        )
        assert _amplitude_at(filtered, 490) < 0.1

    @pytest.mark.parametrize("base_frequency", [0, -50])
    def test_harmonic_notch_needs_positive_base_frequency(self, base_frequency):
        with pytest.raises(ValueError, match="base_frequency must be positive"):
            filters.harmonic_notch_filter(
                _sines(10), base_frequency=base_frequency, sample_frequency=FS
            )


class TestComputePowerLoss:
    def test_unchanged_signal_loses_no_power(self):
        values = _sines(10, 50)
        loss = filters.compute_power_loss(
            values, values, original_frequency=FS, processed_frequency=FS
        )
        assert loss == pytest.approx(0.0, abs=1e-9)

    def test_removing_one_of_two_equal_sines_loses_about_half(self):
        values = _sines(10, 50)
        filtered = filters.lowpass_filter(
            values, cutoff_frequency=25, sample_frequency=FS, order=6
        )
        loss = filters.compute_power_loss(
            values,
            filtered,
            original_frequency=FS,
            processed_frequency=FS,
            n_segment=500,
        )
        assert loss == pytest.approx(50.0, abs=5.0)


def test_missing_scipy_raises_optional_dependency_error(monkeypatch):
    real_import = builtins.__import__

    def _fail_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_scipy)

    with pytest.raises(OptionalDependencyError, match="Filtering requires SciPy"):
        filters.lowpass_filter(_sines(10), cutoff_frequency=20, sample_frequency=FS)
