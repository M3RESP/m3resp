"""Breath features are computed for every detected breath, including
breaths whose onset/offset window is marked invalid. The validity flag
from 'emg.onoffpeak_baseline_crossing' is kept alongside the features, and
added to the session's quality flags, so the user can decide which breaths
to use; the feature values themselves are never replaced with NaN. The
same holds for the Pocc pressure-time product.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.core.session import M3Session
from m3resp.processing.intervals import onoff_from_baseline_crossings
from m3resp.processing.metrics import time_to_peak as _time_to_peak
from m3resp.processing.metrics import window_integral
from m3resp.workflows.steps.emg import (
    area_under_baseline,
    onoffpeak_baseline_crossing,
    pseudo_slope,
    time_product,
    time_to_peak,
)
from m3resp.workflows.steps.ventilator import pocc_time_product

FS = 1000.0


def _two_overlapping_breaths_and_one_clean_breath() -> tuple[dict[str, Any], Any, Any]:
    """Envelope with a constant baseline of 1.0.

    Samples 500-1500 rise above baseline once but hold two peaks (800 and
    1200). The first peak's window then runs past the second peak, so the
    first breath is marked invalid. Samples 2000-2500 hold one separate
    breath (peak 2250) that is valid.
    """
    n_samples = 3000
    envelope = np.full(n_samples, 0.5)
    envelope[500:1500] = 1.5
    envelope[800] = 3.0
    envelope[1200] = 2.5
    envelope[2000:2500] = 1.5
    envelope[2250] = 2.0
    baseline = np.ones(n_samples)
    peak_indices = np.array([800, 1200, 2250])
    processed_emg = {"envelope": envelope, "fs": FS}
    return processed_emg, baseline, peak_indices


def test_invalid_breath_keeps_its_computed_features():
    processed_emg, baseline, peak_indices = (
        _two_overlapping_breaths_and_one_clean_breath()
    )
    windows = onoffpeak_baseline_crossing(
        M3Session(), processed_emg, baseline, peak_indices
    )
    starts = windows["start_indices"]
    ends = windows["end_indices"]
    validity = windows["start_end_validity"]
    assert validity.tolist() == [False, True, True]

    absolute_times, percent_times = time_to_peak(processed_emg, starts, ends)[
        "time_to_peak"
    ]
    slopes = pseudo_slope(processed_emg, starts, ends)["pseudo_slope"]
    products = time_product(processed_emg, starts, ends, baseline)["time_product"]
    areas, references = area_under_baseline(
        processed_emg, peak_indices, starts, ends, baseline
    )["area_under_baseline"]

    for values in (absolute_times, percent_times, slopes, products, areas, references):
        values = np.asarray(values, dtype=float)
        assert len(values) == len(peak_indices)
        assert np.all(np.isfinite(values)), "no breath should be set to NaN"

    expected_absolute, expected_percent = _time_to_peak(
        processed_emg["envelope"], starts, ends
    )
    np.testing.assert_array_equal(absolute_times, expected_absolute)
    np.testing.assert_array_equal(percent_times, expected_percent)
    np.testing.assert_array_equal(
        products,
        window_integral(processed_emg["envelope"], FS, starts, ends, baseline),
    )


def test_onset_offset_validity_is_added_to_the_session_quality_flags():
    processed_emg, baseline, peak_indices = (
        _two_overlapping_breaths_and_one_clean_breath()
    )
    session = M3Session()
    windows = onoffpeak_baseline_crossing(
        session, processed_emg, baseline, peak_indices
    )

    flags = windows["start_end_validity_flags"]
    assert [flag.passed for flag in flags] == [False, True, True]
    assert [flag.breath_id for flag in flags] == ["0", "1", "2"]
    for index, flag in enumerate(flags):
        assert flag.name == "start_end_validity"
        assert flag.modality == "emg"
        assert flag.metadata["peak_sample_index"] == int(peak_indices[index])
        assert flag.metadata["start_sample_index"] == int(
            windows["start_indices"][index]
        )
        assert flag.metadata["end_sample_index"] == int(windows["end_indices"][index])
        assert any(flag is stored for stored in session.quality)

    # The first breath's window runs past the second peak, so its offset
    # is the side marked invalid.
    assert flags[0].metadata["valid_start"] is True
    assert flags[0].metadata["valid_end"] is False


def test_invalid_pocc_manoeuvre_keeps_its_pressure_time_product():
    processed_emg, baseline, peak_indices = (
        _two_overlapping_breaths_and_one_clean_breath()
    )
    pressure = processed_emg["envelope"]
    starts, ends, _valid_starts, _valid_ends, valid_peaks = (
        onoff_from_baseline_crossings(pressure, baseline, peak_indices)
    )
    assert valid_peaks == [False, True, True]

    result = pocc_time_product(
        M3Session(),
        {"pressure": pressure, "fs": FS, "unit": "cmH2O"},
        starts,
        ends,
        baseline,
        include_aub=False,
    )

    time_products = result["pocc_time_products"]
    assert np.all(np.isfinite(time_products)), "no manoeuvre should be set to NaN"
    np.testing.assert_array_equal(
        time_products, window_integral(pressure, FS, starts, ends, baseline)
    )
