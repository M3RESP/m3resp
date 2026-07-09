"""Regression tests for shared quality mapping helpers."""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.processing.quality import (
    fraction_flag,
    quality_flag_from_result,
    skipped_quality_flag,
    threshold_flag,
    timing_window_flag,
)


def test_threshold_fraction_and_timing_flags_encode_expected_pass_fail():
    assert threshold_flag(
        "snr_minimum",
        12.5,
        threshold=10.0,
        comparison=">=",
        modality="emg",
    ).passed
    assert not threshold_flag(
        "too_large",
        12.5,
        threshold=10.0,
        comparison="<=",
        modality="emg",
    ).passed

    fraction = fraction_flag(
        "detected_fraction",
        0.7,
        minimum_fraction=0.5,
        modality="emg",
    )
    assert fraction.passed
    assert fraction.value == 0.7
    assert fraction.threshold == 0.5

    timing = timing_window_flag(
        "event_timing",
        np.array([0.1, 0.4, 0.8]),
        min_delta=0.0,
        max_delta=1.0,
        modality="emg",
    )
    assert timing.passed
    assert timing.metadata["valid"] == [True, True, True]

    late = timing_window_flag(
        "event_timing",
        np.array([0.1, 1.4]),
        min_delta=0.0,
        max_delta=1.0,
        modality="emg",
    )
    assert not late.passed
    assert late.metadata["valid"] == [True, False]


def test_quality_flag_from_result_preserves_current_adapter_mapping():
    scalar = quality_flag_from_result(
        "snr_pseudo",
        12.5,
        modality="emg",
        source_method="resurfemg.snr_pseudo",
    )
    assert scalar.passed
    assert scalar.value == 12.5
    assert scalar.metadata == {"source_method": "resurfemg.snr_pseudo"}

    bool_array = quality_flag_from_result(
        "detect_extreme_time_products",
        np.array([True, False]),
        modality="emg",
    )
    assert not bool_array.passed
    assert bool_array.value is None
    assert bool_array.metadata["shape"] == (2,)
    assert bool_array.metadata["dtype"] == "bool"

    numeric_array = quality_flag_from_result(
        "snr_pseudo",
        np.array([1.0, 2.0]),
        modality="emg",
    )
    assert numeric_array.passed
    assert numeric_array.value is None

    skipped = skipped_quality_flag(
        "quality_assessment.pocc_quality",
        "Needs Pocc ventilator inputs.",
        modality="emg",
    )
    assert not skipped.passed
    assert skipped.severity == "warning"
    assert skipped.message == "Needs Pocc ventilator inputs."
    assert skipped.metadata == {"skipped": True}


def test_threshold_flag_rejects_unknown_comparison():
    with pytest.raises(ValueError, match="comparison"):
        threshold_flag("bad", 1.0, threshold=1.0, comparison="approximately")
