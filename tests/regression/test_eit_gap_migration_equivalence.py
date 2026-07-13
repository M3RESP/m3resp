"""Phase 7 (plan/stage2/1_eit_gap_migration_implementation_plan.md) - upstream
equivalence tests for the Stage 2 EIT gap migration's native conversions.

Every `eit.*` step introduced or migrated by the gap migration is a thin
wrapper around one `eitprocessing` call, plus a conversion step (object array
with `None` -> float array with NaN, `PixelMask` -> array, etc.) that turns
the upstream result into an `m3resp` native `ParameterResult`. Phase 1-4's own
tests already cover shapes and error handling; this file instead locks the
*conversion* layer down against an independent, direct call into the same
upstream classes on the committed synthetic Draeger fixture, so a future
refactor of the conversion helpers (or the Stage 3 native reimplementation)
has a frozen numeric baseline to match.

Tolerance: exact equality throughout (`np.testing.assert_array_equal`, which
treats NaN as equal to NaN). Every comparison here is wrapper-vs-direct-call
on the same deterministic upstream algorithm with identical parameters, so
there is no floating-point algorithm divergence to tolerate - a mismatch
means the *conversion*, not the science, has drifted.

Skips cleanly (`pytest.importorskip`) when `eitprocessing` is not installed;
Phase 1-6's adapter/workflow contract tests do not depend on it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from m3resp.core.session import M3Session
from m3resp.workflows.engine import run_pipeline

pytest.importorskip("eitprocessing")


def _fixture_path() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    path = os.path.join(
        repo_root,
        "data",
        "source",
        "data_from_repo",
        "draeger_synthetic_draeger_20Hz.bin",
    )
    assert os.path.exists(path), f"missing committed EIT fixture: {path}"
    return path


_SPEC: dict[str, Any] = {
    "name": "eit-gap-migration-equivalence",
    "steps": [
        {"uses": "eit.load", "with": {"file": _fixture_path(), "vendor": "draeger"}},
        {
            "uses": "eit.detect_rates",
            "in": {"signal": "raw_eit"},
            "with": {"subject_type": "adult", "capture": True},
        },
        {
            "uses": "eit.mdn_filter",
            "in": {"signal": "raw_eit"},
            "with": {"label": "mdn_filtered"},
        },
        {"uses": "eit.global_impedance", "in": {"signal": "filtered_eit"}},
        {
            "uses": "eit.detect_breaths",
            "in": {"signal": "global_impedance"},
            "with": {"min_duration_s": 2 / 3},
        },
        {"uses": "eit.eeli"},
        {"uses": "eit.pixel_tiv"},
        {"uses": "eit.pixel_breaths"},
        {"uses": "eit.roi_tiv_lungspace", "with": {"threshold": 0.15}},
        {"uses": "eit.roi_amplitude_lungspace", "with": {"threshold": 0.15}},
        {"uses": "eit.roi_watershed", "with": {"threshold_fraction": 0.15}},
        {
            "uses": "eit.roi_filter_by_size",
            "in": {"mask": "watershed_lungspace_mask"},
            "with": {"min_region_size": 1},
        },
    ],
}


@pytest.fixture(scope="module")
def pipeline_result():
    return run_pipeline(_SPEC, session=M3Session())


def test_detect_rates_matches_direct_rate_detection_call(pipeline_result):
    from eitprocessing.features.rate_detection import RateDetection

    raw_eit = pipeline_result.value("raw_eit")
    expected_resp, expected_heart = RateDetection("adult").apply(raw_eit)

    assert pipeline_result.value("respiratory_rate_hz") == pytest.approx(expected_resp)
    assert pipeline_result.value("heart_rate_hz") == pytest.approx(expected_heart)

    result = pipeline_result.value("respiratory_rate_result")
    assert result.value == pytest.approx(expected_resp)
    assert result.unit == "Hz"


def test_mdn_filter_matches_direct_mdn_filter_call(pipeline_result):
    from eitprocessing.filters.mdn import MDNFilter

    raw_eit = pipeline_result.value("raw_eit")
    respiratory_rate_hz = pipeline_result.value("respiratory_rate_hz")
    heart_rate_hz = pipeline_result.value("heart_rate_hz")

    expected = MDNFilter(
        respiratory_rate=respiratory_rate_hz, heart_rate=heart_rate_hz
    ).apply(raw_eit, label="mdn_filtered")

    filtered_eit = pipeline_result.value("filtered_eit")
    np.testing.assert_array_equal(
        filtered_eit.pixel_impedance, expected.pixel_impedance
    )

    signal = pipeline_result.value("filtered_eit_signal")
    np.testing.assert_array_equal(signal.values, expected.pixel_impedance)


def test_eeli_matches_direct_eeli_call(pipeline_result):
    from eitprocessing.parameters.eeli import EELI

    global_impedance = pipeline_result.value("global_impedance")
    breath_detector = pipeline_result.value("breath_detector")

    expected = EELI(breath_detection=breath_detector).compute_parameter(
        global_impedance, store=False, result_label="continuous_eelis"
    )

    eeli_result = pipeline_result.value("eeli_result")
    np.testing.assert_array_equal(
        eeli_result.value, np.asarray(expected.values, dtype=float)
    )
    np.testing.assert_array_equal(
        eeli_result.metadata["time"], np.asarray(expected.time, dtype=float)
    )
    assert eeli_result.unit == expected.unit


def test_pixel_tiv_matches_direct_tiv_call(pipeline_result):
    from eitprocessing.parameters.tidal_impedance_variation import TIV

    filtered_eit = pipeline_result.value("filtered_eit")
    global_impedance = pipeline_result.value("global_impedance")
    breath_detector = pipeline_result.value("breath_detector")

    expected = TIV(breath_detection=breath_detector).compute_parameter(
        filtered_eit,
        global_impedance,
        None,
        tiv_timing="continuous",
        store=False,
        result_label="pixel_tivs",
    )
    # `expected.values` is already a plain float array here (TIV's pixel path
    # only produces object-dtype `None` placeholders for missing *timing*,
    # not for the TIV values themselves), so this is a direct cast rather
    # than a None-aware conversion.
    expected_values = np.asarray(expected.values, dtype=float)

    pixel_tiv_result = pipeline_result.value("pixel_tiv_result")
    np.testing.assert_array_equal(pixel_tiv_result.value, expected_values)


def test_pixel_breaths_matches_direct_pixel_breath_call(pipeline_result):
    from eitprocessing.features.breath_detection import BreathDetection
    from eitprocessing.features.pixel_breath import PixelBreath

    filtered_eit = pipeline_result.value("filtered_eit")
    global_impedance = pipeline_result.value("global_impedance")

    expected = PixelBreath(
        breath_detection=BreathDetection(minimum_duration=2 / 3),
        phase_correction_mode="negative amplitude",
    ).find_pixel_breaths(filtered_eit, global_impedance, store=False)

    n_breaths, n_rows, n_cols = np.asarray(expected.values, dtype=object).shape
    expected_landmarks = np.full((n_breaths, n_rows, n_cols, 3), np.nan, dtype=float)
    values = np.asarray(expected.values, dtype=object)
    for breath_index in range(n_breaths):
        for row in range(n_rows):
            for col in range(n_cols):
                breath = values[breath_index, row, col]
                if breath is not None:
                    expected_landmarks[breath_index, row, col] = (
                        breath.start_time,
                        breath.middle_time,
                        breath.end_time,
                    )

    pixel_breath_timing_result = pipeline_result.value("pixel_breath_timing_result")
    np.testing.assert_array_equal(pixel_breath_timing_result.value, expected_landmarks)


def test_roi_tiv_lungspace_matches_direct_call(pipeline_result):
    from eitprocessing.roi.tiv import TIVLungspace

    filtered_eit = pipeline_result.value("filtered_eit")
    global_impedance = pipeline_result.value("global_impedance")

    expected_mask = TIVLungspace(threshold=0.15).apply(
        filtered_eit, timing_data=global_impedance
    )

    result = pipeline_result.value("tiv_lungspace_result")
    np.testing.assert_array_equal(result.value, expected_mask.mask)


def test_roi_amplitude_lungspace_matches_direct_call(pipeline_result):
    from eitprocessing.roi.amplitude import AmplitudeLungspace

    filtered_eit = pipeline_result.value("filtered_eit")
    global_impedance = pipeline_result.value("global_impedance")

    expected_mask = AmplitudeLungspace(threshold=0.15).apply(
        filtered_eit, timing_data=global_impedance
    )

    result = pipeline_result.value("amplitude_lungspace_result")
    np.testing.assert_array_equal(result.value, expected_mask.mask)


def test_roi_watershed_matches_direct_call(pipeline_result):
    from eitprocessing.roi.watershed import WatershedLungspace

    filtered_eit = pipeline_result.value("filtered_eit")
    global_impedance = pipeline_result.value("global_impedance")

    expected_mask = WatershedLungspace(threshold_fraction=0.15).apply(
        filtered_eit, timing_data=global_impedance
    )

    result = pipeline_result.value("watershed_lungspace_result")
    np.testing.assert_array_equal(result.value, expected_mask.mask)


def test_roi_filter_by_size_matches_direct_call(pipeline_result):
    from eitprocessing.roi.filter_by_size import FilterROIBySize

    watershed_mask = pipeline_result.value("watershed_lungspace_mask")
    expected_mask = FilterROIBySize(min_region_size=1).apply(watershed_mask)

    result = pipeline_result.value("size_filtered_roi_result")
    np.testing.assert_array_equal(result.value, expected_mask.mask)
