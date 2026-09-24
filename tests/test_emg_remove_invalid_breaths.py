"""'emg.remove_invalid_breaths' keeps only the breaths that passed the
chosen per-breath quality flags, and shortens peaks, onsets, offsets and
features together so breath i stays row i. The original per-breath outputs
are left unchanged.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
import yaml

from m3resp.core.session import M3Session
from m3resp.data import QualityFlag
from m3resp.workflows import run_pipeline
from m3resp.workflows.steps.emg import (
    area_under_baseline,
    onoffpeak_baseline_crossing,
    remove_invalid_breaths,
    time_product,
    time_to_peak,
)

FS = 1000.0
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _session_with_three_breaths() -> tuple[M3Session, dict[str, Any]]:
    """Constant baseline of 1.0. Samples 500-1500 rise above baseline once
    but hold two peaks (800 and 1200), so the first breath's window runs
    past the second peak and is marked invalid. Samples 2000-2500 hold one
    separate, valid breath (peak 2250)."""

    envelope = np.full(3000, 0.5)
    envelope[500:1500] = 1.5
    envelope[800] = 3.0
    envelope[1200] = 2.5
    envelope[2000:2500] = 1.5
    envelope[2250] = 2.0
    baseline = np.ones(3000)
    peaks = np.array([800, 1200, 2250])
    processed_emg = {"envelope": envelope, "fs": FS}

    session = M3Session()
    windows = onoffpeak_baseline_crossing(session, processed_emg, baseline, peaks)
    starts, ends = windows["start_indices"], windows["end_indices"]
    outputs = {
        "peak_indices": peaks,
        "start_indices": starts,
        "end_indices": ends,
        "time_to_peak": time_to_peak(processed_emg, starts, ends)["time_to_peak"],
        "time_product": time_product(processed_emg, starts, ends, baseline)[
            "time_product"
        ],
        "area_under_baseline": area_under_baseline(
            processed_emg, peaks, starts, ends, baseline
        )["area_under_baseline"],
    }
    return session, outputs


def _emg_flag(name: str, peak: int, passed: bool) -> QualityFlag:
    return QualityFlag(
        name=name,
        passed=passed,
        severity="info",
        modality="emg",
        metadata={"peak_sample_index": peak},
    )


def test_breath_with_invalid_window_is_removed_and_all_outputs_stay_aligned():
    session, outputs = _session_with_three_breaths()
    original_products = np.array(outputs["time_product"], copy=True)

    kept = remove_invalid_breaths(session, **outputs)["valid_breaths"]

    assert kept["breath_numbers"].tolist() == [1, 2]
    np.testing.assert_array_equal(kept["peak_indices"], [1200, 2250])
    np.testing.assert_array_equal(kept["start_indices"], outputs["start_indices"][1:])
    np.testing.assert_array_equal(kept["end_indices"], outputs["end_indices"][1:])
    np.testing.assert_array_equal(
        kept["features"]["time_product"], outputs["time_product"][1:]
    )
    # Features stored as a pair of arrays are shortened part by part.
    for kept_part, original_part in zip(
        kept["features"]["time_to_peak"], outputs["time_to_peak"], strict=True
    ):
        np.testing.assert_array_equal(kept_part, np.asarray(original_part)[1:])
    assert len(kept["features"]["area_under_baseline"][0]) == 2
    assert "pseudo_slope" not in kept["features"], "features not given are left out"

    assert kept["removed"] == [
        {
            "breath_number": 0,
            "peak_sample_index": 800,
            "failed_flags": ["start_end_validity"],
        }
    ]
    # The original per-breath output is not changed.
    np.testing.assert_array_equal(outputs["time_product"], original_products)


def test_flags_are_matched_to_breaths_by_peak_sample():
    session, outputs = _session_with_three_breaths()
    # A check that only covered the last breath (like event timing, which
    # only scores breaths paired with the ventilator).
    session.quality.add(_emg_flag("evaluate_event_timing", 2250, passed=False))

    kept = remove_invalid_breaths(
        session,
        outputs["peak_indices"],
        flag_names=["start_end_validity", "evaluate_event_timing"],
    )["valid_breaths"]

    assert kept["breath_numbers"].tolist() == [1]
    assert [item["failed_flags"] for item in kept["removed"]] == [
        ["start_end_validity"],
        ["evaluate_event_timing"],
    ]


def test_most_recent_result_of_a_check_counts():
    session, outputs = _session_with_three_breaths()
    session.quality.add(_emg_flag("start_end_validity", 800, passed=True))

    kept = remove_invalid_breaths(session, outputs["peak_indices"])["valid_breaths"]

    assert kept["breath_numbers"].tolist() == [0, 1, 2]
    assert kept["removed"] == []


def test_unknown_flag_name_raises():
    session, outputs = _session_with_three_breaths()
    with pytest.raises(ValueError, match="No EMG quality flag named 'snr_pseudo'"):
        remove_invalid_breaths(
            session, outputs["peak_indices"], flag_names=["snr_pseudo"]
        )


def test_whole_recording_flag_cannot_be_used():
    session, outputs = _session_with_three_breaths()
    session.quality.add(
        QualityFlag(
            name="interpeak_dist", passed=False, severity="info", modality="emg"
        )
    )
    with pytest.raises(ValueError, match="not a per-breath flag"):
        remove_invalid_breaths(
            session, outputs["peak_indices"], flag_names=["interpeak_dist"]
        )


def test_feature_with_wrong_number_of_values_raises():
    session, outputs = _session_with_three_breaths()
    with pytest.raises(ValueError, match="'time_product' has 2 values"):
        remove_invalid_breaths(
            session,
            outputs["peak_indices"],
            time_product=outputs["time_product"][:2],
        )


def _example_emg_spec() -> dict[str, Any]:
    """The example EMG pipeline, with input paths made absolute and its
    file export left out."""

    example_dir = os.path.join(REPO_ROOT, "examples", "emg_full_preprocessing")
    with open(os.path.join(example_dir, "emg-full.pipeline.yaml")) as handle:
        spec = yaml.safe_load(handle)
    spec.pop("outputs", None)
    spec["inputs"] = {
        key: os.path.normpath(os.path.join(example_dir, value))
        for key, value in spec["inputs"].items()
    }
    return spec


def test_runs_at_the_end_of_the_example_emg_pipeline():
    pytest.importorskip("resurfemg")
    spec = _example_emg_spec()
    spec["steps"].append(
        {
            "id": "remove_invalid_breaths",
            "uses": "emg.remove_invalid_breaths",
            "with": {"flag_names": ["start_end_validity", "evaluate_bell_curve_error"]},
        }
    )

    result = run_pipeline(spec, session=M3Session())

    kept = result.value("valid_breaths")
    peaks = result.value("peak_indices")
    n_kept = len(kept["breath_numbers"])
    assert n_kept + len(kept["removed"]) == len(peaks)
    np.testing.assert_array_equal(kept["peak_indices"], peaks[kept["breath_numbers"]])
    for key in ("pseudo_slope", "amplitude", "time_product"):
        assert len(kept["features"][key]) == n_kept, key
    # Upstream outputs keep every breath.
    assert len(result.value("time_product")) == len(peaks)


def test_changes_nothing_that_other_steps_read():
    """Removing breaths must not change the full per-breath outputs or the
    session's breath events, quality flags, parameter results or signals.
    Other steps (e.g. linking EMG with EIT and ventilator breaths, which
    uses session.events) read these, and EMG flags and parameter results
    number breaths by their position in 'peak_indices'. As long as this
    holds, it does not matter where the step sits in a pipeline."""

    pytest.importorskip("resurfemg")
    result = run_pipeline(_example_emg_spec(), session=M3Session())
    session = result.session
    inputs = {
        key: result.value(key)
        for key in (
            "peak_indices",
            "start_indices",
            "end_indices",
            "time_to_peak",
            "pseudo_slope",
            "amplitude",
            "time_product",
            "area_under_baseline",
        )
    }
    # Fail the first three breaths so the step really removes some; on
    # this recording every breath passes the built-in checks.
    for peak in inputs["peak_indices"][:3]:
        session.quality.add(_emg_flag("manual_check", int(peak), passed=False))

    input_copies = {key: _deep_copy(value) for key, value in inputs.items()}
    events_before = {name: list(events) for name, events in session.events.items()}
    quality_before = list(session.quality.items)
    parameters_before = list(session.parameter_results.items)
    signals_before = list(session.signals.items)

    kept = remove_invalid_breaths(
        session, **inputs, flag_names=["start_end_validity", "manual_check"]
    )["valid_breaths"]
    assert len(kept["removed"]) == 3

    for key, value in inputs.items():
        _assert_same(value, input_copies[key], key)
    assert session.events.keys() == events_before.keys()
    for name, events in events_before.items():
        assert session.events[name] == events, name
    assert session.quality.items == quality_before
    assert session.parameter_results.items == parameters_before
    assert session.signals.items == signals_before


def _deep_copy(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_deep_copy(part) for part in value)
    return np.array(value, copy=True)


def _assert_same(value: Any, expected: Any, name: str) -> None:
    if isinstance(expected, tuple):
        assert isinstance(value, tuple) and len(value) == len(expected), name
        for part, expected_part in zip(value, expected, strict=True):
            _assert_same(part, expected_part, name)
        return
    np.testing.assert_array_equal(np.asarray(value), expected, err_msg=name)
