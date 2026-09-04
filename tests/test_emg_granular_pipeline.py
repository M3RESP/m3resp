"""Regression coverage for the granular `emg.*` postprocessing steps.

Confirms the per-function `emg.*` pipeline steps (src/m3resp/workflows/steps/emg.py)
are a faithful factoring of `ReSurfEMGAdapter._postprocess_default`, not a
behavioral rewrite, by running both against the same committed sample data
and comparing numeric output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from m3resp.core.session import M3Session
from m3resp.workflows import run_pipeline

pytest.importorskip("resurfemg")
np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
EMG_PATH = (
    REPO_ROOT
    / "data"
    / "source"
    / "data_from_repo"
    / "emg_data_synth_quiet_breathing.Poly5"
)
VENT_PATH = (
    REPO_ROOT
    / "data"
    / "source"
    / "data_from_repo"
    / "vent_data_synth_quiet_breathing.Poly5"
)

GRANULAR_SPEC = {
    "name": "emg-granular-postprocess",
    "inputs": {"emg_file": str(EMG_PATH), "vent_file": str(VENT_PATH)},
    "steps": [
        {"uses": "emg.load", "with": {"file": "@emg_file"}},
        {"uses": "ventilator.load", "with": {"file": "@vent_file"}},
        {
            "uses": "emg.preprocess",
            "with": {"channel": 0, "high_pass_hz": 80, "envelope_window_seconds": 0.5},
        },
        {
            "uses": "emg.detect_breaths",
            "with": {"min_breath_width_seconds": 1.0},
        },
        {"uses": "emg.peak_indices"},
        {
            "uses": "ventilator.channels",
            "with": {"pressure_channel": 0, "flow_channel": 1, "volume_channel": 2},
        },
        {
            "uses": "emg.moving_baseline",
            "with": {"window_seconds": 30.0, "step_seconds": 1.0, "percentile": 33.0},
        },
        {"uses": "ventilator.find_occluded_breaths"},
        {"uses": "ventilator.detect_breaths", "with": {"breath_width_seconds": 0.5}},
        {"uses": "ventilator.detect_non_consecutive_manoeuvres"},
        {
            "uses": "ventilator.normalize_breaths",
            "with": {"breath_width_seconds": 0.5},
        },
    ],
}


def test_granular_emg_pipeline_matches_monolithic_postprocess():
    result = run_pipeline(GRANULAR_SPEC)

    reference_session = M3Session()
    reference_session.load_emg(str(EMG_PATH), verbose=False)
    reference_session.preprocess_emg(
        channel=0, high_pass_hz=80, envelope_window_seconds=0.5
    )
    reference_session.detect_emg_breaths(min_breath_width_seconds=1.0)
    ventilator = reference_session.emg_adapter.load(str(VENT_PATH), verbose=False)
    reference = reference_session.postprocess_emg(
        ventilator=ventilator,
        selected_functions={
            "baseline": {"moving_baseline": True},
            "event_detection": {
                "find_occluded_breaths": True,
                "detect_ventilator_breath": True,
            },
            "quality_assessment": {"detect_non_consecutive_manoeuvres": True},
        },
    )

    np.testing.assert_allclose(
        result.value("baseline"),
        reference["computed"]["baseline"]["moving_baseline"],
    )
    np.testing.assert_array_equal(
        result.value("ventilator_breath_indices"),
        reference["computed"]["event_detection"]["detect_ventilator_breath"],
    )
    np.testing.assert_array_equal(
        result.value("pocc_indices"),
        reference["computed"]["event_detection"]["find_occluded_breaths"],
    )
    np.testing.assert_array_equal(
        result.value("detect_non_consecutive_manoeuvres"),
        reference["computed"]["quality_assessment"][
            "detect_non_consecutive_manoeuvres"
        ],
    )
    assert len(result.session.events.get("ventilator_breaths", [])) == len(
        reference_session.events.get("ventilator_breaths", [])
    )
    assert len(result.value("ventilator_breath_indices")) > 0
