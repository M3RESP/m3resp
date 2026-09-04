"""Stage 2 ReSurfEMG gap migration, Phase 6.2/8 (plan/stage2/
2_resurfemg_gap_migration_implementation_plan.md): step-level provenance for
migrated EMG steps, recorded through the existing `M3Session._record()` seam.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from m3resp.workflows import run_pipeline

pytest.importorskip("resurfemg")

REPO_ROOT = Path(__file__).resolve().parents[1]
EMG_PATH = (
    REPO_ROOT
    / "data"
    / "source"
    / "data_from_repo"
    / "emg_data_synth_quiet_breathing.Poly5"
)


def _provenance_for(session, action: str):
    matches = [record for record in session.provenance if record.action == action]
    assert matches, f"no provenance recorded for {action!r}"
    return matches[-1]


def test_upstream_backed_step_records_the_shared_provenance_schema():
    result = run_pipeline(
        {
            "name": "provenance-smoke",
            "inputs": {"emg_file": str(EMG_PATH)},
            "steps": [
                {"uses": "emg.load", "with": {"file": "@emg_file"}},
                {
                    "uses": "emg.preprocess",
                    "with": {"channel": 0},
                    "out": {"processed_emg": "processed_emg_before_ecg"},
                },
                {
                    "uses": "emg.ecg_detect_peaks",
                    "in": {"processed_emg": "processed_emg_before_ecg"},
                },
            ],
        }
    )

    record = _provenance_for(result.session, "emg.ecg_detect_peaks")
    assert record.modality == "emg"
    params = record.parameters
    assert params["source_package"] == "resurfemg"
    assert params["source_function"] == (
        "resurfemg.preprocessing.ecg_removal.detect_ecg_peaks"
    )
    assert params["implementation"] == "upstream_adapter"
    assert params["operation"] == "emg.ecg_detect_peaks"
    assert params["step"] == "emg.ecg_detect_peaks"
    assert set(params["reads"]) == {"session", "processed_emg"}
    assert set(params["writes"]) == {
        "ecg_peak_indices",
        "ecg_peak_events",
        "ecg_peak_count_result",
    }
    # Reference version is looked up dynamically (installed resurfemg
    # version), not hardcoded to whatever version this plan was written
    # against.
    assert "upstream_version" in params


def test_native_primitive_step_records_m3resp_as_the_source_package():
    # A flat pressure signal never crosses its own median baseline, which
    # trips an unrelated pre-existing edge case in
    # onoff_from_baseline_crossings (no crossing found after the last peak);
    # dip below PEEP around each peak so real crossings exist.
    peep = 5.0
    pressure = [peep] * 200
    for center in (50, 150):
        for offset in range(-20, 20):
            pressure[center + offset] = peep - 2.0

    result = run_pipeline(
        {
            "name": "provenance-native-smoke",
            "inputs": {"vent_file": str(EMG_PATH)},  # unused, just needs a value
            "steps": [
                {
                    "uses": "ventilator.pocc_intervals",
                    # This synthetic trace carries no volume channel, so PEEP
                    # cannot be estimated from end-expiration; state it.
                    "with": {"peep": peep},
                    "in": {
                        "ventilator_signals": "_ventilator_signals_input",
                        "pocc_indices": "_pocc_indices_input",
                    },
                },
            ],
        },
        extra_context={
            "_ventilator_signals_input": {
                "pressure": pressure,
                "fs": 100.0,
            },
            "_pocc_indices_input": [50, 150],
        },
    )

    record = _provenance_for(result.session, "ventilator.pocc_intervals")
    params = record.parameters
    assert params["source_package"] == "m3resp"
    assert params["implementation"] == "m3resp.processing.intervals"
    assert params["source_function"] == (
        "m3resp.processing.intervals.onoff_from_baseline_crossings"
    )


def test_each_migrated_step_call_adds_exactly_one_provenance_record():
    result = run_pipeline(
        {
            "name": "provenance-count-smoke",
            "inputs": {"emg_file": str(EMG_PATH)},
            "steps": [
                {"uses": "emg.load", "with": {"file": "@emg_file"}},
                {
                    "uses": "emg.preprocess",
                    "with": {"channel": 0},
                    "out": {"processed_emg": "processed_emg_before_ecg"},
                },
                {
                    "uses": "emg.ecg_detect_peaks",
                    "in": {"processed_emg": "processed_emg_before_ecg"},
                },
            ],
        }
    )

    matches = [
        r for r in result.session.provenance if r.action == "emg.ecg_detect_peaks"
    ]
    assert len(matches) == 1
