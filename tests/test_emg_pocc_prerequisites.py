"""Stage 2 ReSurfEMG gap migration, Phase 4 (plan/stage2/
2_resurfemg_gap_migration_implementation_plan.md): Pocc interval and
pressure-time-product prerequisite steps for `emg.pocc_quality`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from m3resp.core.events import BreathEvent
from m3resp.data import ParameterResult
from m3resp.processing.ventilator import estimate_peep
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

POCC_SPEC = {
    "name": "pocc-prerequisites",
    "inputs": {"emg_file": str(EMG_PATH), "vent_file": str(VENT_PATH)},
    "steps": [
        {"uses": "emg.load", "with": {"file": "@emg_file"}},
        {"uses": "ventilator.load", "with": {"file": "@vent_file"}},
        {"uses": "ventilator.channels"},
        {"uses": "ventilator.find_occluded_breaths"},
        {"uses": "ventilator.pocc_intervals"},
        {"uses": "ventilator.pocc_time_product"},
    ],
}


class TestPoccIntervals:
    def test_writes_start_end_validity_and_ventilator_modality_events(self):
        result = run_pipeline(POCC_SPEC)
        o = result.outputs

        pocc_indices = o["pocc_indices"]
        starts = o["pocc_start_indices"]
        ends = o["pocc_end_indices"]
        validity = o["pocc_interval_validity"]
        events = o["pocc_events"]

        assert len(pocc_indices) > 0
        assert (
            len(starts)
            == len(ends)
            == len(validity)
            == len(events)
            == len(pocc_indices)
        )
        assert np.all(starts <= pocc_indices)
        assert np.all(ends >= pocc_indices)

        for index, event in enumerate(events):
            assert isinstance(event, BreathEvent)
            assert event.modality == "ventilator"
            assert event.metadata["event_type"] == "pocc"
            assert event.start_index == int(starts[index])
            assert event.end_index == int(ends[index])
            assert event.peak_index == int(pocc_indices[index])
            assert event.sample_frequency == result.value("ventilator_signals")["fs"]
            assert event.metadata["valid"] == bool(validity[index])

        assert result.session.events["pocc_breaths"] == events

    def test_uses_the_same_peep_rule_as_find_occluded_breaths(self):
        result = run_pipeline(POCC_SPEC)
        signals = result.value("ventilator_signals")
        expected_peep = estimate_peep(signals["pressure"], signals["volume"])
        for event in result.value("pocc_events"):
            assert event.metadata["peep"] == pytest.approx(expected_peep)

    def test_explicit_peep_overrides_the_estimate(self):
        # Close to the estimated end-expiratory PEEP (~4.97) so the override
        # is exercised
        # without tripping the unrelated onoff_from_baseline_crossings edge
        # case where a far-off baseline never crosses again after the last
        # peak (a pre-existing primitive limitation, not this step's bug).
        spec = {**POCC_SPEC, "steps": [*POCC_SPEC["steps"]]}
        spec["steps"][-2] = {"uses": "ventilator.pocc_intervals", "with": {"peep": 5.2}}
        result = run_pipeline(spec)
        for event in result.value("pocc_events"):
            assert event.metadata["peep"] == 5.2

    def test_does_not_assume_emg_and_ventilator_fs_are_equal(self):
        result = run_pipeline(POCC_SPEC)
        emg_fs = result.session.emg.metadata["fs"]
        vent_fs = result.value("ventilator_signals")["fs"]
        assert emg_fs != vent_fs
        # Every Pocc event's sample_frequency is the ventilator's, not EMG's.
        assert all(
            event.sample_frequency == vent_fs for event in result.value("pocc_events")
        )


class TestPoccTimeProduct:
    def test_writes_one_value_per_pocc_with_combined_pressure_time_unit(self):
        result = run_pipeline(POCC_SPEC)
        o = result.outputs

        time_products = o["pocc_time_products"]
        result_obj = o["pocc_time_product_result"]

        assert isinstance(result_obj, ParameterResult)
        assert len(time_products) == len(o["pocc_indices"])
        assert result_obj.unit == "cmH2O*s"
        assert result_obj.modality == "ventilator"
        assert result_obj.category == "airway_pressure"
        np.testing.assert_array_equal(result_obj.value, time_products)
        assert result_obj.metadata["start_indices"] == [
            int(x) for x in o["pocc_start_indices"]
        ]
        assert result_obj.metadata["end_indices"] == [
            int(x) for x in o["pocc_end_indices"]
        ]
        assert np.all(time_products > 0)

    def test_matches_the_native_window_integral_equivalence_proof(self):
        # This is the same computation
        # test_processing_metric_equivalence.py's negative-deflection test
        # proves matches upstream time_product; re-derive it here manually
        # to pin the step's own wiring (peep baseline, indices) too.
        from m3resp.processing.metrics import window_integral

        result = run_pipeline(POCC_SPEC)
        signals = result.value("ventilator_signals")
        pressure = np.asarray(signals["pressure"])
        fs = float(signals["fs"])
        peep = estimate_peep(pressure, signals["volume"])
        baseline = np.full(pressure.shape, peep)

        expected = window_integral(
            pressure,
            fs,
            result.value("pocc_start_indices"),
            result.value("pocc_end_indices"),
            baseline,
        )
        np.testing.assert_array_equal(result.value("pocc_time_products"), expected)
