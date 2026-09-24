"""Stage 2 ReSurfEMG gap migration, Phase 4 (plan/stage2/
2_resurfemg_gap_migration_implementation_plan.md): Pocc interval and
pressure-time-product prerequisite steps for `emg.pocc_quality`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from m3resp.core.events import BreathEvent
from m3resp.data import ParameterResult
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
        {"uses": "emg.load", "with": {"file_path": "@emg_file"}},
        {"uses": "ventilator.load", "with": {"file_path": "@vent_file"}},
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

    def test_crossings_use_a_moving_pressure_baseline(self):
        # ReSurfEMG takes Pocc on/offset against the moving baseline of the
        # airway pressure (percentile 33, 7.5 s window), not the constant PEEP.
        from resurfemg.postprocessing.baseline import moving_baseline

        result = run_pipeline(POCC_SPEC)
        signals = result.value("ventilator_signals")
        pressure = np.asarray(signals["pressure"], dtype=float)
        fs = float(signals["fs"])
        expected = moving_baseline(
            pressure, int(7.5 * fs), int(0.2 * fs), set_percentile=33
        )
        baseline = result.value("pressure_baseline")
        np.testing.assert_array_equal(baseline, expected)
        assert np.ptp(baseline) > 0

    def test_baseline_parameters_are_passed_through(self):
        spec = {**POCC_SPEC, "steps": [*POCC_SPEC["steps"]]}
        spec["steps"][-2] = {
            "uses": "ventilator.pocc_intervals",
            "with": {"baseline_percentile": 20.0},
        }
        low = run_pipeline(spec).value("pressure_baseline")
        default = run_pipeline(POCC_SPEC).value("pressure_baseline")
        assert np.all(low <= default)
        assert np.any(low < default)

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

    def test_matches_resurfemg_ptpocc(self):
        # ReSurfEMG's PTPocc (basic_emg_pipeline_annotated.ipynb):
        # calculate_time_products(include_aub=True, aub_window_s=5 * fs,
        # aub_reference_signal=p_vent.y_baseline) = time product against the
        # moving baseline plus the area under that baseline.
        from resurfemg.postprocessing import features as feat

        result = run_pipeline(POCC_SPEC)
        signals = result.value("ventilator_signals")
        pressure = np.asarray(signals["pressure"], dtype=float)
        fs = float(signals["fs"])
        baseline = result.value("pressure_baseline")
        starts = result.value("pocc_start_indices")
        ends = result.value("pocc_end_indices")
        peaks = result.value("pocc_indices")

        time_products = feat.time_product(
            signal=pressure,
            fs=fs,
            start_idxs=starts,
            end_idxs=ends,
            baseline=baseline,
        )
        aub, _ = feat.area_under_baseline(
            signal=pressure,
            fs=fs,
            peak_idxs=peaks,
            start_idxs=starts,
            end_idxs=ends,
            aub_window_s=int(5 * fs),
            baseline=baseline,
            ref_signal=baseline,
        )
        np.testing.assert_allclose(
            result.value("pocc_time_products"), np.asarray(time_products) + aub
        )

    def test_area_under_baseline_can_be_left_out(self):
        from m3resp.processing.metrics import window_integral

        spec = {**POCC_SPEC, "steps": [*POCC_SPEC["steps"]]}
        spec["steps"][-1] = {
            "uses": "ventilator.pocc_time_product",
            "with": {"include_aub": False},
        }
        result = run_pipeline(spec)
        signals = result.value("ventilator_signals")
        expected = window_integral(
            np.asarray(signals["pressure"]),
            float(signals["fs"]),
            result.value("pocc_start_indices"),
            result.value("pocc_end_indices"),
            result.value("pressure_baseline"),
        )
        np.testing.assert_array_equal(result.value("pocc_time_products"), expected)
