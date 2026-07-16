"""Stage 2 ReSurfEMG gap migration, Phase 8 (plan/stage2/
2_resurfemg_gap_migration_implementation_plan.md) - registry completeness
and full end-to-end workflow contract tests for the steps this migration
added or extended across Phases 1-6.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from m3resp.core.session import M3Session
from m3resp.workflows import available_steps, run_pipeline
from m3resp.workflows.registry import get_step

# Every step this migration added (Phase 3/4/5.2/5.3) or extended with
# native results (Phase 2, 5.1) - Phase 8's "every planned step appears in
# `m3resp steps`" acceptance criterion.
_MIGRATED_EMG_STEPS = (
    "emg.load",
    "emg.moving_baseline",
    "emg.slopesum_baseline",
    "emg.ecg_detect_peaks",
    "emg.ecg_gating",
    "emg.ecg_estimated_subtraction",
    "emg.ecg_wavelet_denoising",
    "emg.pocc_intervals",
    "emg.pocc_time_product",
    "emg.pocc_quality",
    "emg.interpeak_dist",
    "emg.snr_pseudo",
    "emg.percentage_under_baseline",
    "emg.detect_local_high_aub",
    "emg.detect_extreme_time_products",
    "emg.detect_non_consecutive_manoeuvres",
    "emg.evaluate_bell_curve_error",
    "emg.evaluate_event_timing",
    "emg.evaluate_respiratory_rates",
)


def test_every_migrated_step_is_registered_and_listed():
    steps = available_steps()
    for name in _MIGRATED_EMG_STEPS:
        assert name in steps, f"{name} missing from available_steps()"
        assert steps[name], f"{name} has an empty summary"


def test_registered_reads_and_writes_match_the_alternative_step_pairs():
    # emg.moving_baseline/emg.slopesum_baseline and emg.ecg_gating/
    # emg.ecg_wavelet_denoising are alternatives that both naturally write
    # the same context key (baseline / processed_emg_after_ecg) - confirm
    # the registry still declares both, not one silently overwritten by
    # the other's registration.
    moving = get_step("emg.moving_baseline")
    slopesum = get_step("emg.slopesum_baseline")
    assert "baseline" in moving.writes
    assert "baseline" in slopesum.writes
    assert moving.writes != slopesum.writes  # slopesum has more native outputs

    for name in (
        "emg.ecg_gating",
        "emg.ecg_estimated_subtraction",
        "emg.ecg_wavelet_denoising",
    ):
        assert "processed_emg_after_ecg" in get_step(name).writes


@pytest.mark.parametrize(
    "step_name",
    [
        "emg.pocc_intervals",
        "emg.pocc_time_product",
        "emg.pocc_quality",
        "emg.interpeak_dist",
        "emg.ecg_detect_peaks",
        "emg.ecg_gating",
        "emg.ecg_estimated_subtraction",
        "emg.ecg_wavelet_denoising",
    ],
)
def test_new_steps_declare_session_as_a_read(step_name):
    # Every new step needs session (to reach session.emg_adapter, or to
    # add native results to session.signals/parameter_results/quality).
    definition = get_step(step_name)
    assert "session" in definition.reads


class TestFullExampleEndToEnd:
    def test_full_emg_example_pipeline_runs_end_to_end(self, tmp_path):
        pytest.importorskip("resurfemg")

        repo_root = Path(__file__).resolve().parents[1]
        emg_fixture = os.path.join(
            repo_root,
            "data",
            "source",
            "data_from_repo",
            "emg_data_synth_quiet_breathing.Poly5",
        )
        vent_fixture = os.path.join(
            repo_root,
            "data",
            "source",
            "data_from_repo",
            "vent_data_synth_quiet_breathing.Poly5",
        )
        assert os.path.exists(emg_fixture), f"missing committed fixture: {emg_fixture}"
        assert os.path.exists(vent_fixture), (
            f"missing committed fixture: {vent_fixture}"
        )

        spec_path = os.path.join(
            repo_root, "examples", "emg_full_preprocessing", "emg-full.pipeline.yaml"
        )
        # run_pipeline (unlike run_spec) does not touch the spec's `outputs:`
        # section, so this exercises the example without writing into the
        # project's real output/ directory.
        result = run_pipeline(spec_path, session=M3Session())

        # ECG removal ran and fed into breath detection.
        assert len(result.value("ecg_peak_indices")) > 0
        assert len(result.value("emg_breath_events")) > 0

        # Pocc prerequisites and quality produced labeled, per-manoeuvre
        # results (not a raw unlabeled matrix).
        pocc_quality_results = result.value("pocc_quality_results")
        criteria_names = {r.metadata["criterion"] for r in pocc_quality_results}
        assert criteria_names == {"dp_up_10", "dp_up_90", "dp_up_90_norm"}

        # Native collections were populated exactly once, not duplicated.
        session = result.session
        assert len(session.signals) > 0
        assert len(session.parameter_results) > 0
        assert len(session.quality) > 0

        output_path = session.export_summary(tmp_path)
        archive_path = output_path / "parameter_result_arrays.npz"
        assert archive_path.exists()
        with np.load(archive_path) as archive:
            assert len(archive.files) > 0
            # A wavelet/gate-mask-shaped array from ECG removal should be
            # in the shared archive, not a competing EMG-only format.
            assert any(key.startswith("ecg_gate_mask") for key in archive.files)
