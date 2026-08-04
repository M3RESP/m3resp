"""Stage 2 ReSurfEMG gap migration, Phase 6.3 (plan/stage2/
2_resurfemg_gap_migration_implementation_plan.md): EMG array-valued
ParameterResults (gate masks, wavelet decomposition/thresholds, bell-curve
fit parameters) reuse the shared `parameter_result_arrays.npz` exporter
rather than a competing EMG-specific array format - this reloads every
array and compares values, NaN pattern, shape, and dtype.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from m3resp.core.session import M3Session
from m3resp.workflows.steps.emg import ecg_detect_peaks, ecg_wavelet_denoising

pytest.importorskip("resurfemg")
np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_parameter_results_csv(output_dir: Path) -> list[dict]:
    with (output_dir / "parameter_results.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_ecg_wavelet_array_results_round_trip_through_the_shared_archive(tmp_path):
    session = M3Session()
    fs = 2048.0
    n = 20 * 2048 + 5  # non-power-of-16 length, exercises real padding
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n)
    processed_emg = {
        "filtered": signal,
        "raw_channel": signal,
        "envelope": np.abs(signal),
        "fs": fs,
        "channel": 0,
        "metadata": {"labels": ["EMGdi"], "units": ["uV"]},
        "filter": {"envelope_window_seconds": 0.5},
    }
    peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]
    out = ecg_wavelet_denoising(session, processed_emg, peaks, levels=4)

    decomposition = out["wavelet_decomposition_result"]
    thresholds = out["wavelet_thresholds_result"]
    gate_mask = out["wavelet_gate_mask_result"]

    # All three must actually be in the session collection - not just
    # returned from the step - or export_summary() can't reach them.
    assert decomposition in list(session.parameter_results)
    assert thresholds in list(session.parameter_results)
    assert gate_mask in list(session.parameter_results)

    output_dir = session.export_summary(tmp_path)
    archive_path = output_dir / "parameter_result_arrays.npz"
    assert archive_path.exists()

    rows = _read_parameter_results_csv(output_dir)
    rows_by_name = {
        row["name"]: row for row in rows if row["name"].startswith("ecg_wavelet")
    }
    assert set(rows_by_name) == {
        "ecg_wavelet_decomposition",
        "ecg_wavelet_thresholds",
        "ecg_wavelet_gate_mask",
    }

    with np.load(archive_path) as archive:
        for result, row in (
            (decomposition, rows_by_name["ecg_wavelet_decomposition"]),
            (thresholds, rows_by_name["ecg_wavelet_thresholds"]),
            (gate_mask, rows_by_name["ecg_wavelet_gate_mask"]),
        ):
            array_key = row["array_key"]
            reloaded = archive[array_key]
            original = np.asarray(result.value)

            assert reloaded.shape == original.shape
            assert reloaded.dtype == original.dtype
            assert row["shape"] == str(list(original.shape))
            assert row["dtype"] == str(original.dtype)
            if np.issubdtype(original.dtype, np.floating):
                np.testing.assert_array_equal(np.isnan(reloaded), np.isnan(original))
            np.testing.assert_array_equal(reloaded, original)  # NaN-aware


def test_ecg_gate_mask_round_trips_and_is_a_bool_array(tmp_path):
    from m3resp.workflows.steps.emg import ecg_gating

    session = M3Session()
    fs = 2048.0
    n = 8192
    rng = np.random.default_rng(1)
    signal = rng.normal(size=n)
    processed_emg = {
        "filtered": signal,
        "raw_channel": signal,
        "envelope": np.abs(signal),
        "fs": fs,
        "channel": 0,
        "metadata": {"labels": ["EMGdi"], "units": ["uV"]},
        "filter": {"envelope_window_seconds": 0.5},
    }
    peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]
    out = ecg_gating(session, processed_emg, peaks)
    gate_mask_result = out["ecg_gate_mask_result"]

    assert gate_mask_result in list(session.parameter_results)

    output_dir = session.export_summary(tmp_path)
    with np.load(output_dir / "parameter_result_arrays.npz") as archive:
        rows = _read_parameter_results_csv(output_dir)
        row = next(r for r in rows if r["name"] == "ecg_gate_mask")
        reloaded = archive[row["array_key"]]
        assert reloaded.dtype == bool
        np.testing.assert_array_equal(reloaded, gate_mask_result.value)
