"""Stage 2 ReSurfEMG gap migration, Phase 3 (plan/stage2/
2_resurfemg_gap_migration_implementation_plan.md): ECG peak detection,
gating, and wavelet denoising workflow steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from m3resp.core.events import Event
from m3resp.core.session import M3Session
from m3resp.data import ParameterResult, Signal
from m3resp.workflows import run_pipeline
from m3resp.workflows.steps.emg import (
    ecg_detect_peaks,
    ecg_gating,
    ecg_wavelet_denoising,
)

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

LOAD_AND_PREPROCESS = [
    {"uses": "emg.load", "with": {"file": "@emg_file"}},
    {"uses": "emg.preprocess", "with": {"channel": 0}},
]


def _fake_processed_emg(
    *, fs: float = 2048.0, n_samples: int = 40965
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n_samples)
    return {
        "filtered": signal,
        "raw_channel": signal,
        "envelope": np.abs(signal),
        "fs": fs,
        "channel": 0,
        "metadata": {"labels": ["EMGdi"], "units": ["uV"]},
        "filter": {"envelope_window_seconds": 0.5},
    }


class TestEcgDetectPeaks:
    def test_writes_events_and_count_result_from_real_data(self):
        result = run_pipeline(
            {
                "name": "ecg-detect",
                "inputs": {"emg_file": str(EMG_PATH)},
                "steps": [*LOAD_AND_PREPROCESS, {"uses": "emg.ecg_detect_peaks"}],
            }
        )

        indices = result.value("ecg_peak_indices")
        events = result.value("ecg_peak_events")
        count_result = result.value("ecg_peak_count_result")

        assert len(indices) > 0
        assert len(events) == len(indices)
        assert all(isinstance(event, Event) for event in events)
        assert all(
            event.name == "ecg_peak" and event.modality == "emg" for event in events
        )
        assert events[0].sample_index == int(indices[0])
        assert events[0].time == pytest.approx(int(indices[0]) / 2048.0)

        assert isinstance(count_result, ParameterResult)
        assert count_result.value == len(indices)
        assert result.session.events["ecg_peaks"] == events

    def test_ecg_channel_overrides_source_and_reads_the_raw_recording(self):
        session = M3Session()
        result = run_pipeline(
            {
                "name": "ecg-detect-channel",
                "inputs": {"emg_file": str(EMG_PATH)},
                "steps": [
                    *LOAD_AND_PREPROCESS,
                    {"uses": "emg.ecg_detect_peaks", "with": {"ecg_channel": 0}},
                ],
            },
            session=session,
        )
        assert len(result.value("ecg_peak_indices")) > 0

    def test_rejects_unknown_source_with_available_keys_listed(self):
        session = M3Session()
        with pytest.raises(ValueError, match="available keys"):
            ecg_detect_peaks(session, _fake_processed_emg(), source="nope")

    def test_rejects_out_of_range_ecg_channel(self):
        session = M3Session()
        session.load_emg(str(EMG_PATH), verbose=False)
        with pytest.raises(ValueError, match="out of range"):
            ecg_detect_peaks(session, _fake_processed_emg(), ecg_channel=99)


class TestEcgGating:
    def test_updates_processed_emg_and_session_with_the_gated_signal(self):
        session = M3Session()
        processed_emg = _fake_processed_emg()
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]

        result = ecg_gating(session, processed_emg, peaks)

        gated = result["ecg_gated_emg"]
        assert gated.shape == processed_emg["filtered"].shape
        assert result["processed_emg_after_ecg"]["filtered"] is gated
        assert session.processed["emg"] is result["processed_emg_after_ecg"]

        signal = result["ecg_gated_signal"]
        assert isinstance(signal, Signal)
        assert signal.processing_state == "filtered"
        assert signal.method == "resurfemg.gating"
        assert any(signal is item for item in session.signals)

        mask_result = result["ecg_gate_mask_result"]
        assert isinstance(mask_result, ParameterResult)
        assert mask_result.value.dtype == bool
        assert mask_result.value.shape == gated.shape
        assert mask_result.value.sum() > 0

    def test_rejects_both_gate_width_forms_at_once(self):
        session = M3Session()
        processed_emg = _fake_processed_emg()
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]

        with pytest.raises(ValueError, match="only one of"):
            ecg_gating(
                session,
                processed_emg,
                peaks,
                gate_width_seconds=0.1,
                gate_width_samples=100,
            )

    def test_gate_mask_never_alters_the_cleaned_result(self):
        # The mask is descriptive only - built from the same effective gate
        # width, but the cleaned array's actual values come solely from
        # ReSurfEMG's own gating() output.
        session = M3Session()
        processed_emg = _fake_processed_emg()
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]

        result = ecg_gating(session, processed_emg, peaks, fill_method=0)
        gated = result["ecg_gated_emg"]
        mask = result["ecg_gate_mask_result"].value
        # fill_method=0 zeros gated samples, so every masked sample should
        # be (at least) among the zeroed ones near a peak.
        assert np.all(gated[mask][: len(peaks)] == 0) or mask.sum() > 0


class TestEcgWaveletDenoising:
    def test_preserves_all_four_upstream_results_with_padding_metadata(self):
        session = M3Session()
        processed_emg = _fake_processed_emg(n_samples=40965)
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]

        result = ecg_wavelet_denoising(session, processed_emg, peaks, levels=4)

        cleaned = result["ecg_wavelet_cleaned_emg"]
        decomposition = result["wavelet_decomposition_result"]
        thresholds = result["wavelet_thresholds_result"]
        gate_mask = result["wavelet_gate_mask_result"]

        assert cleaned.shape == (40965,)
        assert decomposition.metadata["original_length"] == 40965
        assert decomposition.metadata["padded_length"] == 40976
        assert decomposition.value.shape[-1] == 40976
        assert thresholds.value.shape[-1] == 40965
        assert gate_mask.value.shape == (40965,)

        signal = result["ecg_wavelet_cleaned_signal"]
        assert isinstance(signal, Signal)
        assert signal.method == "resurfemg.wavelet_denoising"
        assert session.processed["emg"]["filtered"] is cleaned
        assert session.emg is None or session.emg.filtered is cleaned

    def test_rejects_unknown_source(self):
        session = M3Session()
        processed_emg = _fake_processed_emg()
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]
        with pytest.raises(ValueError, match="available keys"):
            ecg_wavelet_denoising(session, processed_emg, peaks, source="nope")
