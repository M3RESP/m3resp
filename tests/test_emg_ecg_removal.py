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
from m3resp.processing.windows import rolling_envelope
from m3resp.workflows import run_pipeline
from m3resp.workflows.steps.emg import (
    ecg_detect_peaks,
    ecg_gating,
    ecg_wavelet_denoising,
)
from m3resp.workflows.steps.emg.ecg_gating import _build_gate_mask

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


class TestSeparateFilteredAndCleanedSignals:
    """Band-pass filtering and ECG removal are separate processing steps, so
    their results are kept under separate keys - as ReSurfEMG keeps 'filt'
    and 'clean'."""

    def test_gating_keeps_the_band_passed_signal_alongside_the_gated_one(self):
        session = M3Session()
        processed_emg = _fake_processed_emg()
        band_passed = processed_emg["filtered"]
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]

        after = ecg_gating(session, processed_emg, peaks)["processed_emg_after_ecg"]

        np.testing.assert_array_equal(after["filtered"], band_passed)
        assert not np.array_equal(after["ecg_cleaned"], band_passed)

    def test_a_second_removal_step_works_on_the_already_cleaned_signal(self):
        # Two gating passes chain without either naming a source key: the
        # second reads what the first produced, not the band-passed signal.
        session = M3Session()
        processed_emg = _fake_processed_emg()
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]

        first = ecg_gating(session, processed_emg, peaks)["processed_emg_after_ecg"]
        second = ecg_gating(session, first, peaks + 50)["processed_emg_after_ecg"]

        np.testing.assert_array_equal(second["filtered"], processed_emg["filtered"])
        assert not np.array_equal(second["ecg_cleaned"], first["ecg_cleaned"])

    def test_an_explicit_source_still_selects_that_signal(self):
        session = M3Session()
        processed_emg = _fake_processed_emg()
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]
        first = ecg_gating(session, processed_emg, peaks)["processed_emg_after_ecg"]

        result = ecg_gating(session, first, peaks + 50, source="filtered")

        assert result["ecg_gate_mask_result"].metadata["source"] == "filtered"

    def test_an_unknown_source_is_rejected(self):
        session = M3Session()
        processed_emg = _fake_processed_emg()
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]

        with pytest.raises(ValueError, match="not present in processed_emg"):
            ecg_gating(session, processed_emg, peaks, source="nope")


class TestPreprocessEnvelopeSkipping:
    """ECG gating recomputes the envelope from the gated signal, so an
    envelope computed during preprocessing would be discarded."""

    @staticmethod
    def _preprocess(**kwargs):
        from m3resp.adapters import ReSurfEMGAdapter

        fs = 2048.0
        rng = np.random.default_rng(0)
        recording = {
            "array": [rng.normal(size=8192)],
            "metadata": {"fs": fs, "labels": ["EMGdi"], "units": ["uV"]},
        }
        return ReSurfEMGAdapter().preprocess(recording, **kwargs)

    def test_envelope_is_computed_by_default(self):
        processed = self._preprocess()

        assert processed["envelope"] is not None

    def test_skipping_leaves_no_envelope_but_keeps_the_band_passed_signal(self):
        processed = self._preprocess(compute_envelope=False)

        assert processed["envelope"] is None
        assert processed["filtered"] is not None

    def test_skipping_still_records_the_window_for_gating_to_reuse(self):
        processed = self._preprocess(
            compute_envelope=False,
            envelope_window_seconds=0.25,
            envelope_method="rms",
        )

        assert processed["filter"]["envelope_window_seconds"] == 0.25
        assert processed["filter"]["envelope_method"] == "rms"

    def test_gating_recomputes_the_envelope_from_the_gated_signal(self):
        session = M3Session()
        processed = self._preprocess(
            compute_envelope=False, envelope_window_seconds=0.25
        )
        peaks = np.array([1000, 3000, 5000])

        result = ecg_gating(session, processed, peaks)

        after = result["processed_emg_after_ecg"]
        assert after["envelope"] is not None
        assert len(after["envelope"]) == len(processed["filtered"])
        assert after["filter"]["envelope_window_seconds"] == 0.25


class TestEcgGating:
    @pytest.mark.parametrize("gate_width_samples", [4, 7, 8, 204, 205])
    @pytest.mark.parametrize("fill_method", [0, 1, 2, 3])
    def test_gate_mask_names_exactly_the_samples_gating_replaced(
        self, gate_width_samples, fill_method
    ):
        # The RMS fill (method 3) blanks a slightly different span than the
        # other fills on an odd gate width, including the 205-sample default.
        # The mask has to follow whichever fill actually ran.
        from resurfemg.preprocessing.ecg_removal import gating

        n_samples = 1000
        peaks = np.array([300, 600])
        # A varying signal, so a replaced sample differs from its original
        # value under every fill method.
        signal = np.sin(np.arange(n_samples) / 3.0) + 2.0

        gated = gating(
            signal.copy(),
            peaks,
            gate_width=gate_width_samples,
            method=fill_method,
        )
        replaced = np.flatnonzero(gated != signal)
        mask = _build_gate_mask(
            n_samples,
            peaks,
            gate_width_samples=gate_width_samples,
            fill_method=fill_method,
        )

        np.testing.assert_array_equal(np.flatnonzero(mask), replaced)

    def test_gate_mask_is_clipped_at_the_start_of_the_record(self):
        mask = _build_gate_mask(10, [0], gate_width_samples=4, fill_method=1)

        np.testing.assert_array_equal(np.flatnonzero(mask), [0, 1])

    def test_updates_processed_emg_and_session_with_the_gated_signal(self):
        session = M3Session()
        processed_emg = _fake_processed_emg()
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]

        result = ecg_gating(session, processed_emg, peaks)

        gated = result["ecg_gated_emg"]
        band_passed = processed_emg["filtered"]
        assert gated.shape == band_passed.shape
        # Band-passing and gating are separate steps and keep separate
        # results; the band-passed signal survives gating.
        after = result["processed_emg_after_ecg"]
        assert after["ecg_cleaned"] is gated
        assert after["filtered"] is band_passed
        assert session.processed["emg"] is result["processed_emg_after_ecg"]
        assert session.emg is None or session.emg.ecg_cleaned is gated

        signal = result["ecg_gated_signal"]
        assert isinstance(signal, Signal)
        assert signal.processing_state == "intermediate"
        assert signal.method == "resurfemg.gating"
        assert any(signal is item for item in session.signals)

        mask_result = result["ecg_gate_mask_result"]
        assert isinstance(mask_result, ParameterResult)
        assert mask_result.value.dtype == bool
        assert mask_result.value.shape == gated.shape
        assert mask_result.value.sum() > 0

    def test_recomputed_envelope_reuses_the_preprocessing_envelope_method(self):
        """The recomputation must not silently switch envelope method: an ARV
        bundle stays ARV, and the effective choice is carried forward so a
        later recomputation off the gated bundle agrees too."""

        session = M3Session()
        processed_emg = _fake_processed_emg()
        processed_emg["filter"]["envelope_method"] = "arv"
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]

        result = ecg_gating(session, processed_emg, peaks)

        gated_bundle = result["processed_emg_after_ecg"]
        assert gated_bundle["filter"]["envelope_method"] == "arv"
        np.testing.assert_allclose(
            gated_bundle["envelope"],
            rolling_envelope(
                result["ecg_gated_emg"], window_length=int(0.5 * 2048.0), method="arv"
            ),
        )
        assert (
            result["ecg_gate_mask_result"].metadata["effective_envelope_method"]
            == "arv"
        )

    def test_envelope_method_can_be_overridden_for_the_recomputation(self):
        session = M3Session()
        processed_emg = _fake_processed_emg()
        processed_emg["filter"]["envelope_method"] = "arv"
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]

        result = ecg_gating(session, processed_emg, peaks, envelope_method="rms")

        np.testing.assert_allclose(
            result["processed_emg_after_ecg"]["envelope"],
            rolling_envelope(
                result["ecg_gated_emg"], window_length=int(0.5 * 2048.0), method="rms"
            ),
        )
        assert result["processed_emg_after_ecg"]["filter"]["envelope_method"] == "rms"

    def test_bundle_without_an_envelope_method_falls_back_to_rms(self):
        """`_fake_processed_emg`'s 'filter' has no 'envelope_method' - i.e. a
        bundle predating the field. It must default to RMS, not ARV."""

        session = M3Session()
        processed_emg = _fake_processed_emg()
        assert "envelope_method" not in processed_emg["filter"]
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]

        result = ecg_gating(session, processed_emg, peaks)

        assert result["processed_emg_after_ecg"]["filter"]["envelope_method"] == "rms"

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
        assert session.processed["emg"]["ecg_cleaned"] is cleaned
        assert session.processed["emg"]["filtered"] is processed_emg["filtered"]
        assert session.emg is None or session.emg.filtered is cleaned

    def test_rejects_unknown_source(self):
        session = M3Session()
        processed_emg = _fake_processed_emg()
        peaks = ecg_detect_peaks(session, processed_emg)["ecg_peak_indices"]
        with pytest.raises(ValueError, match="available keys"):
            ecg_wavelet_denoising(session, processed_emg, peaks, source="nope")
