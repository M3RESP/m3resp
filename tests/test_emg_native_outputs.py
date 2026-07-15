"""Stage 2 ReSurfEMG gap migration, Phase 2 (plan/stage2/
2_resurfemg_gap_migration_implementation_plan.md): native output contracts
for `emg.load`, `emg.moving_baseline`, and `emg.slopesum_baseline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from m3resp.core.session import M3Session
from m3resp.data import Signal
from m3resp.data.units import normalize_unit
from m3resp.workflows import run_pipeline
from m3resp.workflows.steps.emg import moving_baseline, slopesum_baseline

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


class TestEmgLoad:
    def test_writes_channel_major_native_signals_with_labels_units_and_fs(self):
        result = run_pipeline(
            {
                "name": "emg-load",
                "inputs": {"emg_file": str(EMG_PATH)},
                "steps": [{"uses": "emg.load", "with": {"file": "@emg_file"}}],
            }
        )

        recording = result.value("emg_recording")
        signals = result.value("raw_emg_signals")

        assert recording is not None
        assert isinstance(signals, list) and len(signals) == recording.raw.shape[0]
        for index, signal in enumerate(signals):
            assert isinstance(signal, Signal)
            assert signal.modality == "emg"
            assert signal.processing_state == "raw"
            assert signal.sample_frequency == recording.metadata["fs"]
            assert signal.unit == normalize_unit(recording.metadata["units"][index])
            assert signal.channel == recording.metadata["labels"][index]
            np.testing.assert_array_equal(signal.values, recording.raw[index])
            assert len(signal.time) == len(signal.values)
            assert signal.metadata["channel_index"] == index

        # Every native signal is added to the session's typed collection once.
        assert len(result.session.signals) == len(signals)

    def test_loader_options_must_be_a_mapping(self):
        with pytest.raises(TypeError, match="loader_options"):
            run_pipeline(
                {
                    "name": "emg-load-bad-options",
                    "inputs": {"emg_file": str(EMG_PATH)},
                    "steps": [
                        {
                            "uses": "emg.load",
                            "with": {"file": "@emg_file", "loader_options": "nope"},
                        }
                    ],
                }
            )

    def test_injected_loader_still_produces_native_signals(self):
        from m3resp.adapters.resurfemg_adapter import ReSurfEMGAdapter

        fake_array = np.asarray([[0.0, 1.0, 0.0, -1.0], [2.0, 2.0, 2.0, 2.0]])

        def fake_loader(path: str, **kwargs: Any) -> dict[str, Any]:
            return {
                "array": fake_array,
                "dataframe": None,
                "metadata": {
                    "fs": 1000.0,
                    "labels": ["a", "b"],
                    "units": ["uV", "uV"],
                },
            }

        session = M3Session(emg_adapter=ReSurfEMGAdapter(loader=fake_loader))
        result = run_pipeline(
            {
                "name": "emg-load-injected",
                "inputs": {"emg_file": "unused.Poly5"},
                "steps": [{"uses": "emg.load", "with": {"file": "@emg_file"}}],
            },
            session=session,
        )

        signals = result.value("raw_emg_signals")
        assert [s.channel for s in signals] == ["a", "b"]
        np.testing.assert_array_equal(signals[0].values, fake_array[0])


def _fake_processed_emg(*, fs: float = 1000.0, n_samples: int = 5000) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    envelope = np.abs(rng.normal(size=n_samples))
    return {
        "envelope": envelope,
        "fs": fs,
        "channel": 0,
        "metadata": {"labels": ["EMGdi"], "units": ["uV"]},
    }


class TestMovingBaselineNativeOutput:
    def test_writes_a_derived_signal_matching_the_envelope_time_axis(self):
        session = M3Session()
        processed_emg = _fake_processed_emg()

        result = moving_baseline(
            session,
            processed_emg,
            window_seconds=1.0,
            step_seconds=0.1,
            percentile=33.0,
        )

        baseline = result["baseline"]
        signal = result["baseline_signal"]
        assert isinstance(signal, Signal)
        np.testing.assert_array_equal(signal.values, baseline)
        assert len(signal.time) == len(processed_emg["envelope"])
        assert signal.sample_frequency == processed_emg["fs"]
        assert signal.channel == "EMGdi"
        assert signal.unit == normalize_unit("uV")
        assert signal.processing_state == "derived"
        assert signal.derived_from == "processed"
        assert signal.method == "resurfemg.moving_baseline"
        assert signal.metadata["effective_window_samples"] == 1000
        assert signal.metadata["effective_step_samples"] == 100
        assert any(signal is item for item in session.signals)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"window_seconds": 0},
            {"window_seconds": -1.0},
            {"step_seconds": 0},
            {"percentile": -1.0},
            {"percentile": 101.0},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs):
        session = M3Session()
        with pytest.raises(ValueError):
            moving_baseline(session, _fake_processed_emg(), **kwargs)


class TestSlopesumBaselineNativeOutput:
    def test_writes_baseline_and_running_mean_std_signals(self):
        session = M3Session()
        processed_emg = _fake_processed_emg()

        result = slopesum_baseline(
            session,
            processed_emg,
            window_seconds=1.0,
            step_seconds=0.1,
            percentile=33.0,
            augmented_percentile=25.0,
        )

        for key in (
            "baseline_signal",
            "baseline_running_mean_signal",
            "baseline_running_std_signal",
        ):
            signal = result[key]
            assert isinstance(signal, Signal)
            assert len(signal.time) == len(processed_emg["envelope"])
            assert signal.processing_state == "derived"
            assert signal.method == "resurfemg.slopesum_baseline"
            assert any(signal is item for item in session.signals)

        # The compatibility mapping keeps the pandas series; the native
        # detail stays NumPy-only and does not cross that boundary.
        detail = result["slopesum_baseline_detail"]
        native_detail = result["slopesum_baseline_native_detail"]
        assert "series" in detail
        assert "series" not in native_detail
        assert set(native_detail) == {"running_mean", "running_std"}
        np.testing.assert_array_equal(
            native_detail["running_mean"], result["baseline_running_mean_signal"].values
        )

    def test_default_moving_average_and_percentile_window_seconds_match_prior_defaults(
        self,
    ):
        # Regression: before Phase 2, these were hardcoded as
        # `max(1, int(fs // 2))` / `max(1, int(fs))` samples. The new
        # seconds-based defaults (0.5s / 1.0s) must reproduce the same
        # effective sample counts for an integer-valued fs.
        session = M3Session()
        processed_emg = _fake_processed_emg(fs=2048.0)

        result = slopesum_baseline(session, processed_emg)

        metadata = result["baseline_signal"].metadata
        assert metadata["effective_moving_average_samples"] == 2048 // 2
        assert metadata["effective_percentile_window_samples"] == 2048

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"moving_average_seconds": 0},
            {"percentile_window_seconds": -1.0},
            {"augmented_percentile": 200.0},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs):
        session = M3Session()
        with pytest.raises(ValueError):
            slopesum_baseline(session, _fake_processed_emg(), **kwargs)
