"""Cutting one recording to a time window (`m3resp.modalities.emg` and
`m3resp.modalities.ventilator`). The EIT counterparts are tested in
tests/test_eit_slice.py, which needs a real eitprocessing Sequence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from m3resp.modalities import emg, ventilator
from m3resp.modalities.emg import EMGRecording


def _emg(n_samples: int = 100, fs: float = 10.0) -> EMGRecording:
    return EMGRecording(
        data={
            "array": np.arange(float(n_samples)).reshape(1, n_samples),
            "metadata": {"fs": fs},
        },
        path=Path("synthetic.npy"),
    )


def _ventilator(n_samples: int = 100, fs: float | None = 10.0) -> dict:
    metadata = {"fs": fs} if fs is not None else {}
    return {"array": np.vstack([np.arange(float(n_samples))] * 3), "metadata": metadata}


class TestEmgSampleWindow:
    def test_seconds_become_sample_numbers(self):
        window = emg.sample_window(_emg(), 1.0, 5.0)

        assert (window.start_index, window.end_index) == (10, 50)
        assert window.n_samples == 100
        assert window.shift_seconds == 1.0
        assert window.sample_frequency == 10.0

    def test_without_an_end_the_window_runs_to_the_end(self):
        window = emg.sample_window(_emg(), 1.0, None)

        assert window.end_seconds == 10.0
        assert window.end_index == 100

    @pytest.mark.parametrize(("start", "end"), [(-1.0, 5.0), (5.0, 5.0), (0.0, 11.0)])
    def test_a_window_outside_the_recording_is_refused(self, start, end):
        with pytest.raises(ValueError, match="start < end"):
            emg.sample_window(_emg(), start, end)


class TestEmgKeepSamples:
    def test_keeping_every_sample_changes_nothing(self):
        recording = _emg()
        before = recording.data["array"].copy()

        emg.keep_samples(recording, 0, 100)

        np.testing.assert_array_equal(recording.data["array"], before)

    def test_cuts_the_recording_in_place_and_refreshes_raw(self):
        recording = _emg()

        emg.keep_samples(recording, 10, 100)

        assert recording.data["array"].shape == (1, 90)
        assert recording.data["array"][0, 0] == 10.0
        assert recording.raw is recording.data["array"]


class TestVentilatorCutting:
    def test_sample_window_names_the_recording_in_errors(self):
        with pytest.raises(ValueError, match="'monitor'"):
            ventilator.sample_window(_ventilator(), 0.0, 11.0, name="monitor")

    def test_seconds_are_cut_off_both_ends_at_its_own_rate(self):
        payload = _ventilator(n_samples=200, fs=20.0)

        ventilator.cut_seconds_off_ends(payload, 1.0, 2.0)

        # 1 s = 20 samples off the start, 2 s = 40 samples off the end.
        assert payload["array"].shape == (3, 140)
        assert payload["array"][0, 0] == 20.0

    def test_cutting_every_sample_is_refused(self):
        payload = _ventilator()

        with pytest.raises(ValueError, match="remove every sample"):
            ventilator.cut_seconds_off_ends(payload, 6.0, 5.0)
        assert payload["array"].shape == (3, 100)

    def test_a_recording_without_a_sampling_rate_is_left_alone(self):
        payload = _ventilator(fs=None)

        ventilator.cut_seconds_off_ends(payload, 1.0, 1.0)

        assert payload["array"].shape == (3, 100)

    def test_a_time_axis_in_the_metadata_is_cut_too(self):
        payload = _ventilator()
        payload["metadata"]["time"] = 36528.5 + np.arange(100) / 10.0

        ventilator.keep_samples(payload, 10, 100)

        assert payload["metadata"]["time"][0] == pytest.approx(36529.5)
        assert len(payload["metadata"]["time"]) == 90
